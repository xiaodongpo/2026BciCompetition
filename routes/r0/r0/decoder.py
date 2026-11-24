"""从原始微伏EEG分块输入到R0评分的完整因果接口。

仅保留最近1500点原始信号；统计量允许使用当前记录更早的无标签数据。
每条独立记录调用reset。状态切换不触发reset，不读取事件或测试标签。
"""
from pathlib import Path
from collections import deque
import json
import numpy as np
import torch
import lightgbm as lgb
from .model import EnergyFusionCNN
from .preprocess import transform
from .signal import current_quality,extract_batch


class RunningMoments:
    def __init__(self,mean,sd,prior,half):
        self.mean=mean.copy();self.weight=float(prior)
        self.m2=(sd**2+1e-12)*prior
        self.decay=1. if half is None else .5**(1/half)

    def z(self,x):
        return (x-self.mean)/np.sqrt(self.m2/self.weight)

    def update(self,x):
        self.weight=self.weight*self.decay+1;self.m2*=self.decay
        d=x-self.mean;self.mean+=d/self.weight;self.m2+=d*(x-self.mean)


class R0Decoder:
    def __init__(self,model_dir,device='cpu',threads=1):
        root=Path(model_dir);self.config=json.loads((root/'config.json').read_text())
        if self.config['model']!='R0_LAG_CUMULATIVE_PROTO':raise ValueError('Not an R0 model bundle')
        self.device=torch.device(device);torch.set_num_threads(threads)
        self.models=[]
        for seed in self.config['seeds']:
            c=torch.load(root/('energy_seed%d.pth'%seed),map_location='cpu')
            m=EnergyFusionCNN();m.load_state_dict(c['model'],strict=True);self.models.append(m.to(self.device).eval())
        self.tree=lgb.Booster(model_file=str(root/'lightgbm.txt'))
        with np.load(root/'statistics.npz',allow_pickle=False) as z:
            self.feature_mean=z['feature_mean'];self.feature_sd=z['feature_sd']
            self.m0=z['m0'];self.m1=z['m1'];self.variance=z['variance'];self.score_scale=z['score_scale']
        assert self.m0.shape==(3,144) and self.feature_mean.shape==(165,)
        self.freeze_adaptation=False;self.reset()

    def reset(self):
        self.buffer=np.empty(0,np.float64);self.mask=np.empty(0,bool)
        self.count=0;self.next_end=1000;self.probs=deque(maxlen=3)
        c=self.config
        self.moments=[RunningMoments(self.feature_mean,self.feature_sd,c['feature_prior'],h) for h in c['feature_halves']]
        self.logit_sum=0.;self.logit_n=0
        self.n0=self.n1=float(c['kappa']);self.s0=self.m0*c['kappa'];self.s1=self.m1*c['kappa']

    @staticmethod
    def _logit(p):
        p=np.clip(p,1e-6,1-1e-6);return float(np.log(p/(1-p)))

    @staticmethod
    def _quality4(x):
        if x.shape!=(1000,) or not np.isfinite(x).all():return False
        if np.any(np.abs(x)<.05) or np.any(np.abs(x)>=403000):return False
        same=np.abs(np.diff(x))<=.04808104324823876/4
        d=np.diff(np.r_[False,same,False].astype(np.int8));spans=np.flatnonzero(d==-1)-np.flatnonzero(d==1)
        return not (len(spans) and spans.max()+1>=250)

    @torch.no_grad()
    def _decide(self,end):
        x4=self.buffer[-1000:];c=self.config
        invalid=dict(end_sample=end,time_sec=end/250.,score=.5,state=1,valid=False,reason='invalid_4s')
        if not self.mask[-1000:].all() or not self._quality4(x4):return invalid
        _,wave,_,_,_,keep=transform(x4[None,None].astype(np.float32))
        if not keep[0]:return invalid
        xx=torch.from_numpy(wave[0]).to(self.device)
        embeddings=[];pp=[]
        for m in self.models:
            z=m.embedding(xx);a=m.classifier(z)
            embeddings.append(z[0].cpu().numpy().astype(np.float64));pp.append(float(a.softmax(-1)[0,1]))
        embedding=np.stack(embeddings);p=float(np.mean(pp));start=end-1000
        if self.probs and self.probs[-1][0]+250!=start:self.probs.clear()
        self.probs.append((start,p));assert end-self.probs[0][0]<=1500
        raw_logit=self._logit(np.mean([v for _,v in self.probs]))
        energy=raw_logit-self.logit_sum/(c['center_prior']+self.logit_n)
        fused=energy;tree_logit=None
        good6=len(self.buffer)==1500 and self.mask.all() and current_quality(self.buffer)[0]
        feature=None
        if good6:
            ff,_=extract_batch(self.buffer[None],np.array([True]));feature=ff[0].astype(np.float64)
            zz=np.concatenate([s.z(feature) for s in self.moments])
            tp=float(self.tree.predict(zz[None],num_threads=1)[0]);tree_logit=self._logit(tp)
            fused=c['tree_weight']*tree_logit+(1-c['tree_weight'])*energy
        # 所有原型评分使用更新前的类中心；方差与尺度来自训练数据且固定。
        a,b=self.s0/self.n0,self.s1/self.n1
        current=np.sum(((b-a)/self.variance)*(embedding-.5*(a+b)),axis=1)
        source=np.sum(((self.m1-self.m0)/self.variance)*(embedding-.5*(self.m0+self.m1)),axis=1)
        correction=float(np.mean((current-source)/self.score_scale))
        final_logit=fused+c['alpha']*correction
        score=float(1/(1+np.exp(-np.clip(final_logit,-60,60))))
        result=dict(end_sample=end,time_sec=end/250.,score=score,state=int(score>=c['threshold']),valid=True,
            reason='valid',tree_valid=bool(good6),energy_logit=float(energy),tree_logit=tree_logit,
            prototype_delta=correction,fused_logit=float(fused),final_logit=float(final_logit))
        # 先输出所需量，再更新历史统计；不使用final_logit反馈伪标签。
        if not self.freeze_adaptation:
            if feature is not None:
                for s in self.moments:s.update(feature)
            self.logit_sum+=raw_logit;self.logit_n+=1
            if fused>c['tau']:self.n1+=1;self.s1+=embedding
            elif fused<-c['tau']:self.n0+=1;self.s0+=embedding
        return result

    def push(self,chunk,valid_mask=None):
        """chunk形状[N]或[1,N]，单位µV、250Hz；可附同长度采集有效掩码。"""
        x=np.asarray(chunk,dtype=np.float64)
        if x.ndim==2 and x.shape[0]==1:x=x[0]
        if x.ndim!=1:raise ValueError('Expected one EEG channel [N] or [1,N]')
        mask=np.isfinite(x) if valid_mask is None else np.asarray(valid_mask,bool).reshape(-1)&np.isfinite(x)
        if len(mask)!=len(x):raise ValueError('valid_mask length mismatch')
        rows=[];at=0
        while at<len(x):
            take=min(self.next_end-self.count,len(x)-at)
            self.buffer=np.r_[self.buffer,x[at:at+take]][-1500:]
            self.mask=np.r_[self.mask,mask[at:at+take]][-1500:]
            self.count+=take;at+=take
            if self.count==self.next_end:
                rows.append(self._decide(self.count));self.next_end+=250
        return rows
