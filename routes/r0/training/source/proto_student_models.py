"""固定收缩原型与六秒学生。所有部署统计量固定，只读当前已到达六秒。"""
import numpy as np
import torch
from torch import nn
from model_representation import EnergyFusionCNN
from strict6s_pipeline import current_quality,extract_batch,bounded_filter,transform_features,transform_wave,sigmoid


def representation(base,wave):
    fm=base.features(wave)
    lp=base.energy_filters(wave).square().mean(-1).clamp_min(1e-8).log()
    return torch.cat([fm.mean(-1),fm.std(-1,unbiased=False),base.energy_projection(lp)],1)


class SixSecondStudent(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=EnergyFusionCNN()
        # 原分类头不参与学生前向，避免保存未使用的可训练参数。
        self.encoder.classifier=nn.Identity()
        self.hand=nn.Sequential(nn.LayerNorm(165),nn.Linear(165,32),nn.ELU(),nn.Dropout(.15))
        self.head=nn.Sequential(nn.Linear(176,64),nn.ELU(),nn.Dropout(.3),nn.Linear(64,2))

    def forward(self,wave,features):
        assert wave.shape[1:]==(1,1500) and features.shape[1]==165
        return self.head(torch.cat([representation(self.encoder,wave),self.hand(features)],1))


def fit_prototype(x,y,weights,rho=.1):
    """训练折加权标准化、类内协方差收缩；不更新测试原型。"""
    x=np.asarray(x,np.float64);w=weights.astype(np.float64);w=w/w.sum()
    mean=np.sum(x*w[:,None],0);sd=np.sqrt(np.sum((x-mean)**2*w[:,None],0)).clip(1e-5)
    z=(x-mean)/sd
    m=np.stack([np.average(z[y==c],axis=0,weights=w[y==c]) for c in [0,1]])
    residual=z-m[y]
    cov=(residual*w[:,None]).T@residual
    target=max(float(np.trace(cov)/len(mean)),1e-6)
    covariance=(1-rho)*cov+rho*target*np.eye(len(mean))
    direction=np.linalg.solve(covariance,m[1]-m[0]);mid=(m[0]+m[1])/2
    score=(z-mid)@direction
    scale=max(float(np.sqrt(np.average((score-np.average(score,weights=w))**2,weights=w))),1e-5)
    return dict(mean=mean,sd=sd,direction=direction,mid=mid,score_scale=np.array(scale),rho=np.array(rho))


def prototype_score(x,params):
    return (((x-params['mean'])/params['sd']-params['mid'])@params['direction'])/float(params['score_scale'])


class StudentDecoder:
    def __init__(self,model,scaler,device='cuda'):
        self.model=model.to(device).eval();self.scaler=scaler;self.device=device;self.buffer=np.empty(0)
    def reset(self):self.buffer=np.empty(0)
    @torch.no_grad()
    def predict_window(self,raw):
        valid,reason=current_quality(raw)
        if not valid:return dict(valid=False,score=.5,reason=reason)
        raw=np.asarray(raw,np.float64)
        f,_=extract_batch(raw[None],np.array([True]));w=bounded_filter(raw[None]).astype(np.float32)
        xx=torch.from_numpy(transform_wave(w,self.scaler)).to(self.device)
        ff=torch.from_numpy(transform_features(f,self.scaler)).to(self.device)
        a=self.model(xx,ff);score=float((a[:,1]-a[:,0]).sigmoid().item())
        return dict(valid=True,score=score,label=int(score>=.5))
    def push(self,chunk):
        self.buffer=np.r_[self.buffer,np.asarray(chunk,np.float64).ravel()][-1500:]
        return self.predict_window(self.buffer)


class StaticPrototypeDecoder:
    def __init__(self,models,tree,scaler,prototypes,device='cuda'):
        self.models=[m.to(device).eval() for m in models];self.tree=tree;self.scaler=scaler
        self.prototypes=prototypes;self.device=device;self.buffer=np.empty(0)
    @torch.no_grad()
    def predict_window(self,raw):
        valid,reason=current_quality(raw)
        if not valid:return dict(valid=False,score=.5,reason=reason)
        f,w=extract_batch(np.asarray(raw)[None],np.array([True]))
        xx=torch.from_numpy(transform_wave(w,self.scaler)).to(self.device)
        ff=transform_features(f,self.scaler);zz=[];pp=[]
        for model,params in zip(self.models,self.prototypes):
            emb=representation(model,xx);a=model.classifier(emb)
            zz.append(float((a[:,1]-a[:,0]).item()));pp.append(float(prototype_score(emb.cpu().numpy(),params)[0]))
        tree=float(self.tree.predict(ff,raw_score=True,num_threads=1)[0])
        base=.5*tree+.5*np.mean(zz);delta=.25*np.tanh(np.mean(pp))
        score=float(sigmoid(base+delta));return dict(valid=True,score=score,label=int(score>=.5))
    def push(self,chunk):
        self.buffer=np.r_[self.buffer,np.asarray(chunk,np.float64).ravel()][-1500:]
        return self.predict_window(self.buffer)


def audit_decoder(make,raw):
    a,b,c=make(),make(),make()
    pa=a.push(np.r_[np.zeros(2000),raw]);pb=b.push(np.r_[np.full(2000,1e8),raw])
    assert pa['valid'] and abs(pa['score']-pb['score'])<2e-6
    for chunk in np.array_split(raw,17):pc=c.push(chunk)
    assert abs(pc['score']-pa['score'])<2e-6
    def trace(future):
        decoder=make();return [decoder.push(ch)['score'] for ch in np.array_split(np.r_[raw,future],7)]
    assert np.allclose(trace(np.zeros(250))[:6],trace(np.full(250,1e8))[:6],atol=2e-6,rtol=0)
    return dict(status='PASS',older_than_6s_invariance=True,chunking_invariance=True,future_invariance=True,
                deployment_online_statistics=False,max_signal_seconds=6)
