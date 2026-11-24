"""严格有界的微伏特征、固定归一化和异构推理。

每次预测只读取当前及过去1500个原始样本。无在线统计、历史评分或滤波状态。
滤波、线性去趋势全部在已经到达的2/4/6秒子窗内重算。
"""
from types import SimpleNamespace
from collections import deque
import numpy as np
import torch
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch, lfilter, lfilter_zi
from strict6s_handcrafted import HandcraftedFeaturizer

FS = 250
GROUPS = ['band_abs','band_rel','ratios','shape','nonstat','time','band_tail','transient']


def build_featurizers():
    result=[]
    for seconds in [2,4,6]:
        cfg=SimpleNamespace(quality=None,features=SimpleNamespace(groups=GROUPS,min_segments=1,
                            segment_sec=2.0,segment_hop_sec=0.5),
                            stream=SimpleNamespace(window_samples=seconds*FS))
        result.append(HandcraftedFeaturizer(cfg))
    return result


def feature_names():
    return ['s%d_%s'%(seconds,name) for seconds,f in zip([2,4,6],build_featurizers()) for name in f.names]


def current_quality(window):
    """仅检查当前原始6秒窗；阈值来自当前数据的ADC量程及已审计补零编码。

    不向模型提供缺失长度、窗口可用比例或记录编号。
    原始数据的0.024µV补零编码仅适用于这套BDF增益，不能用于标准化后的信号。
    """
    x=np.asarray(window,dtype=np.float64)
    if x.shape!=(1500,) or not np.isfinite(x).all():return False,'warmup_or_nonfinite'
    if np.any(np.abs(x)<0.05):return False,'audited_adc_gap_code'
    if np.any(np.abs(x)>=403000):return False,'adc_rail'
    if np.any(np.abs(np.diff(x))>20000):return False,'large_jump'
    same=np.abs(np.diff(x))<=0.04808104324823876/4
    d=np.diff(np.r_[False,same,False].astype(np.int8))
    spans=np.flatnonzero(d==-1)-np.flatnonzero(d==1)
    if len(spans) and spans.max()+1>=250:return False,'flatline_1s'
    return True,'valid'


def bounded_filter(windows):
    """独立窗口的因果1–50Hz带通、50Hz陷波及窗口内线性去趋势。

    每次都从当前窗首样本初始化状态，不读取前一窗口滤波状态。
    去趋势使用的整个窗口均已到达，判断对应窗口末端。
    """
    x=np.asarray(windows,dtype=np.float64)
    if x.ndim==2:x=x[:,None,:]
    assert x.ndim==3 and x.shape[1]==1
    sos=butter(4,[1,50],btype='bandpass',fs=FS,output='sos')
    zi=sosfilt_zi(sos)[:,None,None,:]*x[None,...,0,None]
    y,_=sosfilt(sos,x,zi=zi,axis=-1)
    b,a=iirnotch(50,30,fs=FS)
    y,_=lfilter(b,a,y,zi=lfilter_zi(b,a)[None,None,:]*y[...,0,None],axis=-1)
    y=y-y.mean(axis=-1,keepdims=True)
    t=np.arange(y.shape[-1],dtype=np.float64)-(y.shape[-1]-1)/2
    y=y-(y@t)[...,None]/(t@t)*t
    return y


def extract_batch(raw_windows, valid):
    """输出固定165维特征及物理微伏4秒波形；无效窗口保持零但不用于拟合。"""
    raw=np.asarray(raw_windows,dtype=np.float64)
    n=len(raw);features=np.zeros((n,len(feature_names())),np.float32)
    wave=np.zeros((n,1,1000),np.float32)
    featurizers=build_featurizers()
    ids=np.flatnonzero(valid)
    for begin in range(0,len(ids),256):
        ix=ids[begin:begin+256];pieces=[]
        for seconds,f in zip([2,4,6],featurizers):
            filtered=bounded_filter(raw[ix,-seconds*FS:])[:,0]
            if seconds==4:wave[ix,0]=filtered.astype(np.float32)
            pieces.append(np.stack([f(x) for x in filtered]))
        features[ix]=np.concatenate(pieces,axis=1).astype(np.float32)
    assert np.isfinite(features).all() and np.isfinite(wave).all()
    return features,wave


def fit_static_scaler(features,wave,weights,record_ids):
    """只在当前训练折拟合参数；微伏波形共享一个固定尺度，保留跨窗幅值关系。"""
    w=np.asarray(weights,np.float64);f=np.asarray(features,np.float64)
    mean=np.average(f,axis=0,weights=w)
    sd=np.sqrt(np.average((f-mean)**2,axis=0,weights=w))
    sd=np.maximum(sd,1e-6)
    record_scales=[]
    for rid in sorted(set(record_ids)):
        x=wave[np.asarray(record_ids)==rid,0]
        med=np.median(x,axis=1,keepdims=True)
        mad=1.4826*np.median(np.abs(x-med),axis=1)
        record_scales.append(float(np.median(mad)))
    return dict(mean=mean.tolist(),sd=sd.tolist(),wave_scale_uv=max(float(np.median(record_scales)),1.0),
                fit_records=sorted(set(record_ids)),scope='training fold only; frozen during inference')


def transform_features(f,scaler):
    return np.clip((f-np.asarray(scaler['mean']))/np.asarray(scaler['sd']),-8,8).astype(np.float32)


def transform_wave(wave,scaler):
    return np.clip(wave/scaler['wave_scale_uv'],-20,20).astype(np.float32)


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-60,60)))


class StrictSixSecondDecoder:
    """无标签分块推理接口；原始缓存之外没有跨判断状态。

    同一预测时刻，不管6秒以前的信号如何变化，输出都应相同。
    """
    def __init__(self,cnn_models,tree,scaler,device='cuda'):
        self.models=[m.to(device).eval() for m in cnn_models]
        self.tree=tree;self.scaler=scaler;self.device=device
        self.buffer=np.empty(0,np.float64)

    def reset(self):self.buffer=np.empty(0,np.float64)

    @classmethod
    def load(cls,fold_directory,device='cuda'):
        import json
        from pathlib import Path
        import lightgbm as lgb
        from model_representation import EnergyFusionCNN
        directory=Path(fold_directory)
        scaler=json.loads((directory/'scaler.json').read_text())
        models=[]
        for seed in [42,2024,3407]:
            checkpoint=torch.load(directory/('seed%d'%seed)/'epoch_50.pth',map_location='cpu')
            assert checkpoint['epoch']==50
            model=EnergyFusionCNN();model.load_state_dict(checkpoint['model']);models.append(model)
        tree=lgb.Booster(model_file=str(directory/'lightgbm_round50.txt'))
        return cls(models,tree,scaler,device=device)

    @torch.no_grad()
    def predict_window(self,raw):
        raw=np.asarray(raw,dtype=np.float64)
        good,reason=current_quality(raw)
        if not good:return dict(valid=False,reason=reason,score=0.5,label=1)
        f,w=extract_batch(raw[None],np.array([True]))
        zz=transform_features(f,self.scaler)
        xx=torch.from_numpy(transform_wave(w,self.scaler)).to(self.device)
        packed=torch.cat([xx[:,0],xx.new_zeros((1,6))],dim=1)
        logits=[]
        for model in self.models:
            a=model(packed);logits.append(float((a[:,1]-a[:,0]).item()))
        tree=float(self.tree.predict(zz,raw_score=True,num_threads=1)[0])
        net=float(np.mean(logits));score=float(sigmoid(0.5*tree+0.5*net))
        return dict(valid=True,score=score,label=int(score>=0.5),tree_logit=tree,cnn_logit=net)

    def push(self,chunk):
        chunk=np.asarray(chunk,dtype=np.float64).ravel()
        assert len(chunk)>0
        self.buffer=np.concatenate([self.buffer,chunk])[-1500:]
        return self.predict_window(self.buffer)
