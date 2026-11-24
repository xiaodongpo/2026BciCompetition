"""严格有界的微伏特征、固定归一化和异构推理。

每次预测只读取当前及过去1500个原始样本。无在线统计、历史评分或滤波状态。
滤波、线性去趋势全部在已经到达的2/4/6秒子窗内重算。
"""
from types import SimpleNamespace
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch, lfilter, lfilter_zi
from .features import HandcraftedFeaturizer

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
