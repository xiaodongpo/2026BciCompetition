"""Pure single-window representations: no identities, clock features or persistent statistics."""
from functools import lru_cache
import numpy as np
from scipy.signal import firwin,lfilter,periodogram,resample_poly
from src.features import extract_features

@lru_cache(maxsize=16)
def taps(low,high):return firwin(251,[low,high],pass_zero=False,fs=250)

def validate(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim==1:x=x[None,:]
    if x.ndim!=2 or x.shape[1]!=1500 or not np.isfinite(x).all():raise ValueError('Expected finite [batch,1500] raw microvolt windows')
    return x

def filtered(x,low=1,high=45):
    x=validate(x);return lfilter(taps(low,high),[1.],x-x.mean(-1,keepdims=True),axis=-1)[:,250:]

def robust_signal(x):
    z=filtered(x);median=np.median(z,axis=-1,keepdims=True)
    scale=np.maximum(1.4826*np.median(abs(z-median),axis=-1,keepdims=True),1.)
    return np.clip(z,median-6*scale,median+6*scale),scale

def robust_features(x):
    x=validate(x);z,scale=robust_signal(x)
    # Five non-overlapping 1s frames stay wholly within the current 6s raw support.
    frames=z.reshape(len(z),5,250)
    f,p=periodogram(frames,fs=250,window='hann',detrend='linear',axis=-1)
    bands=[(1,4),(4,8),(8,13),(13,20),(20,30),(30,45),(1,45)]
    powers=np.stack([p[:,:,(f>=a)&(f<b)].sum(-1) for a,b in bands],axis=-1)
    log=np.log10(powers+1e-12);rel=np.log10((powers[:,:,:6]+1e-12)/(powers[:,:,-1:]+1e-12))
    feats=[extract_features(z),np.median(log,axis=1),np.std(log,axis=1),np.quantile(log,.9,axis=1)-np.quantile(log,.1,axis=1),np.median(rel,axis=1)]
    # Differences describe variability within the already-arrived buffer, not elapsed trial time.
    feats.extend([np.mean(np.abs(np.diff(log,axis=1)),axis=1),np.log10(scale)])
    return np.concatenate(feats,axis=1).astype('float32')

def log_covariance(rows):
    rows=rows-rows.mean(-1,keepdims=True)
    cov=rows@np.swapaxes(rows,-1,-2)/rows.shape[-1]
    trace=np.maximum(np.trace(cov,axis1=-2,axis2=-1),1e-12)
    n=cov.shape[-1];cov=.95*cov/trace[:,None,None]+.05*np.eye(n)[None,:,:]/n
    val,vec=np.linalg.eigh(cov);log=(vec*np.log(np.maximum(val,1e-12))[:,None,:])@np.swapaxes(vec,-1,-2)
    i,j=np.triu_indices(n);v=log[:,i,j];v[:,i!=j]*=np.sqrt(2)
    return np.concatenate((v,np.log10(trace)[:,None]),axis=1)

def filterbank_features(x):
    x=validate(x)
    # These are frequency transforms of ONE channel, not simulated spatial electrodes.
    rows=np.stack([filtered(x,a,b) for a,b in [(1,4),(4,8),(8,13),(13,20),(20,30),(30,45)]],axis=1)
    z,_=robust_signal(x);lags=[0,1,2,4,8,16]
    lagged=np.stack([z[:,16-lag:len(z[0])-lag] for lag in lags],axis=1)
    return np.concatenate([robust_features(x),log_covariance(rows),log_covariance(lagged)],axis=1).astype('float32')

def singlem_input(x):
    z,scale=robust_signal(x)
    # Hardware-independent amplitude mapping, fixed beforehand: robust scale maps to 0.2.
    z=np.clip(z/scale,-6,6)*.2
    z=resample_poly(z,64,125,axis=-1).astype('float32')
    return z.reshape(len(z),5,128)

def get_features(x,kind):
    if kind=='legacy':return extract_features(x,mode='fir')
    if kind=='robust':return robust_features(x)
    if kind=='filterbank':return filterbank_features(x)
    raise ValueError(kind)
