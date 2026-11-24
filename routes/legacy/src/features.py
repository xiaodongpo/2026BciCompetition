"""All operations are functions of a single trailing <=6 s window."""
import numpy as np
from scipy.signal import welch,firwin,lfilter,detrend

BANDS=[(.5,4),(4,8),(8,13),(13,20),(20,30),(30,45),(45,55),(55,90)]

def extract_features(x, fs=250, mode='full'):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim==1:x=x[None,:]
    if x.shape[1]>1500 or x.shape[1]<250: raise ValueError('Window outside 1-6 s')
    if mode in ('fir','fir_2_40','fir_4_40'):
        bands={'fir':[1,45],'fir_2_40':[2,40],'fir_4_40':[4,40]}
        taps=firwin(251,bands[mode],pass_zero=False,fs=fs)
        x=lfilter(taps,[1.],x-x.mean(axis=-1,keepdims=True),axis=-1)[:,250:]
    elif mode=='detrend':
        x=detrend(x,axis=-1,type='linear')
    elif mode not in ('full','relative'):
        raise ValueError(f'Unknown features: {mode}')
    z=x-x.mean(axis=-1,keepdims=True)
    f,p=welch(z,fs=fs,nperseg=min(500,x.shape[1]),noverlap=min(250,x.shape[1]//2),axis=-1,detrend='constant')
    df=f[1]-f[0]
    band=np.stack([p[:,(f>=a)&(f<b)].sum(axis=1)*df for a,b in BANDS],axis=1)
    total=band[:,:6].sum(axis=1,keepdims=True)+1e-12
    rel=band/(total+1e-12)
    if mode=='relative': return np.log10(rel[:,:6]+1e-12).astype(np.float32)
    fine=np.stack([p[:,(f>=a)&(f<a+2)].sum(axis=1)*df for a in range(1,45,2)],axis=1)
    var=z.var(axis=1)+1e-12; d=np.diff(z,axis=1); dd=np.diff(d,axis=1)
    mobility=np.sqrt(d.var(axis=1)/var)
    complexity=np.sqrt(dd.var(axis=1)/(d.var(axis=1)+1e-12))/(mobility+1e-12)
    norm=z/np.sqrt(var[:,None])
    q=p[:,(f>=.5)&(f<45)]; q=q/(q.sum(axis=1,keepdims=True)+1e-12)
    stats=np.stack([np.log10(var),np.log10(np.ptp(z,axis=1)+1e-12),mobility,complexity,
                    np.mean(norm**3,axis=1),np.mean(norm**4,axis=1),
                    -(q*np.log(q+1e-12)).sum(axis=1),np.mean(np.abs(d),axis=1)/np.sqrt(var)],axis=1)
    return np.nan_to_num(np.concatenate([np.log10(band+1e-12),np.log10(rel+1e-12),
                                        np.log10(fine+1e-12),stats],axis=1)).astype(np.float32)
