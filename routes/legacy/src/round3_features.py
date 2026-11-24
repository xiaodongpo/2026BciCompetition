"""Round-three representations use one already-arrived 1500-sample window."""
import numpy as np
from src.features import extract_features
from src.round2_features import validate,filtered,filterbank_features


def normalize_gain(x):
    x=validate(x);z=filtered(x)
    median=np.median(z,axis=-1,keepdims=True)
    scale=np.maximum(1.4826*np.median(abs(z-median),axis=-1,keepdims=True),1e-6)
    return (x-x.mean(axis=-1,keepdims=True))/scale*20


def get_features(x,kind):
    x=validate(x)
    if kind=='base':return filterbank_features(x)
    if kind=='normalized':return filterbank_features(normalize_gain(x))
    if kind=='multiscale':
        z=filtered(x)
        return np.concatenate([filterbank_features(x),extract_features(z[:,-500:]),extract_features(z[:,-750:])],axis=1).astype('float32')
    raise ValueError(kind)
