"""Training-only gain augmentation; deployment remains one 125-feature window."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from scripts.round2_train import sample_weights

def training_arrays(x,meta):
    if x.ndim!=2 or x.shape[1] not in (125,375):raise ValueError('Expected original or three feature views.')
    n=x.shape[1]//125;w=sample_weights(meta)
    return np.concatenate(np.split(x,n,axis=1)),np.tile(meta.y.to_numpy(),n),np.tile(w/n,n)

def fit(c,x,meta):
    assert (meta.groupby('record').y.nunique()==2).all()
    if 'width' in c:assert x.shape[1]==(125 if c['width']==0 else 375)
    scaler=StandardScaler().fit(x[:,:125]);xx,y,w=training_arrays(x,meta)
    classifier=LogisticRegression(C=c['C'],max_iter=2000,random_state=20261104)
    with threadpool_limits(limits=4):classifier.fit(scaler.transform(xx),y,sample_weight=w)
    return make_pipeline(scaler,classifier)

def predict(model,x):
    with threadpool_limits(limits=4):return model.predict_proba(x[:,:125])[:,1]
