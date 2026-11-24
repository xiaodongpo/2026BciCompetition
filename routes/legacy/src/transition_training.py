"""Offline supervision policies. Nothing here is consumed by deployed inference."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


def policy(meta, config, progress=1.):
    y=meta.y.to_numpy(dtype=float);t=meta.t.to_numpy();switch=meta.valid_switch.to_numpy(bool)
    T=np.where(y==1,config.get('T_task',0),config.get('T_relax',0)).astype(float)
    frac=np.ones(len(meta));active=switch & (T>0)
    frac[active]=np.clip(t[active]/T[active],0,1)
    q=y.copy();mult=np.ones(len(meta));kind=config['kind']
    if kind=='pure':mult=meta.pure.to_numpy(dtype=float)
    elif kind=='drop':mult=(~active | (t>=T)).astype(float)
    elif kind=='weight':
        floor=np.where(y==1,config.get('min_task',.25),config.get('min_relax',.25))
        mult=floor+(1-floor)*frac
    elif kind=='soft':q=np.where(y==1,.5+.5*frac,.5-.5*frac)
    elif kind=='curriculum':mult=np.where(active & (t<T),progress,1.)
    elif kind!='hard':raise ValueError(kind)
    # Preserve equal total weight per record and official class AFTER supervision changes.
    frame=meta[['record','y']].copy();frame['mult']=mult
    totals=frame.groupby(['record','y']).mult.transform('sum').to_numpy()
    if (totals<=0).any():raise ValueError('Policy removed an entire record/class.')
    w=mult/totals;w*=np.count_nonzero(mult)/w.sum()
    return q,w


def sample_summary(meta,c):
    q,w=policy(meta,c);keep=w>0;y=meta.y.to_numpy()
    return dict(total_rows=len(meta),retained_rows=int(keep.sum()),relax_rows=int((keep&(y==0)).sum()),task_rows=int((keep&(y==1)).sum()),task_fraction=float(y[keep].mean()),effective_weight=float(w.sum()),soft_targets=int(np.sum((q>0)&(q<1))),T_task=c.get('T_task',0),T_relax=c.get('T_relax',0))


def fit_lr(x,meta,c):
    q,w=policy(meta,c);keep=w>0;scaler=StandardScaler().fit(x[keep]);z=scaler.transform(x[keep]);q=q[keep];w=w[keep]
    if c['kind']=='soft':
        xx=np.concatenate([z,z]);yy=np.r_[np.zeros(len(z)),np.ones(len(z))];ww=np.r_[w*(1-q),w*q];active=ww>0;xx=xx[active];yy=yy[active];ww=ww[active]
    else:xx=z;yy=meta.y.to_numpy()[keep];ww=w
    lr=LogisticRegression(C=10,max_iter=2000,random_state=20261104)
    with threadpool_limits(limits=4):lr.fit(xx,yy,sample_weight=ww)
    return make_pipeline(scaler,lr)


def predict_lr(model,x):
    with threadpool_limits(limits=4):return model.predict_proba(x)[:,1]
