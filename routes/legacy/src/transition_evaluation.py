import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score,log_loss


def measures(y,p,threshold=.46):
    y=np.asarray(y);p=np.asarray(p);pred=p>=threshold;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sens=float(tp/(tp+fn)) if tp+fn else None;spec=float(tn/(tn+fp)) if tn+fp else None
    return dict(n=len(y),bacc=(sens+spec)/2 if sens is not None and spec is not None else None,auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,sensitivity=sens,specificity=spec,f1=float(f1_score(y,pred,zero_division=0)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),loss=float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])))


def summary(meta,p,threshold=.46):
    f=meta.copy();f['p']=p
    records=pd.DataFrame([dict(record=rid,**measures(g.y,g.p,threshold)) for rid,g in f.groupby('record')])
    return dict(pooled=measures(f.y,f.p,threshold),macro_bacc=float(records.bacc.mean()),macro_auc=float(records.auc.mean()),score=float((records.bacc.mean()+records.auc.mean())/2)),records


def paired_gate(selected,reference):
    a=selected.set_index('record');b=reference.set_index('record').loc[a.index]
    diff=(a.bacc+a.auc-b.bacc-b.auc).to_numpy()/2;rng=np.random.default_rng(20261104)
    lo,hi=np.quantile(diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(1),[.025,.975])
    folds=[float(((g.bacc+g.auc)/2-(b.loc[g.index].bacc+b.loc[g.index].auc)/2).mean()) for _,g in a.groupby('outer_fold')]
    bacc=float((a.bacc-b.bacc).mean())
    return dict(mean_gain=float(diff.mean()),bootstrap95=[float(lo),float(hi)],fold_gains=folds,bacc_gain=bacc,record_passed=bool(lo>0 and min(folds)>0 and bacc>=0),replicates=10000,unit='record, not verified participant')
