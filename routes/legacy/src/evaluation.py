import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,recall_score,f1_score,confusion_matrix

def metrics(y,p,threshold=.5):
    h=np.asarray(p)>=threshold
    tn,fp,fn,tp=confusion_matrix(y,h,labels=[0,1]).ravel()
    return dict(bacc=float(balanced_accuracy_score(y,h)),auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'),
                sensitivity=float(tp/max(1,tp+fn)),specificity=float(tn/max(1,tn+fp)),f1=float(f1_score(y,h,zero_division=0)),
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def record_metrics(frame,threshold=.5):
    return pd.DataFrame([dict(record=g,**metrics(d.y,d.p,threshold)) for g,d in frame.groupby('record')])

def choose_threshold(y,p):
    # Coarse prespecified grid limits threshold-search degrees of freedom.
    candidates=np.linspace(.25,.75,51)
    scores=[balanced_accuracy_score(y,np.asarray(p)>=t) for t in candidates]
    best=max(scores)
    return float(min((t for t,s in zip(candidates,scores) if s==best),key=lambda t:abs(t-.5)))
