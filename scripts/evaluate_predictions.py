"""Evaluate explicit labels and P(task); exclude only explicitly invalid rows."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,accuracy_score

def evaluate(y,p,threshold=.5):
    y=np.asarray(y);p=np.asarray(p,dtype=float)
    if y.ndim!=1 or y.shape!=p.shape or not len(y) or not np.isin(y,[0,1]).all():raise ValueError('Expected matching nonempty binary labels and scores')
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('P(task) must be finite in [0,1]')
    if not 0<=threshold<=1:raise ValueError('Invalid threshold')
    if len(np.unique(y))<2:raise ValueError('AUC requires both classes')
    auc=float(roc_auc_score(y,p));bacc=float(balanced_accuracy_score(y,p>=threshold));acc=float(accuracy_score(y,p>=threshold))
    return dict(n=len(y),auc=auc,bacc=bacc,accuracy=acc,threshold=threshold,preliminary_score=(auc+bacc)/2,finals_score=(auc+acc)/2)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('csv',type=Path);ap.add_argument('--threshold',type=float,default=.5);a=ap.parse_args()
    d=pd.read_csv(a.csv)
    if 'valid' in d:
        mask=d.valid.astype(str).str.lower()
        if not mask.isin(['true','false','1','0']).all():raise ValueError('valid must be true/false or 1/0')
        d=d[mask.isin(['true','1'])]
    print(json.dumps(evaluate(d.y,d.p_task,a.threshold),indent=2))
if __name__=='__main__':main()
