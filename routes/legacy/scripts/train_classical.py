import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from sklearn.kernel_approximation import Nystroem
from threadpoolctl import threadpool_limits
from src.evaluation import metrics,record_metrics,choose_threshold

def source_digest():
    h=hashlib.sha256()
    for p in sorted([*Path('src').rglob('*.py'),*Path('scripts').rglob('*.py')]):h.update(str(p).encode());h.update(p.read_bytes())
    return h.hexdigest()

def build(c):
    name=c['model']
    if name=='lr':return make_pipeline(StandardScaler(),LogisticRegression(C=c['C'],class_weight='balanced',max_iter=1500,random_state=42))
    if name=='rbf':return make_pipeline(StandardScaler(),Nystroem(gamma=c['gamma'],n_components=256,random_state=42),LogisticRegression(C=c['C'],class_weight='balanced',max_iter=1000,random_state=42))
    if name=='hgb':return HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=c['leaves'],learning_rate=.05,l2_regularization=5,min_samples_leaf=80,early_stopping=False,class_weight='balanced',random_state=42)
    if name=='extra':return ExtraTreesClassifier(n_estimators=250,min_samples_leaf=c['leaf'],max_features=.8,class_weight='balanced',n_jobs=4,random_state=42)
    raise ValueError(name)

def configs():
    for s in (2,4,6):
        for C in (.01,1):yield dict(model='lr',seconds=s,C=C,features='full')
        yield dict(model='lr',seconds=s,C=1,features='relative')
        yield dict(model='hgb',seconds=s,leaves=15,features='full')
    for gamma in (.01,.05):yield dict(model='rbf',seconds=6,C=1,gamma=gamma,features='full')
    yield dict(model='extra',seconds=6,leaf=30,features='full')
    for gamma,C in ((.003,1),(.01,10),(.003,10),(.01,.1)):
        yield dict(model='rbf',seconds=6,C=C,gamma=gamma,features='full')
    for mode in ('fir','detrend'):
        yield dict(model='lr',seconds=6,C=1,features=mode)
        yield dict(model='rbf',seconds=6,C=1,gamma=.01,features=mode)
    for C in (.01,.1,10):yield dict(model='lr',seconds=6,C=C,features='fir')
    for mode in ('fir_2_40','fir_4_40'):yield dict(model='lr',seconds=6,C=1,features=mode)

def load(c):
    s=c['seconds']; X=np.load(f'cache/features_{s}s.npy'); meta=pd.read_csv(f'cache/windows_{s}s.csv',dtype={'record':str})
    if c['features']=='relative':X=X[:,8:14]
    elif c['features'] in ('fir','detrend','fir_2_40','fir_4_40'):
        from src.features import extract_features
        path=Path(f"cache/features_6s_{c['features']}.npy")
        if not path.exists():
            raw=np.load('cache/raw_6s.npy',mmap_mode='r')
            features=[extract_features(raw[a:a+256],mode=c['features']) for a in range(0,len(raw),256)]
            np.save(path,np.concatenate(features))
        X=np.load(path)
    return X,meta

def run():
    split=json.loads(Path('configs/splits.json').read_text())
    for c in configs():
        eid='_'.join(f'{k}-{v}' for k,v in c.items()); out=Path('experiments')/eid;out.mkdir(exist_ok=True)
        if (out/'summary.json').exists():continue
        X,meta=load(c); y=meta.y.to_numpy(); pred=[];folds=[];start=time.perf_counter()
        for fold,valid in enumerate(split['folds']):
            tr=meta.record.isin(set(split['development'])-set(valid)); va=meta.record.isin(valid)
            assert not set(meta[tr].record)&set(meta[va].record)
            model=build(c)
            with threadpool_limits(limits=4): model.fit(X[tr],y[tr]);p=model.predict_proba(X[va])[:,1]
            part=meta[va].copy();part['p']=p;part['fold']=fold;pred.append(part)
            folds.append(dict(fold=fold,**metrics(y[va],p)))
            joblib.dump(model,out/f'fold{fold}.joblib',compress=3)
        oof=pd.concat(pred);oof.to_csv(out/'oof.csv',index=False)
        pd.DataFrame(folds).to_csv(out/'fold_metrics.csv',index=False)
        rm=record_metrics(oof);rm.to_csv(out/'record_metrics.csv',index=False)
        threshold=choose_threshold(oof.y,oof.p)
        summary=dict(id=eid,config=c,seed=42,source_sha256=source_digest(),split='configs/splits.json',
                     **metrics(oof.y,oof.p),threshold=threshold,tuned_bacc=metrics(oof.y,oof.p,threshold)['bacc'],
                     record_bacc_mean=float(rm.bacc.mean()),record_auc_mean=float(rm.auc.mean()),
                     fold_bacc_std=float(np.std([m['bacc'] for m in folds])),fold_auc_std=float(np.std([m['auc'] for m in folds])),
                     train_seconds=time.perf_counter()-start,gpu_peak_mb=0,epochs=0,checkpoint=str(out))
        (out/'config.json').write_text(json.dumps(c,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2))
        update_tables();print(json.dumps(summary),flush=True)

def update_tables():
    rows=[json.loads(p.read_text()) for p in Path('experiments').glob('*/summary.json')]
    for r in rows:r['config']=json.dumps(r['config'],sort_keys=True)
    df=pd.DataFrame(rows);df['selection_score']=(df.bacc+df.auc)/2
    df.to_csv('results/experiments.csv',index=False)
    df.sort_values('selection_score',ascending=False).to_csv('results/leaderboard.csv',index=False)
if __name__=='__main__':run()
