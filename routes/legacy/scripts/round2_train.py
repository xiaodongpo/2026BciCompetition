import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import json,time,hashlib,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from src.evaluation import metrics,record_metrics
from scripts.round2_protocol import ranked

def sample_weights(frame):
    counts=frame.groupby(['record','y']).y.transform('size').to_numpy()
    w=1/counts;return w/w.mean()

def build(c):return make_pipeline(StandardScaler(),LogisticRegression(C=c['C'],max_iter=2000,class_weight=None if c['record_weight'] else 'balanced',random_state=20261103))

def fit(c,x,meta):
    model=build(c);kwargs={}
    if c['record_weight']:kwargs['logisticregression__sample_weight']=sample_weights(meta)
    with threadpool_limits(limits=4):model.fit(x,meta.y,**kwargs)
    return model

def score(frame):
    rm=record_metrics(frame);return float((rm.bacc.mean()+rm.auc.mean())/2)

def threshold(frame):
    w=sample_weights(frame);y=frame.y.to_numpy();p=frame.p.to_numpy()
    candidates=np.linspace(.35,.65,31);scores=[]
    for t in candidates:
        # Each record contributes equally and its two classes have equal total weight.
        scores.append(float(np.average((p>=t)==y,weights=w)))
    best=max(scores);return float(min((t for t,s in zip(candidates,scores) if abs(s-best)<1e-12),key=lambda t:abs(t-.5)))

def inner_search(train_records,meta,features,candidates,out):
    order=ranked(train_records,'round2-inner-');folds=[order[k::3] for k in range(3)]
    (out/'inner_splits.json').write_text(json.dumps(dict(training_records=train_records,validation_folds=folds),indent=2))
    results=[]
    for c in candidates:
        cp=out/(c['id']+'_inner.csv');summary=out/(c['id']+'_inner.json')
        if summary.exists():results.append(json.loads(summary.read_text()));continue
        start=time.perf_counter();parts=[];x=features[c['features']]
        for k,valid in enumerate(folds):
            tr=meta.record.isin(set(train_records)-set(valid));va=meta.record.isin(valid)
            assert not set(meta[tr].record)&set(meta[va].record)
            model=fit(c,x[tr],meta[tr]);p=model.predict_proba(x[va])[:,1]
            joblib.dump(dict(model=model,config=dict(**c,seconds=6),threshold=.5,training_records=sorted(set(meta[tr].record)),seed=20261103),out/(c['id']+f'_inner{k}.joblib'),compress=3)
            part=meta[va].copy();part['p']=p;part['fold']=k;parts.append(part)
        oof=pd.concat(parts);oof.to_csv(cp,index=False)
        row=dict(id=c['id'],config=c,inner_score=score(oof),threshold=threshold(oof),seconds=time.perf_counter()-start,seed=20261103,source_manifest='experiments/round2/run_manifest.json',split_file=str(out/'inner_splits.json'),input_features=int(x.shape[1]),classifier_parameters=int(x.shape[1]+1),checkpoints=[str(out/(c['id']+f'_inner{k}.joblib')) for k in range(3)])
        summary.write_text(json.dumps(row,indent=2));results.append(row)
    chosen=max(results,key=lambda r:r['inner_score'])
    (out/'inner_selection.json').write_text(json.dumps(chosen,indent=2))
    pd.DataFrame([{**r,'config':json.dumps(r['config'])} for r in results]).to_csv(out/'inner_leaderboard.csv',index=False)
    return chosen

def run_protocol(name,folds,protocol,meta,features):
    all_records=protocol['development'];root=Path('experiments/round2')/name;root.mkdir(exist_ok=True)
    for k,valid in enumerate(folds):
        out=root/f'outer{k}';out.mkdir(exist_ok=True)
        if (out/'done.json').exists():continue
        tr=meta.record.isin(set(all_records)-set(valid));va=meta.record.isin(valid)
        assert set(meta[va].record)==set(valid)
        selection=inner_search(sorted(set(all_records)-set(valid)),meta,features,protocol['candidates'],out)
        (out/'outer_split.json').write_text(json.dumps(dict(train=sorted(set(all_records)-set(valid)),validation=valid),indent=2))
        for kind,c,t in [('selected',selection['config'],selection['threshold']),('reference',protocol['candidates'][0],.49)]:
            model=fit(c,features[c['features']][tr],meta[tr]);p=model.predict_proba(features[c['features']][va])[:,1]
            result=meta[va].copy();result['p']=p;result['threshold']=t;result['prediction']=(p>=t).astype(int);result['outer_fold']=k;result['selected_id']=c['id']
            result.to_csv(out/(kind+'_outer.csv'),index=False)
            joblib.dump(dict(model=model,config=dict(**c,seconds=6),threshold=t,training_records=sorted(set(meta[tr].record)),seed=20261103),out/(kind+'.joblib'),compress=3)
        (out/'done.json').write_text(json.dumps(dict(selected=selection['id'],inner_score=selection['inner_score'])))
        print(name,k,'inner-selected',selection['id'],flush=True)
    for kind in ('selected','reference'):
        df=pd.concat([pd.read_csv(root/f'outer{k}'/(kind+'_outer.csv'),dtype={'record':str}) for k in range(len(folds))])
        df.to_csv(f'results/round2/{name}_{kind}_oof.csv',index=False)

def run():
    protocol=json.loads(Path('configs/round2/protocol.json').read_text());meta=pd.read_csv('cache/round2/windows.csv',dtype={'record':str})
    assert set(meta.record)==set(protocol['development']);assert not set(meta.record)&set(protocol['excluded_previous_holdout'])
    features={k:np.load(f'cache/round2/{k}.npy') for k in {c['features'] for c in protocol['candidates']}}
    for x in features.values():assert len(x)==len(meta) and np.isfinite(x).all()
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*Path('src').glob('*.py'),*Path('scripts').glob('round2*.py'),Path('configs/round2/protocol.json')]}
    manifest=Path('experiments/round2/run_manifest.json')
    if manifest.exists():assert json.loads(manifest.read_text())['sources']==sources,'Source changed since run; preserve outputs and start a new version.'
    else:
        import sklearn,scipy
        manifest.write_text(json.dumps(dict(seed=20261103,sources=sources,feature_provenance=json.loads(Path('reports/round2/feature_provenance.json').read_text()),versions=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__)),indent=2))
    with zipfile.ZipFile('experiments/round2/source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in [*Path('src').glob('*.py'),*Path('scripts').glob('round2*.py'),Path('configs/round2/protocol.json')]:z.write(p)
    run_protocol('record_nested',protocol['outer_folds'],protocol,meta,features)
    # The stress protocol is preregistered and always completed, regardless of record-fold results.
    info=pd.read_csv('reports/records.csv',dtype={'record':str});info=info[info.record.isin(protocol['development'])]
    dates=[sorted(d.record.to_list()) for _,d in info.groupby(info.origin.str[:10])]
    run_protocol('date_nested',dates,protocol,meta,features)
    out=Path('experiments/round2/final_selection');out.mkdir(exist_ok=True)
    selected=inner_search(protocol['development'],meta,features,protocol['candidates'],out)
    model=fit(selected['config'],features[selected['config']['features']],meta)
    bundle=dict(model=model,config=dict(**selected['config'],seconds=6),threshold=selected['threshold'],training_records=protocol['development'],fs=250,seconds=6,units='uV')
    joblib.dump(bundle,'checkpoints/round2/research_candidate.joblib',compress=3)
    Path('configs/round2/selected_model.json').write_text(json.dumps(selected,indent=2));print('FINAL_CANDIDATE',selected,flush=True)

if __name__=='__main__':run()
