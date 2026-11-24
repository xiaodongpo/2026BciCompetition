"""Nested record/date validation with fixed round-two comparator."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,sklearn,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scripts.round2_train import sample_weights,score,threshold
from scripts.round3_protocol import ranked

SEED=2026110303


def build(c,dim):
    if c['model']=='hgb':
        return HistGradientBoostingClassifier(max_iter=c['max_iter'],max_leaf_nodes=c['leaves'],learning_rate=c['learning_rate'],l2_regularization=c['l2'],min_samples_leaf=c['min_leaf'],max_bins=c['max_bins'],early_stopping=False,random_state=SEED)
    lr=LogisticRegression(C=c['C'],max_iter=2000,random_state=SEED)
    if c['model']=='lr':return make_pipeline(StandardScaler(),lr)
    if c['model']=='rff':return make_pipeline(StandardScaler(),RBFSampler(gamma=c['gamma_factor']/dim,n_components=c['components'],random_state=SEED),lr)
    raise ValueError(c)


def fit(c,x,meta):
    assert (meta.groupby('record').y.nunique()==2).all()
    model=build(c,x.shape[1]);weights=sample_weights(meta)
    with threadpool_limits(limits=4):
        if c['model']=='hgb':model.fit(x,meta.y,sample_weight=weights)
        else:model.fit(x,meta.y,logisticregression__sample_weight=weights)
    return model


def predict(model,x):
    with threadpool_limits(limits=4):return model.predict_proba(x)[:,1]


def structure(model,dim):
    if hasattr(model,'named_steps'):
        lr=model.named_steps['logisticregression'];r=dict(input_features=int(dim),classifier_parameters=int(lr.coef_.size+lr.intercept_.size))
        if 'rbfsampler' in model.named_steps:
            rff=model.named_steps['rbfsampler'];r['fixed_projection_values']=int(rff.random_weights_.size+rff.random_offset_.size)
        return r
    trees=[tree for stage in model._predictors for tree in stage]
    return dict(input_features=int(dim),trees=len(trees),tree_nodes=sum(len(t.nodes) for t in trees),leaf_values=int(sum(t.nodes['is_leaf'].sum() for t in trees)))


def bundle(model,c,t,records):
    return dict(model=model,config=dict(**c,seconds=6),threshold=float(t),training_records=sorted(records),seed=SEED,fs=250,units='uV')


def inner_search(records,meta,features,candidates,out):
    out.mkdir(parents=True,exist_ok=True)
    order=ranked(records,'round3-inner-2026110303-');folds=[sorted(order[k::3]) for k in range(3)]
    (out/'inner_splits.json').write_text(json.dumps(dict(training_records=sorted(records),validation_folds=folds),indent=2))
    results=[]
    for c in candidates:
        prefix=out/c['id'];summary=prefix.with_suffix('.json')
        if summary.exists():results.append(json.loads(summary.read_text()));continue
        start=time.perf_counter();parts=[];fits=[];x=features[c['features']]
        for k,valid in enumerate(folds):
            train=sorted(set(records)-set(valid));tr=meta.record.isin(train);va=meta.record.isin(valid)
            assert not set(meta[tr].record)&set(meta[va].record)
            t0=time.perf_counter();model=fit(c,x[tr],meta[tr]);fit_s=time.perf_counter()-t0
            p=predict(model,x[va]);frame=meta[va].copy();frame['p']=p;frame['inner_fold']=k;parts.append(frame)
            cp=out/(c['id']+f'_inner{k}.joblib');joblib.dump(bundle(model,c,.5,train),cp,compress=3)
            fits.append(dict(fold=k,seconds=fit_s,structure=structure(model,x.shape[1]),checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
        oof=pd.concat(parts);oof.to_csv(prefix.with_suffix('.csv'),index=False)
        result=dict(id=c['id'],config=c,inner_score=score(oof),threshold=threshold(oof),seconds=time.perf_counter()-start,seed=SEED,fits=fits,source_manifest='experiments/round3/run_manifest.json')
        summary.write_text(json.dumps(result,indent=2));results.append(result)
    chosen=max(results,key=lambda r:r['inner_score'])
    (out/'selection.json').write_text(json.dumps(chosen,indent=2))
    pd.DataFrame([dict(id=r['id'],inner_score=r['inner_score'],threshold=r['threshold'],seconds=r['seconds']) for r in results]).to_csv(out/'leaderboard.csv',index=False)
    return chosen


def outer_protocol(name,folds,protocol,meta,features):
    root=Path('experiments/round3')/name;root.mkdir(exist_ok=True)
    for k,valid in enumerate(folds):
        out=root/f'outer{k}';out.mkdir(exist_ok=True)
        if (out/'done.json').exists():continue
        records=sorted(set(protocol['development'])-set(valid));tr=meta.record.isin(records);va=meta.record.isin(valid)
        (out/'outer_split.json').write_text(json.dumps(dict(train=records,validation=valid),indent=2))
        chosen=inner_search(records,meta,features,protocol['candidates'],out/'inner')
        timings={}
        ref=dict(protocol['fixed_reference']);ref_t=ref.pop('threshold')
        for kind,c,t in [('selected',chosen['config'],chosen['threshold']),('reference',ref,ref_t)]:
            start=time.perf_counter();x=features[c['features']];model=fit(c,x[tr],meta[tr]);p=predict(model,x[va])
            frame=meta[va].copy();frame['p']=p;frame['threshold']=t;frame['prediction']=(p>=t).astype(int);frame['outer_fold']=k;frame['selected_id']=c['id']
            frame.to_csv(out/(kind+'_outer.csv'),index=False);cp=out/(kind+'.joblib');joblib.dump(bundle(model,c,t,records),cp,compress=3)
            timings[kind]=dict(seconds=time.perf_counter()-start,structure=structure(model,x.shape[1]),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
        (out/'done.json').write_text(json.dumps(dict(selected=chosen['id'],inner_score=chosen['inner_score'],timings=timings),indent=2))
        print(name,k,'inner-selected',chosen['id'],flush=True)
    for kind in ('selected','reference'):
        pd.concat([pd.read_csv(root/f'outer{k}'/(kind+'_outer.csv'),dtype={'record':str}) for k in range(len(folds))]).to_csv(f'results/round3/{name}_{kind}_oof.csv',index=False)


def run():
    protocol=json.loads(Path('configs/round3/protocol.json').read_text());meta=pd.read_csv('cache/round3/windows.csv',dtype={'record':str})
    assert set(meta.record)==set(protocol['development']) and not set(meta.record)&set(protocol['excluded_previous_holdout'])
    features={k:np.load(f'cache/round3/{k}.npy') for k in ('base','normalized','multiscale')}
    for x in features.values():assert len(x)==len(meta) and np.isfinite(x).all()
    sources=[*Path('src').glob('*.py'),*Path('scripts').glob('round3*.py'),Path('scripts/round2_train.py'),Path('scripts/round2_protocol.py'),Path('configs/round3/protocol.json')]
    digests={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest=Path('experiments/round3/run_manifest.json')
    if manifest.exists():assert json.loads(manifest.read_text())['sources']==digests,'Frozen training sources changed.'
    else:
        provenance=json.loads(Path('reports/round3/feature_provenance.json').read_text())
        for k,v in provenance['features'].items():assert hashlib.sha256(Path(f'cache/round3/{k}.npy').read_bytes()).hexdigest()==v['sha256']
        manifest.write_text(json.dumps(dict(seed=SEED,sources=digests,feature_provenance=provenance,versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)),indent=2))
        with zipfile.ZipFile('experiments/round3/source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sources:z.write(p)
    start=time.perf_counter()
    outer_protocol('record_nested',protocol['outer_folds'],protocol,meta,features)
    info=pd.read_csv('reports/records.csv',dtype={'record':str});info=info[info.record.isin(protocol['development'])]
    dates=[sorted(g.record.to_list()) for _,g in info.groupby(info.origin.str[:10])]
    outer_protocol('date_nested',dates,protocol,meta,features)
    chosen=inner_search(protocol['development'],meta,features,protocol['candidates'],Path('experiments/round3/final_selection'))
    c=chosen['config'];model=fit(c,features[c['features']],meta)
    joblib.dump(bundle(model,c,chosen['threshold'],protocol['development']),'checkpoints/round3/research_candidate.joblib',compress=3)
    Path('configs/round3/selected_model.json').write_text(json.dumps(chosen,indent=2))
    Path('experiments/round3/completed.json').write_text(json.dumps(dict(invocation_seconds=time.perf_counter()-start,selected=chosen['id'],structure=structure(model,features[c['features']].shape[1])),indent=2))
    print('FINAL',chosen['id'],'threshold',chosen['threshold'],flush=True)


if __name__=='__main__':run()
