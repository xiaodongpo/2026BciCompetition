"""Nested record/date validation with fixed round-two comparator."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,sklearn,joblib
from scripts.round5_protocol import ranked

SEED=20261104


from src.round5_model import fit,predict
from src.evaluation import record_metrics

def score(frame):
    r=record_metrics(frame,threshold=.46)
    return float((r.bacc.mean()+r.auc.mean())/2)

def structure(model,dim):
    lr=model.named_steps['logisticregression']
    return dict(input_features=int(dim),classifier_parameters=int(lr.coef_.size+lr.intercept_.size))


def bundle(model,c,t,records):
    return dict(model=model,config={**c,'seconds':6},threshold=float(t),training_records=sorted(records),seed=SEED,fs=250,units='uV')


def inner_search(records,meta,features,candidates,out):
    out.mkdir(parents=True,exist_ok=True)
    order=ranked(records,'round5-inner-20261104-');folds=[sorted(order[k::3]) for k in range(3)]
    (out/'inner_splits.json').write_text(json.dumps(dict(training_records=sorted(records),validation_folds=folds),indent=2))
    results=[]
    for c in candidates:
        prefix=out/c['id'];summary=Path(str(prefix)+'.json')
        if summary.exists():results.append(json.loads(summary.read_text()));continue
        start=time.perf_counter();parts=[];fits=[];x=features[c['features']]
        for k,valid in enumerate(folds):
            train=sorted(set(records)-set(valid));tr=meta.record.isin(train);va=meta.record.isin(valid)
            assert not set(meta[tr].record)&set(meta[va].record)
            t0=time.perf_counter();model=fit(c,x[tr],meta[tr]);fit_s=time.perf_counter()-t0
            p=predict(model,x[va]);frame=meta[va].copy();frame['p']=p;frame['inner_fold']=k;parts.append(frame)
            cp=out/(c['id']+f'_inner{k}.joblib');joblib.dump(bundle(model,c,.46,train),cp,compress=3)
            fits.append(dict(fold=k,seconds=fit_s,structure=structure(model,x.shape[1]),checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
        oof=pd.concat(parts);oof.to_csv(Path(str(prefix)+'.csv'),index=False)
        result=dict(id=c['id'],config=c,inner_score=score(oof),threshold=.46,seconds=time.perf_counter()-start,seed=SEED,fits=fits,source_manifest='experiments/round5/run_manifest.json')
        summary.write_text(json.dumps(result,indent=2));results.append(result)
    chosen=max(results,key=lambda r:r['inner_score'])
    (out/'selection.json').write_text(json.dumps(chosen,indent=2))
    pd.DataFrame([dict(id=r['id'],inner_score=r['inner_score'],threshold=r['threshold'],seconds=r['seconds']) for r in results]).to_csv(out/'leaderboard.csv',index=False)
    return chosen


def outer_protocol(name,folds,protocol,meta,features):
    root=Path('experiments/round5')/name;root.mkdir(exist_ok=True)
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
        pd.concat([pd.read_csv(root/f'outer{k}'/(kind+'_outer.csv'),dtype={'record':str}) for k in range(len(folds))]).to_csv(f'results/round5/{name}_{kind}_oof.csv',index=False)


def run():
    protocol=json.loads(Path('configs/round5/protocol.json').read_text());meta=pd.read_csv('cache/round5/windows.csv',dtype={'record':str})
    assert set(meta.record)==set(protocol['development']) and not set(meta.record)&set(protocol['excluded_previous_holdout'])
    features={k:np.load(f'cache/round5/{k}.npy') for k in ('base','ordinal','fusion')}
    for x in features.values():assert len(x)==len(meta) and np.isfinite(x).all()
    sources=[*Path('src').glob('*.py'),*Path('scripts').glob('round5*.py'),Path('scripts/round2_train.py'),Path('scripts/round2_protocol.py'),Path('scripts/round3_protocol.py'),Path('scripts/round2_evaluate.py'),Path('configs/round5/protocol.json'),Path('reports/round5/protocol.md'),Path('tests/test_round5.py')]
    digests={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest=Path('experiments/round5/run_manifest.json')
    if manifest.exists():
        saved=json.loads(manifest.read_text());assert saved['sources']==digests,'Frozen training sources changed.'
        for k,v in saved['feature_provenance']['features'].items():assert hashlib.sha256(Path(f'cache/round5/{k}.npy').read_bytes()).hexdigest()==v['sha256']
        assert hashlib.sha256(Path('cache/round5/windows.csv').read_bytes()).hexdigest()==saved['feature_provenance']['metadata_sha256']
    else:
        provenance=json.loads(Path('reports/round5/feature_provenance.json').read_text())
        for k,v in provenance['features'].items():assert hashlib.sha256(Path(f'cache/round5/{k}.npy').read_bytes()).hexdigest()==v['sha256']
        assert hashlib.sha256(Path('cache/round5/windows.csv').read_bytes()).hexdigest()==provenance['metadata_sha256']
        manifest.write_text(json.dumps(dict(seed=SEED,sources=digests,feature_provenance=provenance,versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)),indent=2))
        with zipfile.ZipFile('experiments/round5/source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sources:z.write(p)
    if Path('experiments/round5/completed.json').exists():
        print('Already complete; sources and caches verified; no refits.',flush=True);return
    start=time.perf_counter()
    outer_protocol('record_nested',protocol['outer_folds'],protocol,meta,features)
    info=pd.read_csv('reports/records.csv',dtype={'record':str});info=info[info.record.isin(protocol['development'])]
    dates=[sorted(g.record.to_list()) for _,g in info.groupby(info.origin.str[:10])]
    outer_protocol('date_nested',dates,protocol,meta,features)
    chosen=inner_search(protocol['development'],meta,features,protocol['candidates'],Path('experiments/round5/final_selection'))
    c=chosen['config'];model=fit(c,features[c['features']],meta)
    joblib.dump(bundle(model,c,chosen['threshold'],protocol['development']),'checkpoints/round5/research_candidate.joblib',compress=3)
    Path('configs/round5/selected_model.json').write_text(json.dumps(chosen,indent=2))
    Path('experiments/round5/completed.json').write_text(json.dumps(dict(invocation_seconds=time.perf_counter()-start,selected=chosen['id'],structure=structure(model,features[c['features']].shape[1])),indent=2))
    print('FINAL',chosen['id'],'threshold',chosen['threshold'],flush=True)


if __name__=='__main__':run()
