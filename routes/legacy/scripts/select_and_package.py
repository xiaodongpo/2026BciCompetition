"""Freeze the winning development configuration, then evaluate the untouched holdout once."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import json,time,shutil,hashlib,zipfile,sys,subprocess,html
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from scripts.train_classical import build,load,source_digest
from src.evaluation import metrics,record_metrics
from src.inference import StreamingPredictor

def bootstrap_records(rm):
    rng=np.random.default_rng(2026);arr=rm[['bacc','auc']].to_numpy();draw=arr[rng.integers(0,len(arr),size=(2000,len(arr)))].mean(1)
    return dict(record_macro_bacc_95ci=np.quantile(draw[:,0],[.025,.975]).tolist(),record_macro_auc_95ci=np.quantile(draw[:,1],[.025,.975]).tolist())

def run():
    table=pd.read_csv('results/leaderboard.csv');winner=table.iloc[0]
    summary=json.loads((Path(winner.checkpoint)/'summary.json').read_text());c=summary['config']
    # Never silently substitute a lower-ranked model if the winning backend is unsupported.
    if c['model'] not in ('lr','rbf','hgb','extra'):raise RuntimeError('Add selected backend to exporter: '+c['model'])
    lock=Path('configs/selected_model.json')
    selection=dict(experiment_id=summary['id'],config=c,threshold=summary['threshold'],selection='mean of development OOF bACC@0.5 and AUC',source_sha256=source_digest())
    if lock.exists():
        old=json.loads(lock.read_text());assert old['experiment_id']==selection['experiment_id'],'Holdout already locked; no reselection'
    else:lock.write_text(json.dumps(selection,indent=2))
    split=json.loads(Path('configs/splits.json').read_text());X,meta=load(c);tr=meta.record.isin(split['development']);ho=meta.record.isin(split['holdout'])
    with threadpool_limits(limits=4):model=build(c).fit(X[tr],meta.y[tr]);p=model.predict_proba(X[ho])[:,1]
    bundle=dict(model=model,config=c,threshold=summary['threshold'],training_records=split['development'],label_map={'relax':0,'task':1},fs=250,units='uV',selection=selection)
    joblib.dump(bundle,'checkpoints/best_model.joblib',compress=3);joblib.dump(bundle,'submission/model.joblib',compress=3)
    pred=meta[ho].copy();pred['p']=p;pred.to_csv('results/final_holdout_predictions.csv',index=False)
    rm=record_metrics(pred,summary['threshold']);rm.to_csv('results/final_holdout_records.csv',index=False)
    result=dict(experiment_id=summary['id'],holdout_records=split['holdout'],threshold=summary['threshold'],
                pooled=metrics(pred.y,p,summary['threshold']),at_05=metrics(pred.y,p),
                record_macro_bacc=float(rm.bacc.mean()),record_macro_auc=float(rm.auc.mean()),**bootstrap_records(rm))
    # Exercise the actual deployed adapter on continuous streams, including phase-transition windows.
    rows=[];timings=[];coverage=[]
    for rid in split['holdout']:
        d=np.load('cache/records/'+rid+'.npz');s=StreamingPredictor('submission/model.joblib');outputs=[];start=time.perf_counter()
        with threadpool_limits(limits=1):
            for a in range(0,len(d['x']),250):outputs.extend(s.process(d['x'][a:a+250],start_sample=a,valid=d['valid'][a:a+250]))
        timings.append(time.perf_counter()-start)
        nvalid=0;neligible=0
        for row in outputs:
            end=row['sample_index'];y=int(d['labels'][end])
            if y>=0:
                neligible+=1
                if row['valid']:
                    nvalid+=1;rows.append(dict(record=rid,end=end,y=y,p=row['p_task']))
        coverage.append(dict(record=rid,eligible=neligible,predicted=nvalid,coverage=nvalid/max(1,neligible)))
    stream=pd.DataFrame(rows);stream.to_csv('results/stream_holdout_predictions.csv',index=False)
    pd.DataFrame(coverage).to_csv('results/stream_coverage.csv',index=False)
    result['stream_endpoint_labels']=metrics(stream.y,stream.p,summary['threshold']);result['stream_total_cpu_seconds']=sum(timings)
    result['stream_record_seconds']=timings;result['stream_coverage']=sum(r['predicted'] for r in coverage)/sum(r['eligible'] for r in coverage)
    result['note']='Local record holdout only; not official hidden test. Final model fits 45 development records only.'
    Path('results/final_evaluation.json').write_text(json.dumps(result,indent=2))
    # Synthetic I/O example is explicitly NOT a performance evaluation.
    rng=np.random.default_rng(2026);t=np.arange(2500)/250;x=(20*np.sin(2*np.pi*10*t)+rng.normal(0,5,len(t))).astype('float32')
    valid=np.ones(len(x),bool);np.savez('submission/example_input.npz',eeg=x,valid=valid,fs=250)
    for filename in ('__init__.py','features.py','inference.py'):
        dest=Path('submission/src')/filename;dest.parent.mkdir(exist_ok=True);shutil.copy2(Path('src')/filename,dest)
    versions=json.loads(Path('reports/package_versions.json').read_text())
    from importlib.metadata import version
    Path('submission/requirements.txt').write_text('\n'.join(f'{p}=={version(p)}' for p in ('numpy','scipy','scikit-learn','joblib','narwhals','threadpoolctl'))+'\n')
    Path('submission/config.json').write_text(json.dumps(selection,indent=2))
    with zipfile.ZipFile('checkpoints/source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for directory in ('src','scripts','configs','tests'):
            for path in Path(directory).rglob('*'):
                if path.is_file() and '__pycache__' not in str(path):z.write(path)
        z.write('requirements.txt');z.write('AGENTS.md')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':run()
