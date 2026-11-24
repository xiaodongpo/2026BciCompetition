"""Nested training-only transition interventions, complete outer validation for all arms."""
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn,scipy,joblib
from scripts.r6_protocol import axes,additions
from scripts.round3_protocol import ranked
from src.transition_training import fit_lr,predict_lr,sample_summary
from src.transition_evaluation import summary


def save_bundle(model,c,train,path):
    joblib.dump(dict(model=model,config=dict(features='filterbank',seconds=6,training_policy=c),threshold=.46,training_records=sorted(train),seed=20261104,units='uV',fs=250),path,compress=3)


def search(records,meta,x,out):
    out.mkdir(parents=True,exist_ok=True);order=ranked(records,'round6-inner-20261104-');folds=[sorted(order[k::3]) for k in range(3)]
    (out/'splits.json').write_text(json.dumps(dict(training_records=sorted(records),validation_folds=folds),indent=2));results=[]
    def evaluate(c):
        path=out/(c['id']+'.json')
        if path.exists():return json.loads(path.read_text())
        parts=[];fits=[];start=time.perf_counter()
        for k,valid in enumerate(folds):
            train=sorted(set(records)-set(valid));tr=meta.record.isin(train);va=meta.record.isin(valid)
            t0=time.perf_counter();model=fit_lr(x[tr],meta[tr],c);p=predict_lr(model,x[va]);cp=out/(c['id']+f'_inner{k}.joblib');save_bundle(model,c,train,cp)
            f=meta[va].copy();f['p']=p;f['inner_fold']=k;parts.append(f)
            fits.append(dict(fold=k,seconds=time.perf_counter()-t0,checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),samples=sample_summary(meta[tr],c)))
        oof=pd.concat(parts);metrics,_=summary(oof,oof.p);oof.to_csv(out/(c['id']+'.csv'),index=False)
        result=dict(config=c,id=c['id'],metrics=metrics,inner_score=metrics['score'],fits=fits,seconds=time.perf_counter()-start);path.write_text(json.dumps(result,indent=2));return result
    for c in axes():results.append(evaluate(c))
    task=max([r for r in results if r['id']=='full_hard' or r['id'].startswith('drop_task_')],key=lambda r:r['inner_score'])['config']['T_task']
    relax=max([r for r in results if r['id']=='full_hard' or r['id'].startswith('drop_relax_')],key=lambda r:r['inner_score'])['config']['T_relax']
    for c in additions(task,relax):results.append(evaluate(c))
    chosen=max(results,key=lambda r:r['inner_score'])
    (out/'selection.json').write_text(json.dumps(dict(**chosen,axis_T_task=task,axis_T_relax=relax),indent=2))
    pd.DataFrame([dict(**r['config'],inner_score=r['inner_score'],seconds=r['seconds']) for r in results]).to_csv(out/'leaderboard.csv',index=False)
    return results,chosen


def run():
    p=json.loads(Path('configs/round6/protocol.json').read_text());meta=pd.read_csv('cache/round6/windows.csv',dtype={'record':str});x=np.load('cache/round6/base.npy')
    assert set(meta.record)==set(p['development']) and not set(meta.record)&set(p['excluded_previous_holdout'])
    source_paths=[Path(__file__),Path('scripts/r6_protocol.py'),Path('scripts/round3_protocol.py'),Path('src/transition_training.py'),Path('src/transition_evaluation.py'),Path('configs/round6/protocol.json')]
    hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in source_paths};manifest=Path('experiments/round6/lr_manifest.json')
    if manifest.exists():assert json.loads(manifest.read_text())['sources']==hashes
    else:
        manifest.write_text(json.dumps(dict(sources=hashes,data=json.loads(Path('cache/round6/prepared.json').read_text()),versions=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__)),indent=2))
        with zipfile.ZipFile('experiments/round6/lr_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for q in source_paths:z.write(q)
    if Path('experiments/round6/lr_complete.json').exists():print('LR complete; no refits.');return
    start=time.perf_counter();info=pd.read_csv('reports/round6/rechecked_records.csv',dtype={'record':str});info=info[info.record.isin(p['development'])]
    protocols={'record':p['outer_folds'],'date':[sorted(g.record.tolist()) for _,g in info.groupby(info.origin.str[:10])]}
    for name,folds in protocols.items():
        for k,valid in enumerate(folds):
            out=Path(f'experiments/round6/lr/{name}/outer{k}');out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():continue
            train=sorted(set(p['development'])-set(valid));tr=meta.record.isin(train);va=meta.record.isin(valid)
            (out/'split.json').write_text(json.dumps(dict(train=train,validation=valid),indent=2))
            results,chosen=search(train,meta,x,out/'inner');details=[]
            for r in results:
                c=r['config'];t0=time.perf_counter();model=fit_lr(x[tr],meta[tr],c);proba=predict_lr(model,x[va]);cp=out/(c['id']+'.joblib');save_bundle(model,c,train,cp)
                frame=meta[va].copy();frame['p']=proba;frame['threshold']=.46;frame['prediction']=(proba>=.46).astype(int);frame['outer_fold']=k;frame['policy']=c['id'];frame.to_csv(out/(c['id']+'.csv'),index=False)
                metrics,rm=summary(frame,proba);pure=frame.pure.to_numpy();pure_metrics,_=summary(frame[pure],proba[pure]);rm['outer_fold']=k;rm.to_csv(out/(c['id']+'_records.csv'),index=False)
                details.append(dict(id=c['id'],config=c,full=metrics,pure=pure_metrics,samples=sample_summary(meta[tr],c),seconds=time.perf_counter()-t0,checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
            (out/'complete.json').write_text(json.dumps(dict(selected=chosen['id'],inner_score=chosen['inner_score'],ablations=details),indent=2));print(name,k,'selected',chosen['id'],flush=True)
    results,chosen=search(p['development'],meta,x,Path('experiments/round6/lr/final_selection'));model=fit_lr(x,meta,chosen['config']);save_bundle(model,chosen['config'],p['development'],Path('checkpoints/round6/lr_candidate.joblib'))
    Path('experiments/round6/lr_complete.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,selected=chosen,planned_inner_fits=22*3*8,outer_fits=22*7,final_fits=1,total_research_fits=683),indent=2));print('LR complete',chosen['id'],flush=True)


if __name__=='__main__':run()
