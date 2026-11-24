"""Fixed D2 five-fold real-stream reproduction, persistent models and predictions."""
import concurrent.futures as cf
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from scripts.train_final import load_features, train_recipe
from src.data.io import list_recordings, load_recording
from src.evaluation.metrics import evaluate
from src.inference.decoder import StreamingDecoder
from src.utils.exp_log import log_experiment

PATTERNS=sorted({''.join(p) for p in itertools.permutations('rrrttt')})
BASE=Path('checkpoints/local_cv')
PRED=Path('results/local_predictions')

def build(r,pattern):
    x=r.x.astype(np.float64).copy(); x[r.bad]=np.nan
    lab=r.label_array()
    if pattern=='real': return x,lab,r.bad.copy()
    rr=[t for t in r.trials if t.label==0]; tt=[t for t in r.trials if t.label==1]
    order=[rr.pop(0) if c=='r' else tt.pop(0) for c in pattern]
    xs=[x[:r.trials[0].start]]; ls=[np.full(r.trials[0].start,-1)]
    for t in order:
        xs.extend([np.full(1,np.nan),x[t.start:t.end]])
        ls.extend([np.full(1,-1),np.full(t.end-t.start,t.label)])
    x,lab=np.concatenate(xs),np.concatenate(ls)
    return x,lab,~np.isfinite(x)

def stream_record(k,b,c):
    r=load_recording(b,c)
    dec=StreamingDecoder(str(BASE/f'f{k}'))
    outdir=PRED/f'f{k}'; outdir.mkdir(parents=True,exist_ok=True)
    for pattern in ['real']+PATTERNS:
        pth=outdir/f'{r.rec_id}_{pattern}.npz'
        if pth.exists(): continue
        x,lab,bad=build(r,pattern); dec.reset(); outputs=[]
        t0=time.perf_counter()
        for i in range(0,len(x),250): outputs.extend(dec.push(x[i:i+250]))
        seconds=time.perf_counter()-t0
        t=np.array([o.sample_index for o in outputs]); keep=(lab[t]>=0)&~bad[t]
        with pth.with_suffix('.partial').open('wb') as f:
            np.savez_compressed(f,t=t,y=lab[t],p=np.array([o.p_task for o in outputs]),keep=keep,evaluated=np.array([o.evaluated for o in outputs]),window_samples=np.array([o.window_samples for o in outputs]),reason=np.array([o.reason for o in outputs]),seconds=seconds,fold=k,record=r.rec_id,pattern=pattern)
        pth.with_suffix('.partial').replace(pth)
        print('stream',k,r.rec_id,pattern,round(seconds,2),flush=True)
    return r.rec_id

def aggregate(sp,t0):
    by={}; records=[]; held=[]; costs=[]
    for k,fold in enumerate(sp['folds']):
        for rid in fold:
            for pat in ['real']+PATTERNS:
                z=np.load(PRED/f'f{k}'/f'{rid}_{pat}.npz'); m=z['keep']; y=z['y'][m]; p=z['p'][m]
                by.setdefault(pat,([],[],[]))
                for a,v in zip(by[pat],(y,p,np.full(len(y),rid))): a.append(v)
                if pat=='real':
                    held.extend((~z['evaluated'][m]).tolist()); costs.append(float(z['seconds'])/len(z['t']))
                records.append(dict(fold=k,record=rid,pattern=pat,n=len(y),auc=roc_auc_score(y,p),bacc=balanced_accuracy_score(y,p>=.5)))
    auc={}; bacc={}; rec_auc={}
    for pat,(ys,ps,rs) in by.items():
        y,p,r=map(np.concatenate,(ys,ps,rs)); auc[pat]=roc_auc_score(y,p); bacc[pat]=balanced_accuracy_score(y,p>=.5)
        rec_auc[pat]=float(np.mean([roc_auc_score(y[r==q],p[r==q]) for q in np.unique(r)]))
    y,p,r=map(np.concatenate,by['real']); real=evaluate(y,p,r)
    A=np.array([auc[p] for p in PATTERNS]); B=np.array([bacc[p] for p in PATTERNS])
    out=dict(order20_mean_auc=float(A.mean()),order20_mean_bacc=float(B.mean()),order20_score=float((A.mean()+B.mean())/2),order20_min_auc=float(A.min()),order20_min_bacc=float(B.min()),order20_max_auc=float(A.max()),worst_pattern=PATTERNS[int(np.argmin(A+B))],order20_rec_auc_mean=float(np.mean([rec_auc[p] for p in PATTERNS])),infer_ms=float(np.mean(costs)*1000),held_frac=float(np.mean(held)),**real)
    Path('results/order20').mkdir(exist_ok=True)
    Path('results/order20/local_D2_cv.json').write_text(json.dumps(dict(auc=auc,bacc=bacc,rec_auc=rec_auc,**out),indent=2,default=float))
    import pandas as pd
    pd.DataFrame(records).to_csv('results/local_record_metrics.csv',index=False)
    log_experiment('local_D2_cv','recipe_D2','cv5_dev46_stream_order20',0,dict(recipe='D2',seeds=3,chunk=250),out,train_s=time.perf_counter()-t0,compliant='causal-online-adaptation')
    print(json.dumps(out,indent=2,default=float),flush=True)

def main():
    os.environ['D2_LOW_MEMORY']='1'
    t0=time.perf_counter(); BASE.mkdir(parents=True,exist_ok=True)
    sp=json.loads(Path('configs/splits.json').read_text()); dev=set(sum(sp['folds'],[]))
    recipe=json.loads(Path('configs/recipes.json').read_text())['D2']
    paths={Path(b).parent.name:(b,c) for b,c in list_recordings('EEG_dataset') if Path(b).parent.name in dev}
    # Fingerprint all algorithm sources plus training input caches. Resume only the same run.
    digests={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in ['src','scripts','configs'] for p in Path(root).rglob('*') if p.suffix in ['.py','.json']}
    digests.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('cache').glob('feat2_w*_s1.npz')})
    manifest=dict(digests=digests,recipe=recipe,splits=sp,seeds=3,threshold=.5)
    mp=BASE/'manifest.json'
    if mp.exists(): assert json.loads(mp.read_text())==manifest,'Inputs changed; use a new run directory'
    else: mp.write_text(json.dumps(manifest,indent=2))
    for k,test in enumerate(sp['folds']):
        pool=sum([f for j,f in enumerate(sp['folds']) if j!=k],[]); dest=BASE/f'f{k}'
        if not (dest/'config.json').exists():
            print('train fold',k,flush=True); ts=time.perf_counter()
            data=load_features(recipe['features'])
            recs={rid:load_recording(*paths[rid]) for rid in pool}
            train_recipe(pool,str(dest),recipe,data,recs,seeds=3,n_jobs=1,lgb_jobs=4)
            (dest/'training.json').write_text(json.dumps(dict(train=pool,validation=test,seconds=time.perf_counter()-ts),indent=2))
            del data,recs; gc.collect()
        assert set(json.loads((dest/'config.json').read_text())['trained_on'])==set(pool)
        with cf.ProcessPoolExecutor(max_workers=2) as ex:
            fs=[ex.submit(stream_record,k,*paths[rid]) for rid in test]
            for f in cf.as_completed(fs): print('record done',k,f.result(),flush=True)
        print('fold done',k,round(time.perf_counter()-t0,1),flush=True)
    aggregate(sp,t0)
    (BASE/'complete.json').write_text(json.dumps(dict(seconds=time.perf_counter()-t0)))

if __name__=='__main__': main()
