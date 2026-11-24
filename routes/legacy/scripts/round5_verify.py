"""Saved-model support/streaming verification and post-lock reversal diagnostic."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from src.round5_features import get_features
from src.round5_inference import RoundFivePredictor
from src.round5_model import predict
from scripts.round2_evaluate import measures


def verify():
    raw=np.load('cache/round2/raw.npy',mmap_mode='r');rng=np.random.default_rng(20261104);idx=rng.choice(len(raw),20,replace=False)
    protocol=json.loads(Path('configs/round5/protocol.json').read_text());results={}
    paths={c['id']:Path('experiments/round5/final_selection')/(c['id']+'_inner0.joblib') for c in protocol['candidates']}
    paths['final_candidate']=Path('checkpoints/round5/research_candidate.joblib')
    signal=rng.normal(0,50,4000).astype('float32')
    for name,path in paths.items():
        bundle=joblib.load(path);c=bundle['config'];adapter=RoundFivePredictor(path)
        cached=np.load(f"cache/round5/{c['features']}.npy",mmap_mode='r');expected=predict(bundle['model'],cached[idx])
        with threadpool_limits(limits=4):
            start=time.perf_counter();actual=np.array([adapter.predict_window(raw[i]) for i in idx]);ms=(time.perf_counter()-start)/len(idx)*1000
            err=float(np.max(abs(actual-expected)));assert err<5e-4,(name,err)
            full=adapter.process(signal);adapter.reset();chunked=[]
            for a in range(0,len(signal),173):chunked.extend(adapter.process(signal[a:a+173]))
            assert full==chunked
            adapter.reset();single=[]
            for sample in signal:single.extend(adapter.process([sample]))
            assert full==single
            changed=signal.copy();changed[2500:]*=100;adapter.reset();future=adapter.process(changed)
            assert [r for r in full if r['sample_index']<2500]==[r for r in future if r['sample_index']<2500]
            changed=signal.copy();changed[:2000]*=100;adapter.reset();past=adapter.process(changed)
            assert [r for r in full if r['sample_index']>=3499]==[r for r in past if r['sample_index']>=3499]
            mask=np.ones(4000,bool);mask[1800]=False;adapter.reset();gap=adapter.process(signal,valid=mask)
            assert all(not r['valid'] for r in gap if 1800<=r['sample_index']<3300)
            assert all(r['valid'] for r in gap if r['sample_index']>=3300)
            adapter.reset();adapter.process(signal[:1000]);jump=adapter.process(signal[:1000],start_sample=2000)
            assert not any(r['valid'] for r in jump)
            try:adapter.process(signal[:10],start_sample=0)
            except ValueError:pass
            else:raise AssertionError('Out of order accepted')
        results[name]=dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cache_probability_max_abs_error=err,cpu_mean_ms_20_windows=ms,checks=['cache consistency','chunk partition','single sample reference','future isolation','pre-buffer history isolation','invalid reset','sample gap reset','out-of-order rejection'])
    Path('reports/round5/verification.json').write_text(json.dumps(results,indent=2));print('Verified',len(results),'saved-model adapters',flush=True)


def reversal():
    protocol=json.loads(Path('configs/round5/protocol.json').read_text());meta=pd.read_csv('cache/round5/windows.csv',dtype={'record':str})
    raw=np.load('cache/round2/raw.npy',mmap_mode='r');index=np.concatenate([g.index.to_numpy()[::32] for _,g in meta.groupby('record')]);frame=meta.loc[index].copy();rows=[]
    for scenario in ('identity','time_reverse'):
        x=np.asarray(raw[index],dtype=np.float64)
        if scenario=='time_reverse':x=x[:,::-1]
        represented={}
        for k,valid in enumerate(protocol['outer_folds']):
            take=frame.record.isin(valid).to_numpy()
            for kind in ('selected','reference'):
                bundle=joblib.load(Path('experiments/round5/record_nested')/f'outer{k}'/(kind+'.joblib'));c=bundle['config']
                if c['features'] not in represented:represented[c['features']]=np.concatenate([get_features(x[a:a+128],c['features']) for a in range(0,len(x),128)])
                p=predict(bundle['model'],represented[c['features']][take]);part=frame[take].copy();part['scenario']=scenario;part['kind']=kind;part['p']=p;part['prediction']=(p>=bundle['threshold']).astype(int);rows.append(part)
    all_rows=pd.concat(rows);all_rows.to_csv('results/round5/reversal_predictions.csv',index=False);summary=[]
    for (scenario,kind),g in all_rows.groupby(['scenario','kind']):
        records=[measures(f) for _,f in g.groupby('record')]
        summary.append(dict(scenario=scenario,kind=kind,n=len(g),macro_bacc=float(np.mean([r['bacc'] for r in records])),macro_auc=float(np.mean([r['auc'] for r in records]))))
    pd.DataFrame(summary).to_csv('results/round5/reversal_summary.csv',index=False);print('Post-lock reversal check complete',flush=True)


if __name__=='__main__':verify();reversal()
