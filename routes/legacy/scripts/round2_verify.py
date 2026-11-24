"""Executed consistency and bounded-support checks for saved research models."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import joblib
from src.round2_inference import RoundTwoPredictor


def run(stage):
    if stage=='neural':
        import torch
        torch.set_num_threads(4)
    kinds=('legacy','robust','filterbank') if stage=='classical' else ('singlem','singlem_random','fusion')
    raw=np.load('cache/round2/raw.npy',mmap_mode='r');rng=np.random.default_rng(20261103)
    indices=rng.choice(len(raw),12,replace=False);results={}
    for kind in kinds:
        path=Path('experiments/round2/final_selection')/(kind+'_record_C0.1_inner0.joblib')
        predictor=RoundTwoPredictor(path);model=joblib.load(path)['model']
        cached=np.load('cache/round2/'+kind+'.npy',mmap_mode='r')
        expected=model.predict_proba(cached[indices])[:,1]
        actual=np.array([predictor.predict_window(raw[i]) for i in indices])
        error=float(np.max(abs(expected-actual)))
        assert error<5e-4,(kind,error)
        start=time.perf_counter()
        for i in indices:predictor.predict_window(raw[i])
        latency=(time.perf_counter()-start)/len(indices)*1000
        signal=rng.normal(0,50,4000).astype('float32')
        predictor.reset();whole=predictor.process(signal)
        predictor.reset();blocked=[]
        for a in range(0,len(signal),173):blocked.extend(predictor.process(signal[a:a+173]))
        assert whole==blocked,'Block partition changes output'
        changed=signal.copy();changed[2500:]*=1000
        predictor.reset();future=predictor.process(changed)
        assert [r for r in whole if r['sample_index']<2500]==[r for r in future if r['sample_index']<2500]
        changed=signal.copy();changed[:2000]*=1000
        predictor.reset();past=predictor.process(changed)
        assert [r for r in whole if r['sample_index']>=3499]==[r for r in past if r['sample_index']>=3499]
        predictor.reset();mask=np.ones(4000,dtype=bool);mask[1800]=False
        gap=predictor.process(signal,valid=mask)
        assert all(not r['valid'] for r in gap if 1800<=r['sample_index']<3300)
        assert all(r['valid'] for r in gap if r['sample_index']>=3300)
        predictor.reset();predictor.process(signal[:1700]);jump=predictor.process(signal[:1000],start_sample=2000)
        assert not any(r['valid'] for r in jump)
        try:predictor.process(signal[:10],start_sample=1000)
        except ValueError:pass
        else:raise AssertionError('Out-of-order data accepted')
        predictor.reset();uv=predictor.process(signal)
        volts=RoundTwoPredictor(path,units='V').process(signal/1e6)
        np.testing.assert_allclose([r['p_task'] for r in uv],[r['p_task'] for r in volts],atol=5e-5,rtol=0)
        for bad in (np.zeros(1501),np.full(1500,np.nan)):
            try:predictor.predict_window(bad)
            except ValueError:pass
            else:raise AssertionError('Malformed window accepted')
        results[kind]=dict(cache_probability_max_abs_error=error,cpu_ms_per_window=latency,measured_windows=len(indices),tests=['cache equality','block partition','no future dependency','no pre-buffer dependency','invalid mask reset','sample gap reset','out-of-order rejection','volts conversion','input validation'],checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(kind,json.dumps(results[kind]),flush=True)
    Path(f'reports/round2/verification_{stage}.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['classical','neural'],required=True)
    run(parser.parse_args().stage)
