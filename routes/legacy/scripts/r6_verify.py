"""Verify actual saved adapters in separate LR and Torch processes."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def check(adapter,signal):
    whole=adapter.process(signal);adapter.reset();parts=[]
    for a in range(0,len(signal),173):parts.extend(adapter.process(signal[a:a+173]))
    assert whole==parts
    adapter.reset();single=[]
    for value in signal:single.extend(adapter.process([value]))
    assert single==whole
    changed=signal.copy();changed[2500:]*=100;adapter.reset();future=adapter.process(changed)
    assert [r for r in whole if r['sample_index']<2500]==[r for r in future if r['sample_index']<2500]
    changed=signal.copy();changed[:2000]*=100;adapter.reset();past=adapter.process(changed)
    assert [r for r in whole if r['sample_index']>=3499]==[r for r in past if r['sample_index']>=3499]
    mask=np.ones(len(signal),bool);mask[1800]=False;adapter.reset();gap=adapter.process(signal,valid=mask)
    assert all(not r['valid'] for r in gap if 1800<=r['sample_index']<3300)
    assert all(r['valid'] for r in gap if r['sample_index']>=3300)
    adapter.reset();adapter.process(signal[:1000]);jump=adapter.process(signal[:1000],start_sample=2000);assert not any(r['valid'] for r in jump)
    try:adapter.process(signal[:10],start_sample=0)
    except ValueError:pass
    else:raise AssertionError('Out of order accepted')
    try:adapter.process(signal[:10],phase='task')
    except TypeError:pass
    else:raise AssertionError('Event input accepted')


def run(kind):
    raw=np.load('cache/round6/raw.npy',mmap_mode='r');meta=pd.read_csv('cache/round6/windows.csv',dtype={'record':str});rng=np.random.default_rng(20261104);signal=rng.normal(0,30,4000).astype('float32');rows=[]
    if kind=='lr':
        import joblib
        from src.round2_inference import RoundTwoPredictor as Adapter
        from src.transition_training import predict_lr
        paths=list(Path('experiments/round6/lr/final_selection').glob('*_inner0.joblib'))+[Path('checkpoints/round6/lr_candidate.joblib')];x=np.load('cache/round6/base.npy',mmap_mode='r')
    else:
        import torch
        torch.set_num_threads(4)
        from src.transition_inference import TransitionCNNPredictor as Adapter
        paths=list(Path('experiments/round6/cnn').glob('outer*/*/refit/best.pt'))
    for path in paths:
        adapter=Adapter(path);ids=rng.choice(len(meta),20,replace=False)
        with threadpool_limits(limits=4):
            start=time.perf_counter();actual=np.array([adapter.predict_window(raw[i]) for i in ids]);ms=(time.perf_counter()-start)/len(ids)*1000
            if kind=='lr':expected=predict_lr(joblib.load(path)['model'],x[ids]);error=float(np.max(abs(expected-actual)));assert error<5e-4
            else:
                logged=pd.read_csv(path.parent.parent/'outer_predictions.csv',dtype={'record':str});lookup=meta.reset_index().set_index(['record','end'])['index'];chosen=logged.iloc[::max(1,len(logged)//20)].head(20)
                ridx=lookup.loc[list(zip(chosen.record,chosen.end))].to_numpy();actual_oof=np.array([adapter.predict_window(raw[i]) for i in ridx]);error=float(np.max(abs(actual_oof-chosen.p.to_numpy())));assert error<2e-4,(path,error)
            check(adapter,signal)
        rows.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),max_probability_error=error,cpu_ms_per_window=ms,chunk_and_sample_reference=True,future_isolated=True,history_bounded=True,gaps_reset=True,events_rejected=True))
    Path(f'reports/round6/{kind}_stream_verification.json').write_text(json.dumps(rows,indent=2));print('Verified',kind,len(rows),'actual saved models',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['lr','cnn'],required=True);run(parser.parse_args().kind)
