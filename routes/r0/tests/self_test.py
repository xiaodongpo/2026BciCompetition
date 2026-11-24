"""接口、分块不变、未来不变、记录重置与有界信号路径核验，无需训练数据。"""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from r0 import R0Decoder

def run(d,x,n,mask=None):
    rows=[]
    for i in range(0,len(x),n):rows+=d.push(x[i:i+n],None if mask is None else mask[i:i+n])
    return rows

def scores(rows):return np.array([r['score'] for r in rows])

if __name__=='__main__':
    x=np.load(ROOT/'example/raw_eeg_uv.npy');d=R0Decoder(ROOT/'model','cpu');begin=time.perf_counter()
    a=run(d,x,137);d.reset();b=run(d,x,5000)
    assert len(a)==32 and np.array_equal(scores(a),scores(b))
    d.reset();assert np.array_equal(scores(a),scores(run(d,x,1)))
    xx=x.copy();xx[5000:]+=1000000;d.reset();future=run(d,xx,431)
    assert np.array_equal(scores([r for r in a if r['end_sample']<=5000]),scores([r for r in future if r['end_sample']<=5000]))
    mask=np.ones(len(x),bool);mask[1500:1750]=False;d.reset();bad=run(d,x,137,mask)
    assert all(not r['valid'] for r in bad if r['end_sample']>1500 and r['end_sample']-1000<1750)
    assert all(r['state']==int(r['score']>=.5) for r in bad)
    d.reset();d.freeze_adaptation=True
    old1=run(d,np.r_[np.full(4000,40000.),x[-1500:]],250)[-1]['score']
    d.reset();d.freeze_adaptation=True
    old2=run(d,np.r_[np.full(4000,80000.),x[-1500:]],250)[-1]['score']
    assert abs(old1-old2)<1e-7,(old1,old2)
    d.freeze_adaptation=False;d.reset();assert np.array_equal(scores(a),scores(run(d,x,137)))
    report=dict(status='PASS',chunk_invariance=True,sample_by_sample_invariance=True,future_invariance=True,
       reset_repeat=True,explicit_invalid_mask_respected=True,threshold_consistent=True,
       frozen_adaptation_older_signal_invariance=True,raw_buffer_samples_max=1500,
       seconds=time.perf_counter()-begin,decisions=len(a),score_min=float(scores(a).min()),score_max=float(scores(a).max()))
    output=ROOT/'test_results';output.mkdir(exist_ok=True)
    (output/'self_test.json').write_text(json.dumps(report,indent=2))
    print('SELF_TEST_PASS',json.dumps(report))
