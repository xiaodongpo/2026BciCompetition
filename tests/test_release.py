import importlib.util,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from scripts.synthetic_smoke import train_bundle,synthetic_signal,R0Decoder
from scripts.evaluate_predictions import evaluate
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def model(tmp_path_factory):
    path=tmp_path_factory.mktemp('r0_synthetic')/'model';report,_=train_bundle(path)
    assert report['status']=='PASS';return path

def run(model,x,n,mask=None,freeze=False):
    d=R0Decoder(model);d.freeze_adaptation=freeze;rows=[]
    for i in range(0,len(x),n):
        rows+=d.push(x[i:i+n],None if mask is None else mask[i:i+n])
        assert len(d.buffer)<=1500
    return rows

def test_r0_checkpoint_reload_and_direction(model):
    a=run(model,synthetic_signal(),250)
    assert len(a)==32 and all(r['valid'] and 0<=r['score']<=1 and r['state']==int(r['score']>=.5) for r in a)

@pytest.mark.parametrize('chunk',[1,137,5000,8750])
def test_r0_chunk_invariance(model,chunk):
    x=synthetic_signal();assert run(model,x,chunk)==run(model,x,250)

def test_r0_future_perturbation(model):
    x=synthetic_signal();z=x.copy();z[5000:]+=1e6
    assert [r for r in run(model,x,8750) if r['end_sample']<=5000]==[r for r in run(model,z,8750) if r['end_sample']<=5000]

def test_r0_invalid_mask_and_reset(model):
    x=synthetic_signal();mask=np.ones(len(x),bool);mask[1500:1750]=False
    a=run(model,x,250,mask)
    assert all(not r['valid'] for r in a if r['end_sample']>1500 and r['end_sample']-1000<1750)
    d=R0Decoder(model);expected=d.push(x);d.reset();assert d.push(x)==expected

def test_r0_boundary_and_frozen_signal_support(model):
    d=R0Decoder(model);assert d.push(np.empty(0))==[]
    with pytest.raises(ValueError):d.push(np.zeros((2,10)))
    assert all(not r['valid'] for r in d.push(np.zeros(1500)))
    x=synthetic_signal();z=x.copy();z[:-1500]+=12345
    assert abs(run(model,x,250,freeze=True)[-1]['score']-run(model,z,250,freeze=True)[-1]['score'])<1e-7

def test_metrics_direction_and_accuracy_difference():
    y=[0,0,0,1];p=[.1,.2,.6,.8];r=evaluate(y,p)
    assert r['auc']==1 and r['accuracy']==.75
    assert r['bacc']==pytest.approx(5/6)
    assert evaluate(y,1-np.array(p))['auc']==0
    for y,p in [([0,0],[.1,.2]),([0,1],[.1,np.nan]),([0,1],[.1,1.5]),([0,2],[.1,.2])]:
        with pytest.raises(ValueError):evaluate(y,p)

def test_frozen_split_disjointness():
    for rel,dev_size,hold_size in [('routes/legacy/configs/splits.json',45,12),('routes/d2/configs/splits.json',46,11)]:
        c=json.loads((ROOT/rel).read_text());folds=[set(f) for f in c['folds']];dev=set.union(*folds)
        assert len(dev)==dev_size and len(c['holdout'])==hold_size and not dev.intersection(c['holdout'])
        assert sum(map(len,folds))==len(dev)

def test_legacy_suite():
    p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT/'routes/legacy',env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,encoding='utf8')
    assert p.returncode==0,p.stdout+p.stderr

def test_d2_train_reload_and_stream(tmp_path):
    p=subprocess.run([sys.executable,str(ROOT/'scripts/d2_synthetic_smoke.py'),'--out',str(tmp_path/'d2')],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
    assert p.returncode==0,p.stdout+p.stderr

def test_r0_formal_training_entry_smoke(tmp_path):
    p=subprocess.run([sys.executable,str(ROOT/'routes/r0/train_from_scratch.py'),'--stage','smoke','--work-dir',str(tmp_path/'formal_smoke'),'--device','cpu'],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
    assert p.returncode==0,p.stdout+p.stderr

def test_h1_training_export_and_protocol(tmp_path):
    p=subprocess.run([sys.executable,str(ROOT/'scripts/h1_synthetic_smoke.py'),'--out',str(tmp_path/'h1')],cwd=ROOT,capture_output=True,text=True,encoding='utf8')
    assert p.returncode==0,p.stdout+p.stderr
