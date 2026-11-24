"""Exercise the real D2 trainer and decoder using synthetic features and raw signals."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'routes/d2'))
from scripts.train_final import train_recipe
from src.preprocessing.features import extract_multi
from src.inference.decoder import StreamingDecoder

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if (a.out/'config.json').exists():raise ValueError('Use a fresh output directory')
    rng=np.random.default_rng(42);raw=40000+rng.normal(0,20,(40,1500));y=np.arange(40)%2
    F=extract_multi(raw,[1500,500]);assert F.shape==(40,252)
    rec=np.repeat(['synthetic_a','synthetic_b'],20);clean=np.ones(40,bool);end=np.tile(np.arange(20)*250+1499,2)
    spec=dict(n_aug=0,params=dict(n_estimators=2,num_leaves=3,min_child_samples=2),seeds=1)
    recipe=dict(features=['cache/feat2_w6_s1.npz','cache/feat2_w2_s1.npz'],members=[dict(spec,view='rank',lr=False,weight=.8),dict(spec,view='abs',lr=True,weight=.2)])
    cfg=train_recipe(['synthetic_a'],str(a.out),recipe,data=(F,y,rec,clean,clean,end,[f'f{i}' for i in range(252)]),n_jobs=1,lgb_jobs=1)
    np.testing.assert_allclose(cfg['mu'],F[:20].mean(0))
    assert cfg['trained_on']==['synthetic_a']
    t=np.arange(8750)/250;x=40000+20*np.sin(2*np.pi*10*t)+rng.normal(0,3,len(t))
    def run(signal,chunk):
        d=StreamingDecoder(str(a.out));out=[]
        for i in range(0,len(signal),chunk):out.extend(d.push(signal[i:i+chunk]))
        return out
    first=run(x,137);second=run(x,8750)
    assert [r.p_task for r in first]==[r.p_task for r in second]
    xx=x.copy();xx[5000:]+=999999;future=run(xx,8750)
    assert [r.p_task for r in first if r.sample_index<5000]==[r.p_task for r in future if r.sample_index<5000]
    assert all(0<=r.p_task<=1 and r.state==int(r.p_task>=.5) and r.window_samples<=1500 for r in first)
    report=dict(status='PASS',synthetic=True,dimensions=252,train_only_prior=True,trained_members=2,chunk_invariance=True,future_invariance=True,decisions=len(first),full_recipe_reproduced=False)
    (a.out/'smoke_report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
if __name__=='__main__':main()
