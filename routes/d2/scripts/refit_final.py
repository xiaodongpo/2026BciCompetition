"""Frozen all-record D2 final fitting only; never a validation run."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT
import hashlib,json,time
import numpy as np
from scripts.train_final import train_recipe
from src.data.io import list_recordings,load_recording
from src.data.windows import index_windows,extract_windows
from src.preprocessing.features import extract_v2,feature_names_v2

def main():
    os.chdir(D); os.environ['D2_LOW_MEMORY']='1'; t0=time.perf_counter()
    Path('cache/shards').mkdir(parents=True,exist_ok=True)
    dest=Path('checkpoints/final_D2_local')
    if (dest/'complete.json').exists():
        print('Final model already complete'); return
    recs={}
    for b,c in list_recordings('EEG_dataset'):
        rid=Path(b).parent.name; recs[rid]=load_recording(b,c); r=recs[rid]
        for sec in [6,2]:
            p=Path(f'cache/shards/{rid}_w{sec}.npz')
            if p.exists(): continue
            ix=index_windows(r,sec*250,250); x=np.nan_to_num(r.x)
            F=np.concatenate([extract_v2(extract_windows(x,ix.ends[s:s+128],sec*250)) for s in range(0,len(ix.ends),128)])
            np.savez(p,F=F,y=ix.y,rec=np.full(len(ix.y),rid),end=ix.ends,clean=ix.clean,pure=ix.pure,valid=ix.valid_t)
        print('final inputs',rid,flush=True)
    assembled={}
    Path('cache/final').mkdir(exist_ok=True)
    for sec in [6,2]:
        zs=[np.load(f'cache/shards/{rid}_w{sec}.npz') for rid in sorted(recs)]
        assembled[sec]={k:np.concatenate([z[k] for z in zs]) for k in ['F','y','rec','end','clean','pure','valid']}
        np.savez(f'cache/final/feat2_w{sec}_s1.npz',**assembled[sec])
    a,b=assembled[6],assembled[2]
    index={(r,int(e)):i for i,(r,e) in enumerate(zip(b['rec'],b['end']))}
    sel=np.array([index[(r,int(e))] for r,e in zip(a['rec'],a['end'])])
    names=feature_names_v2()+['w2s_'+n for n in feature_names_v2()]
    data=(np.concatenate([a['F'],b['F'][sel]],axis=1).astype(np.float32),a['y'],a['rec'],a['clean'],a['pure'],a['end'],names)
    del assembled,a,b,index,sel
    recipe=json.loads(Path('configs/recipes.json').read_text())['D2']
    print('training final D2; 57 records, 5 seeds/member; no validation or tuning',flush=True)
    cfg=train_recipe(sorted(recs),str(dest),recipe,data,recs,seeds=5,n_jobs=1,lgb_jobs=4)
    inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('cache/final').glob('*.npz')}
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for root in [D/'src',D/'scripts',D/'configs'] for p in root.rglob('*') if p.suffix in ['.py','.json']}
    sources[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    done=dict(seconds=time.perf_counter()-t0,trained_on=sorted(recs),seeds=5,threshold=cfg['threshold'],inputs=inputs,sources=sources,validation_performed=False,recipe=recipe)
    (dest/'complete.json').write_text(json.dumps(done,indent=2))
    print(json.dumps({'complete':str(dest),'seconds':done['seconds'],'records':len(recs)}),flush=True)
if __name__=='__main__':main()
