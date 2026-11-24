import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import windows
from src.features import extract_features

def run():
    records=sorted(p.stem for p in Path('cache/records').glob('*.npz'))
    # Freeze split before feature extraction or model fitting; no participant identity is assumed.
    ranked=sorted(records,key=lambda r:hashlib.sha256(('bci2026-seed42-'+r).encode()).hexdigest())
    holdout=sorted(ranked[:12]); dev=sorted(ranked[12:])
    folds=[sorted(ranked[12+i::3]) for i in range(3)]
    split=dict(seed=42,group='record',holdout=holdout,development=dev,folds=folds)
    sp=Path('configs/splits.json')
    if sp.exists(): assert json.loads(sp.read_text())==split
    else:sp.write_text(json.dumps(split,indent=2))
    for seconds in (2,4,6):
        feats=[];meta=[]; raw=[]
        for rid in records:
            d=np.load('cache/records/'+rid+'.npz')
            rows=list(windows(d['x'],d['valid'],d['labels'],seconds,2))
            if not rows:continue
            x=np.stack([r[0] for r in rows])
            feats.append(extract_features(x))
            meta.extend([dict(record=rid,y=y,end=end) for _,y,end in rows])
            if seconds==6:raw.append(x)
        np.save(f'cache/features_{seconds}s.npy',np.concatenate(feats))
        pd.DataFrame(meta).to_csv(f'cache/windows_{seconds}s.csv',index=False)
        if raw:np.save('cache/raw_6s.npy',np.concatenate(raw))
        print(seconds,len(meta),flush=True)
if __name__=='__main__':run()
