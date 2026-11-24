"""Rebuild imported feature definitions on the 46 fixed development records only."""
import hashlib
import importlib.metadata as md
import json
import platform
import time
from pathlib import Path
import numpy as np
from src.data.io import list_recordings, load_recording
from src.data.windows import index_windows, extract_windows
from src.preprocessing.features import extract_v2, feature_names_v2


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    t0 = time.perf_counter()
    sp = json.loads(Path('configs/splits.json').read_text())
    dev = set(sum(sp['folds'], []))
    inputs = {str(Path(p).resolve()): digest(p) for pair in list_recordings('EEG_dataset') for p in pair}
    source = {str(p): digest(p) for root in ['src', 'scripts', 'configs'] for p in Path(root).rglob('*') if p.suffix in ['.py', '.json']}
    Path('cache/shards').mkdir(parents=True, exist_ok=True)
    Path('cache/recordings').mkdir(exist_ok=True)
    for b,c in list_recordings('EEG_dataset'):
        rid = Path(b).parent.name
        if rid not in dev:
            continue
        r = load_recording(b,c)
        np.savez(f'cache/recordings/{rid}.npz', x=r.x, gaps=np.array(r.gaps, dtype=np.int64).reshape(-1,2))
        for sec in [6,2]:
            dest = Path(f'cache/shards/{rid}_w{sec}.npz')
            if dest.exists():
                continue
            ix = index_windows(r, sec*250, 250)
            x = np.nan_to_num(r.x)
            fs = [extract_v2(extract_windows(x, ix.ends[s:s+128], sec*250)) for s in range(0,len(ix.ends),128)]
            np.savez(dest, F=np.concatenate(fs),y=ix.y,rec=np.full(len(ix.y),rid),end=ix.ends,clean=ix.clean,pure=ix.pure,valid=ix.valid_t)
        print('prepared',rid,round(time.perf_counter()-t0,1),flush=True)
    for sec in [6,2]:
        zs = [np.load(f'cache/shards/{rid}_w{sec}.npz') for rid in sorted(dev)]
        arrays = {k:np.concatenate([z[k] for z in zs]) for k in ['F','y','rec','end','clean','pure','valid']}
        np.savez(f'cache/feat2_w{sec}_s1.npz', **arrays, names=np.array(feature_names_v2()))
        print('features',sec,arrays['F'].shape,flush=True)
    meta = dict(inputs=inputs,source=source,development=sorted(dev),holdout_excluded=sp['holdout'],platform=platform.platform(),versions={n:md.version(n) for n in ['numpy','scipy','pandas','scikit-learn','mne','lightgbm','joblib','pytest']},runtime_s=time.perf_counter()-t0)
    Path('reports/prepare.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

if __name__ == '__main__':
    main()
