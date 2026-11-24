"""Compute and cache stateless window features for every recording."""
import argparse, os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np
from src.data.io import load_all, FS
from src.data.windows import index_windows, extract_windows
from src.preprocessing.features import extract, extract_v2, extract_v3, feature_names, feature_names_v2, feature_names_v3
ap = argparse.ArgumentParser(); ap.add_argument("--win", type=float, default=6.0); ap.add_argument("--step", type=float, default=1.0); ap.add_argument("--v2", action="store_true"); ap.add_argument("--v3", action="store_true")
a = ap.parse_args()
win, step = int(a.win * FS), int(a.step * FS)
fn, names = (extract_v3, feature_names_v3()) if a.v3 else (extract_v2, feature_names_v2()) if a.v2 else (extract, feature_names())
out = f"cache/feat{'3' if a.v3 else '2' if a.v2 else ''}_w{a.win:g}_s{a.step:g}.npz"
F, Y, R, E, C, P, V = [], [], [], [], [], [], []
t0 = time.time()
for r in load_all("EEG_dataset"):
    ix = index_windows(r, win, step)
    x = np.nan_to_num(r.x)
    feats = [fn(extract_windows(x, ix.ends[s:s + 512], win)) for s in range(0, len(ix.ends), 512)]
    F.append(np.concatenate(feats)); Y.append(ix.y); R.append(np.full(len(ix.y), r.rec_id)); E.append(ix.ends); C.append(ix.clean); P.append(ix.pure); V.append(ix.valid_t)
np.savez(out, F=np.concatenate(F), y=np.concatenate(Y), rec=np.concatenate(R), end=np.concatenate(E), clean=np.concatenate(C), pure=np.concatenate(P), valid=np.concatenate(V), names=np.array(names))
print(out, np.concatenate(F).shape, len(names), f"{time.time()-t0:.1f}s")
