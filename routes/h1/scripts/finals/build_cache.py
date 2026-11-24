"""Finals caches: every recording is passed through the submission's causal filter, then
features are computed for 6-s windows on a 1-s grid (window i starts at sample 250*i).

Writes cache/fin/feat.npz (252 features: 6-s view + last 2 s) and, with --bpn,
cache/fin/featbpn.npz (same features after the in-window 1-45 Hz band-pass + 50 Hz notch).
Also stores the filtered signals in cache/fin/x/<rec>.npy for waveform models.

clean_sig: no unusable sample found from the signal alone (what predict can know).
clean:     clean_sig and no annotated gap in the window, with a 0.5-s guard (training only).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "finals", "submission"))
import numpy as np

from fin.features import extract_multi
from fin.filtering import causal_lowpass
from fin.quality import invalid_mask
from src.data.io import load_all
from src.preprocessing.filters import bandpass_notch

ap = argparse.ArgumentParser()
ap.add_argument("--bpn", action="store_true")
a = ap.parse_args()
WIN, STEP, VIEWS = 1500, 250, (1500, 500)
os.makedirs("cache/fin/x", exist_ok=True)
out = dict(F=[], y=[], rec=[], start=[], clean=[], clean_sig=[], pure=[])
t0 = time.time()
for r in load_all("EEG_dataset"):
    xf = causal_lowpass(r.x.astype(np.float64))
    np.save(f"cache/fin/x/{r.rec_id}.npy", xf.astype(np.float32))
    bad_sig = invalid_mask(xf)
    bad_ann = r.bad.copy()
    d = np.diff(np.concatenate([[0], bad_ann.astype(np.int8), [0]]))
    for e in np.flatnonzero(d == -1):
        bad_ann[e : e + 125] = True
    lab = r.label_array()
    tid = np.full(r.n, -1, np.int32)
    for i, t in enumerate(r.trials):
        tid[t.start : t.end] = i
    starts = np.arange(0, r.n - WIN + 1, STEP)
    cs, ca = np.concatenate([[0], np.cumsum(bad_sig)]), np.concatenate([[0], np.cumsum(bad_ann)])
    clean_sig = (cs[starts + WIN] - cs[starts]) == 0
    clean = clean_sig & ((ca[starts + WIN] - ca[starts]) == 0)
    pure = (tid[starts] == tid[starts + WIN - 1]) & (tid[starts] >= 0)
    feats = []
    for s in range(0, len(starts), 512):
        w = xf[starts[s : s + 512, None] + np.arange(WIN)]
        if a.bpn:
            w = bandpass_notch(w - w[:, :1])
        feats.append(extract_multi(w, VIEWS))
    out["F"].append(np.concatenate(feats)); out["y"].append(np.where(pure, lab[starts], -1).astype(np.int8))
    out["rec"].append(np.full(len(starts), r.rec_id)); out["start"].append(starts)
    out["clean"].append(clean); out["clean_sig"].append(clean_sig); out["pure"].append(pure)
    print(r.rec_id, len(starts), f"sig-bad windows {(~clean_sig).mean():.3f} train-bad {(~clean).mean():.3f} {time.time() - t0:.0f}s", flush=True)
path = "cache/fin/featbpn.npz" if a.bpn else "cache/fin/feat.npz"
np.savez(path, **{k: np.concatenate(v) for k, v in out.items()})
print("saved", path, np.concatenate(out["F"]).shape)
