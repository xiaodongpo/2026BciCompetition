"""Real StreamingDecoder over whole recordings, under all 20 relax/task trial orders.

The raw signal is rebuilt as: pre-experiment samples, then the trials in the new
order. A single NaN sample separates consecutive pieces so no window spans an
artificial splice (the decoder treats it as a gap). The same is done for the
original order, so all 20 orders are scored identically. `--real` also scores
the untouched original recording.

mode cv:      five folds of dev recordings.
mode lockbox: train on all dev recordings and score the 11 lockbox recordings.
"""
import argparse
import itertools
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.getcwd())
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from scripts.train_final import load_features, train_recipe
from src.data.io import load_all
from src.evaluation.metrics import evaluate
from src.evaluation.splits import load_splits
from src.inference.decoder import StreamingDecoder
from src.utils.exp_log import log_experiment

PATTERNS = sorted({"".join(p) for p in itertools.permutations("rrrttt")})
ap = argparse.ArgumentParser()
ap.add_argument("--recipe", required=True)
ap.add_argument("--mode", default="cv", choices=["cv", "lockbox"])
ap.add_argument("--tag", required=True)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--chunk", type=int, default=250)
ap.add_argument("--jobs", type=int, default=6)
ap.add_argument("--model_dir", default=None, help="lockbox only: score this already-trained directory instead of training")
a = ap.parse_args()
recipe = json.load(open("configs/recipes.json"))[a.recipe]
sp = load_splits()
data = load_features(recipe.get("features"))
recs = {r.rec_id: r for r in load_all("EEG_dataset")}


def build(rid, pattern):
    r = recs[rid]
    x = r.x.astype(np.float64).copy()
    x[r.bad] = np.nan
    lab = r.label_array()
    if pattern is None:
        return x, lab, r.bad.copy()
    rr = [t for t in r.trials if t.label == 0]
    tt = [t for t in r.trials if t.label == 1]
    order = [rr.pop(0) if c == "r" else tt.pop(0) for c in pattern]
    xs, ls = [x[: r.trials[0].start]], [np.full(r.trials[0].start, -1)]
    for t in order:
        xs += [np.full(1, np.nan), x[t.start : t.end]]
        ls += [np.full(1, -1), np.full(t.end - t.start, t.label)]
    x, lab = np.concatenate(xs), np.concatenate(ls)
    return x, lab, ~np.isfinite(x)


def stream(rid, pattern, d):
    x, lab, bad = build(rid, pattern)
    dec = StreamingDecoder(d)
    out = []
    t0 = time.perf_counter()
    for i in range(0, len(x), a.chunk):
        out += dec.push(x[i : i + a.chunk])
    el = time.perf_counter() - t0
    t = np.array([o.sample_index for o in out])
    p = np.array([o.p_task for o in out])
    keep = (lab[t] >= 0) & ~bad[t]
    return rid, pattern or "real", lab[t][keep], p[keep], el / max(len(out), 1)


tmp = tempfile.mkdtemp()
if a.mode == "lockbox":
    tasks = [(sum(sp["folds"], []), sp["holdout"])]
else:
    tasks = [(sum([f for j, f in enumerate(sp["folds"]) if j != k], []), fold) for k, fold in enumerate(sp["folds"])]
res = []
t00 = time.time()
for k, (pool, test) in enumerate(tasks):
    assert not set(pool) & set(test)
    d = a.model_dir if (a.model_dir and a.mode == "lockbox") else f"{tmp}/f{k}"
    if d.startswith(tmp):
        train_recipe(pool, d, recipe, data, recs, seeds=a.seeds)
    jobs = [(r, p) for r in test for p in PATTERNS + [None]]
    res += Parallel(n_jobs=a.jobs)(delayed(stream)(r, p, d) for r, p in jobs)
    print("part", k, "done", f"{time.time() - t00:.0f}s", flush=True)
shutil.rmtree(tmp)

by = {}
for rid, pat, yy, pp, c in res:
    by.setdefault(pat, ([], [], [], []))
    for lst, v in zip(by[pat], (yy, pp, np.full(len(yy), rid), [c])):
        lst.append(v)
auc, bacc, recauc = {}, {}, {}
for pat, (Y, P, R, C) in by.items():
    y, p, r = np.concatenate(Y).astype(int), np.concatenate(P), np.concatenate(R)
    auc[pat], bacc[pat] = roc_auc_score(y, p), balanced_accuracy_score(y, p >= 0.5)
    recauc[pat] = float(np.mean([roc_auc_score(y[r == q], p[r == q]) for q in np.unique(r)]))
A = np.array([auc[p] for p in PATTERNS])
B = np.array([bacc[p] for p in PATTERNS])
yr, pr, rr_ = (np.concatenate(v) for v in by["real"][:3])
real = evaluate(yr.astype(int), pr, rr_)
ms = float(np.mean(np.concatenate([np.array(v) for v in by["real"][3]])) * 1000)
out = dict(order20_mean_auc=float(A.mean()), order20_mean_bacc=float(B.mean()), order20_score=float((A.mean() + B.mean()) / 2), order20_min_auc=float(A.min()), order20_min_bacc=float(B.min()), order20_max_auc=float(A.max()),
           worst_pattern=PATTERNS[int(np.argmin(A + B))], order20_rec_auc_mean=float(np.mean([recauc[p] for p in PATTERNS])), infer_ms=ms, **real)
split = "lockbox11_stream_order20" if a.mode == "lockbox" else "cv5_dev46_stream_order20"
log_experiment(a.tag, f"recipe_{a.recipe}", split, 0, dict(recipe=recipe, seeds=a.seeds, chunk=a.chunk), out, compliant="yes-online-adapt")
os.makedirs("results/order20", exist_ok=True)
json.dump(dict(auc=auc, bacc=bacc, rec_auc=recauc, **{k: v for k, v in out.items() if not isinstance(v, (np.integer,))}), open(f"results/order20/{a.tag}.json", "w"), indent=1, default=float)
print(f"{a.tag}: 20 orders mean AUC {A.mean():.4f} bACC {B.mean():.4f} | min AUC {A.min():.4f} bACC {B.min():.4f} | max AUC {A.max():.4f} | worst {out['worst_pattern']}")
print(f"{a.tag}: real recording AUC {real['pooled_auc']:.4f} bACC {real['pooled_bacc']:.4f} sens {real['pooled_sens']:.3f} spec {real['pooled_spec']:.3f} F1 {real['pooled_f1']:.3f} recAUC {real['rec_auc_mean']:.4f}+-{real['rec_auc_std']:.3f} | {ms:.1f} ms/decision")
print("confusion (real) tn fp fn tp", real["pooled_tn"], real["pooled_fp"], real["pooled_fn"], real["pooled_tp"])
for p in PATTERNS:
    print(f"  {p} AUC {auc[p]:.4f} bACC {bacc[p]:.4f}")
