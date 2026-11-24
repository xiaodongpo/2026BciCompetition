"""Evaluate exported decoders through the real StreamingDecoder on whole recordings.

mode cv:       five folds of the dev recordings.
mode lockbox:  train on all dev recordings, score the 11 lockbox recordings. Run once, after freezing the choice.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.getcwd())
import numpy as np
from joblib import Parallel, delayed

from scripts.train_final import load_features, train_model_dir
from src.data.io import load_all
from src.evaluation.metrics import best_bacc_threshold, evaluate
from src.evaluation.splits import load_splits
from src.inference.decoder import StreamingDecoder
from src.utils.exp_log import log_experiment

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="cv", choices=["cv", "lockbox"])
ap.add_argument("--tag", required=True)
ap.add_argument("--params", default="configs/lgb_best.json")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--lr", action="store_true")
ap.add_argument("--order_aug", type=int, default=0)
ap.add_argument("--no_adapt", action="store_true")
ap.add_argument("--chunk", type=int, default=250)
a = ap.parse_args()
params = json.load(open(a.params))
sp = load_splits()
data = load_features()
recs = {r.rec_id: r for r in load_all("EEG_dataset")}


def stream_one(rid, d):
    r = recs[rid]
    dec = StreamingDecoder(d, adapt=not a.no_adapt)
    out = []
    t0 = time.perf_counter()
    for i in range(0, r.n, a.chunk):
        out += dec.push(r.x[i : i + a.chunk])
    el = time.perf_counter() - t0
    lab = r.label_array()
    t = np.array([o.sample_index for o in out])
    p = np.array([o.p_task for o in out])
    ev = np.array([o.evaluated for o in out])
    keep = (lab[t] >= 0) & ~r.bad[t]
    return rid, lab[t][keep], p[keep], ev[keep], el / max(len(out), 1)


tmp = tempfile.mkdtemp()
Y, P, R, E, cost, fold_of = [], [], [], [], [], []
tasks = [(sum(sp["folds"], []), sp["holdout"])] if a.mode == "lockbox" else [(sum([f for j, f in enumerate(sp["folds"]) if j != k], []), fold) for k, fold in enumerate(sp["folds"])]
t00 = time.time()
for k, (pool, test) in enumerate(tasks):
    assert not set(pool) & set(test)
    dd = f"{tmp}/f{k}"
    train_model_dir(pool, dd, params, a.seeds, a.lr, 0.5, a.order_aug, not a.no_adapt, 0.5, data, recs)
    for rid, yy, pp, ee, c in Parallel(n_jobs=6)(delayed(stream_one)(r, dd) for r in test):
        Y.append(yy); P.append(pp); R.append(np.full(len(yy), rid)); E.append(ee); cost.append(c); fold_of.append(np.full(len(yy), k))
    print("part", k, "done", f"{time.time() - t00:.0f}s", flush=True)
shutil.rmtree(tmp)
y, p, r, e, fo = map(np.concatenate, (Y, P, R, E, fold_of))
y = y.astype(int)
res = evaluate(y, p, r)
res["held_frac"] = float((~e).mean())
res["infer_ms"] = float(np.mean(cost) * 1000)
if a.mode == "cv":
    # Cross-fit threshold: choose on the other folds, apply here.
    pp = np.zeros(len(y), bool)
    thr = []
    for k in np.unique(fo):
        t = best_bacc_threshold(y[fo != k], p[fo != k])
        thr.append(t)
        pp[fo == k] = p[fo == k] >= t
    from sklearn.metrics import balanced_accuracy_score
    res["crossfit_thr_bacc"] = float(balanced_accuracy_score(y, pp))
    res["thr_mean"] = float(np.mean(thr))
print(f"{a.tag}: n={len(y)} AUC {res['pooled_auc']:.4f} bACC {res['pooled_bacc']:.4f} sens {res['pooled_sens']:.3f} spec {res['pooled_spec']:.3f} F1 {res['pooled_f1']:.3f} recAUC {res['rec_auc_mean']:.4f}+-{res['rec_auc_std']:.3f} recBACC {res['rec_bacc_mean']:.4f}+-{res['rec_bacc_std']:.3f} held {res['held_frac']:.3f} {res['infer_ms']:.1f} ms/decision crossfit_bACC {res.get('crossfit_thr_bacc', float('nan')):.4f} thr {res.get('thr_mean', float('nan')):.3f}")
print("confusion tn fp fn tp", res["pooled_tn"], res["pooled_fp"], res["pooled_fn"], res["pooled_tp"])
split = "lockbox11_stream_all" if a.mode == "lockbox" else "cv5_dev46_stream_all"
log_experiment(a.tag, "lgb" + ("+lr" if a.lr else ""), split, 0, dict(params=params, seeds=a.seeds, chunk=a.chunk, order_aug=a.order_aug, lr=a.lr, no_adapt=a.no_adapt), res, compliant="yes-online-adapt" if not a.no_adapt else "yes", infer_ms=res["infer_ms"])
os.makedirs("results/oof", exist_ok=True)
np.savez(f"results/oof/{a.tag}.npz", p=p, y=y, rec=r, ev=e)
