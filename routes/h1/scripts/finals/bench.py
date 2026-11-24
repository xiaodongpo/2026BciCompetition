"""Finals benchmark at feature level: 5-fold CV over dev recordings, organiser-style reference draws.

Members
  rank : LightGBM on features ranked against the recording's reference windows
  abs  : LightGBM + LR on features scaled by the training prior only (ignores the reference)
  absq : rank of the abs score among the abs scores of the reference windows
Per-recording AUC / ACC / bACC, then the mean over recordings (finals score = mean of (AUC+ACC)/2).
Test windows whose 6 s contain unusable samples get 0.5 here (the real predict uses the clean part).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.finals.sim import PATTERNS, instance, lagged_labels, load_recs, rank_vs_ref, z_vs_ref
from src.evaluation.splits import load_splits
from src.preprocessing.rank_norm import prior_view
from src.utils.exp_log import log_experiment

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--train_orders", default="real", choices=["real", "aug"])
ap.add_argument("--draws", type=int, default=6, help="reference draws per training recording")
ap.add_argument("--eval_seeds", type=int, default=3)
ap.add_argument("--eval_orders", default="real", choices=["real", "all20"])
ap.add_argument("--params", default="configs/lgb_order_best_w62.json")
ap.add_argument("--members", default="rank,abs")
ap.add_argument("--row_step", type=int, default=2, help="keep every n-th training row")
ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--no_log", action="store_true")
ap.add_argument("--abs_feat", default=None, help="feature cache for the abs member (default: same as rank)")
ap.add_argument("--rank_view", default="rank", choices=["rank", "rank+z", "z"])
ap.add_argument("--ref_step", type=int, default=1)
ap.add_argument("--lag", type=float, default=0.0, help="training labels shifted by this many seconds (15-s guard)")
a = ap.parse_args()
P = {**json.load(open(a.params)), "subsample_freq": 1}
sp = load_splits()
DEV = sum(sp["folds"], [])
R = load_recs(ids=set(DEV))
RA = load_recs(a.abs_feat, ids=set(DEV)) if a.abs_feat else R
for _r in R.values():
    _r.yl, _r.keep = lagged_labels(_r, a.lag, 15.0) if a.lag > 0 else (_r.y, _r.clean & _r.pure)


def view(F, Fref):
    if a.rank_view == "rank":
        return rank_vs_ref(F, Fref)
    if a.rank_view == "z":
        return z_vs_ref(F, Fref)
    return np.concatenate([rank_vs_ref(F, Fref), z_vs_ref(F, Fref)], axis=1)

members = a.members.split(",")
rows = []  # per (rec, seed, order): arrays
t0 = time.time()
for k in [int(f) for f in a.folds.split(",")]:
    test_ids = sp["folds"][k]
    pool = [r for r in DEV if r not in test_ids]
    Fall = np.concatenate([RA[r].F[RA[r].clean] for r in pool])
    mu, sd = Fall.mean(0), Fall.std(0) + 1e-6
    rng = np.random.default_rng(1000 + k)
    models = {}
    if "rank" in members:
        X, Y = [], []
        for rid in pool:
            rec = R[rid]
            for _ in range(a.draws):
                order = rng.permutation(6).tolist() if a.train_orders == "aug" else None
                ins = instance(rec, rng, order, ref_step=a.ref_step, train_ok=rec.keep)
                if len(ins["ref_rows"]) < 20:
                    continue
                tr = ins["train_rows"][rng.integers(a.row_step) :: a.row_step]
                X.append(view(rec.F[tr], rec.F[ins["ref_rows"]]))
                Y.append(rec.yl[tr])
        X, Y = np.concatenate(X), np.concatenate(Y)
        models["rank"] = lgb.LGBMClassifier(**P, n_jobs=a.jobs, verbose=-1, random_state=k).fit(X, Y)
        n_train = len(Y)
        del X
    if "abs" in members or "absq" in members:
        Xa = np.concatenate([prior_view(RA[r].F[R[r].keep], mu, sd) for r in pool])
        Ya = np.concatenate([R[r].yl[R[r].keep] for r in pool])
        b = lgb.LGBMClassifier(**P, n_jobs=a.jobs, verbose=-1, random_state=k).fit(Xa, Ya)
        sc = StandardScaler().fit(Xa)
        lr = LogisticRegression(C=0.01, max_iter=500).fit(sc.transform(Xa), Ya)
        models["abs"] = lambda Z, b=b, sc=sc, lr=lr: 0.5 * b.predict_proba(Z)[:, 1] + 0.5 * lr.predict_proba(sc.transform(Z))[:, 1]
        del Xa
    for rid in test_ids:
        rec = R[rid]
        for seed in range(a.eval_seeds):
            for pat in (PATTERNS if a.eval_orders == "all20" else ["real"]):
                erng = np.random.default_rng([seed, int(rid), PATTERNS.index(pat) if pat != "real" else 99])
                ins = instance(rec, erng, None if pat == "real" else rec.order_for(pat), ref_step=a.ref_step)
                t = ins["test_rows"]
                scored = (t >= 0) & rec.pure[np.maximum(t, 0)]
                t = t[scored]
                ok = rec.clean_sig[t] & (len(ins["ref_rows"]) >= 20)
                out = dict(rec=rid, seed=seed, pat=pat, y=rec.y[t].astype(np.int8), ok=ok)
                ref = ins["ref_rows"]
                if "rank" in models:
                    p = np.full(len(t), 0.5)
                    if ok.any():
                        p[ok] = models["rank"].predict_proba(view(rec.F[t[ok]], rec.F[ref]))[:, 1]
                    out["rank"] = p
                    out["rank_ref"] = models["rank"].predict_proba(view(rec.F[ref], rec.F[ref]))[:, 1] if len(ref) >= 20 else np.full(1, 0.5)
                if "abs" in models:
                    p = np.full(len(t), 0.5)
                    s = models["abs"](prior_view(RA[rid].F[t[ok]], mu, sd)) if ok.any() else np.zeros(0)
                    p[ok] = s
                    out["abs"] = p
                    out["abs_ref"] = models["abs"](prior_view(RA[rid].F[ref], mu, sd)) if len(ref) >= 20 else np.full(1, 0.5)
                    if "absq" in members:
                        q = np.full(len(t), 0.5)
                        sref = np.sort(models["abs"](prior_view(RA[rid].F[ref], mu, sd))) if len(ref) else np.zeros(0)
                        if ok.any():
                            q[ok] = np.searchsorted(sref, s) / max(len(sref), 1)
                        out["absq"] = q
                rows.append(out)
    print(f"fold {k} done {time.time() - t0:.0f}s", flush=True)


def summarize(key_fn, label):
    by = {}
    for r in rows:
        if len(np.unique(r["y"])) < 2:
            continue
        p = key_fn(r)
        by.setdefault((r["rec"], r["pat"]), []).append((roc_auc_score(r["y"], p), np.mean((p >= 0.5) == r["y"]), balanced_accuracy_score(r["y"], p >= 0.5)))
    res = {}
    for pat in sorted({k[1] for k in by}):
        m = np.array([np.mean(v, axis=0) for (rid, pt), v in by.items() if pt == pat])
        res[pat] = m.mean(0)
    M = np.array(list(res.values()))
    worst = min(res, key=lambda p: res[p][0] + res[p][1])
    line = f"{label:22s} AUC {M[:, 0].mean():.4f} ACC {M[:, 1].mean():.4f} bACC {M[:, 2].mean():.4f} final {(M[:, 0].mean() + M[:, 1].mean()) / 2:.4f}"
    if len(res) > 1:
        line += f" | worst order {worst} AUC {res[worst][0]:.4f} ACC {res[worst][1]:.4f} | rtrtrt AUC {res['rtrtrt'][0]:.4f} ACC {res['rtrtrt'][1]:.4f}"
    print(line, flush=True)
    return dict(auc=float(M[:, 0].mean()), acc=float(M[:, 1].mean()), bacc=float(M[:, 2].mean()), final=float((M[:, 0].mean() + M[:, 1].mean()) / 2), worst=worst, worst_auc=float(res[worst][0]), worst_acc=float(res[worst][1]))


os.makedirs("results/finals", exist_ok=True)
names = [m for m in ("rank", "abs", "absq") if m in rows[0]]
out = {}
for n in names:
    out[n] = summarize(lambda r, n=n: r[n], n)
if "rank" in names and "abs" in names:
    for w in (0.9, 0.8, 0.7):
        out[f"rank{w}+abs"] = summarize(lambda r, w=w: w * r["rank"] + (1 - w) * r["abs"], f"rank*{w}+abs*{1 - w:.1f}")
if "rank" in names and "absq" in names:
    for w in (0.8, 0.6):
        out[f"rank{w}+absq"] = summarize(lambda r, w=w: w * r["rank"] + (1 - w) * r["absq"], f"rank*{w}+absq*{1 - w:.1f}")
frac_ok = np.mean(np.concatenate([r["ok"] for r in rows]))
print(f"scored windows per recording {np.mean([len(r['y']) for r in rows]):.0f}, with full clean 6 s: {frac_ok:.3f}")
import pickle

pickle.dump(rows, open(f"results/finals/{a.tag}_rows.pkl", "wb"))
if not a.no_log:
    for n, r in out.items():
        log_experiment(f"fin_{a.tag}_{n}", n, f"finals_sim_cv5_dev46_{a.eval_orders}", 0, dict(train_orders=a.train_orders, draws=a.draws, eval_seeds=a.eval_seeds, params=a.params, row_step=a.row_step),
                       dict(pooled_auc=r["auc"], pooled_bacc=r["bacc"], fin_acc=r["acc"], fin_score=r["final"], fin_worst_order=r["worst"], fin_worst_auc=r["worst_auc"], fin_worst_acc=r["worst_acc"]), compliant="finals-reference")
