"""Order-robustness benchmark on dev CV: every model is scored under all 20 relax/task trial orders.

Each test recording is replayed with its trials rearranged into each of the 20
patterns with three relax and three task trials (pre-experiment rows stay first).
Selection uses the mean over patterns of (AUC + bACC) / 2. The lockbox is not read.
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import lightgbm as lgb
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.io import load_all
from src.evaluation.splits import load_splits
from src.preprocessing.rank_norm import prior_view, rank_recording
from src.utils.exp_log import log_experiment

PATTERNS = sorted({"".join(p) for p in itertools.permutations("rrrttt")})
assert len(PATTERNS) == 20

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="V0,V1,V2,V3,VS")
ap.add_argument("--features", default="cache/feat2_w6_s1.npz")
ap.add_argument("--extra", default=None, help="comma-separated feature caches aligned on (rec, end), concatenated to the features")
ap.add_argument("--tag", default="ord")
ap.add_argument("--params", default="configs/lgb_best.json")
ap.add_argument("--n_aug", type=int, default=4)
ap.add_argument("--jobs", type=int, default=6)
ap.add_argument("--lgb_jobs", type=int, default=8)
a = ap.parse_args()

z = np.load(a.features)
F, y, rec, clean, pure, end = np.nan_to_num(z["F"]), z["y"], z["rec"], z["clean"], z["pure"], z["end"]
for extra_path in (a.extra.split(",") if a.extra else []):
    e = np.load(extra_path)
    key = {(r, t): i for i, (r, t) in enumerate(zip(e["rec"], e["end"]))}
    sel = np.array([key[(r, t)] for r, t in zip(rec, end)])
    F = np.concatenate([F, np.nan_to_num(e["F"][sel])], axis=1)
sp = load_splits()
DEV = sum(sp["folds"], [])
recs = {r.rec_id: r for r in load_all("EEG_dataset") if r.rec_id in DEV}
P = {**json.load(open(a.params)), "subsample_freq": 1}

# name: (view, training orders, model)
VARIANTS = {
    "V0": ("rank", 0, "lgb"),  # original order only: the rejected candidate A, as a reference
    "V1": ("rank", a.n_aug, "lgb"),  # order-augmented rank: candidate B
    "V2": ("rank+abs", a.n_aug, "lgb"),
    "V3": ("rank+abs+ctx", a.n_aug, "lgb"),
    "V3lr": ("rank+abs+ctx", a.n_aug, "lgb+lr"),
    "V2lr": ("rank+abs", a.n_aug, "lgb+lr"),
    "V3x8": ("rank+abs+ctx", 8, "lgb"),
    "VS": ("abs", 0, "lgb"),  # no adaptation; should be order-invariant
    "VSlr": ("abs", 0, "lgb+lr"),
    "V1lr": ("rank", a.n_aug, "lgb+lr"),
    "V1x8": ("rank", 8, "lgb"),
    "V1x8lr": ("rank", 8, "lgb+lr"),
    "V1x16": ("rank", 16, "lgb"),
    "V1x8w": ("rank", 8, "lgb_w"),  # every training recording gets the same total weight
}
use = [v.strip() for v in a.variants.split(",")]


def segments(rid):
    i = np.where(rec == rid)[0]
    e = end[i]
    r = recs[rid]
    pre = i[e < r.trials[0].start]
    tr = [i[(e >= t.start) & (e < t.end)] for t in r.trials]
    return pre, tr, [t.label for t in r.trials]


SEG = {r: segments(r) for r in DEV}


def order_for(rid, pattern):
    _, _, labs = SEG[rid]
    rr = [k for k, l in enumerate(labs) if l == 0]
    tt = [k for k, l in enumerate(labs) if l == 1]
    return [rr.pop(0) if c == "r" else tt.pop(0) for c in pattern]


def replay(rid, order, mu, sd, need_ctx):
    pre, tr, _ = SEG[rid]
    idx = np.concatenate([pre] + [tr[k] for k in order])
    R, C = rank_recording(F[idx], clean[idx], mu, sd, context=True) if need_ctx else (rank_recording(F[idx], clean[idx], mu, sd), None)
    return idx, R, C


def view(kind, idx, R, C, mu, sd):
    A = prior_view(F[idx], mu, sd)
    if kind == "abs":
        return A
    if kind == "rank":
        return R
    if kind == "rank+abs":
        return np.concatenate([R, A], axis=1)
    return np.concatenate([R, A, C], axis=1)


def fit(model, X, Y, seed=0, groups=None):
    sw = None
    if model.endswith("_w"):
        _, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
        sw = 1.0 / cnt[inv]
        sw = sw / sw.mean()
        model = model[:-2]
    b = lgb.LGBMClassifier(**P, n_jobs=a.lgb_jobs, verbose=-1, random_state=seed).fit(X, Y, sample_weight=sw)
    if model == "lgb":
        return lambda Z: b.predict_proba(Z)[:, 1]
    s = StandardScaler().fit(X)
    lr = LogisticRegression(C=0.01, max_iter=500).fit(s.transform(X), Y)
    return lambda Z: 0.5 * b.predict_proba(Z)[:, 1] + 0.5 * lr.predict_proba(s.transform(Z))[:, 1]


need_ctx = any("ctx" in VARIANTS[v][0] for v in use)
store = {v: {p: ([], [], []) for p in PATTERNS} for v in use}
t0 = time.time()
for k, fold in enumerate(sp["folds"]):
    pool = sum([f for j, f in enumerate(sp["folds"]) if j != k], [])
    d = np.isin(rec, pool) & clean
    mu, sd = F[d].mean(0), F[d].std(0) + 1e-6
    # Training replays, shared between variants with the same number of orders.
    rng = np.random.default_rng(100 + k)
    train_sets = {}
    for n_aug in sorted({VARIANTS[v][1] for v in use}):
        jobs = [(r, list(range(6))) for r in pool] if n_aug == 0 else [(r, rng.permutation(6).tolist()) for r in pool for _ in range(n_aug)]
        train_sets[n_aug] = Parallel(n_jobs=a.jobs)(delayed(replay)(r, o, mu, sd, need_ctx) for r, o in jobs)
    tests = Parallel(n_jobs=a.jobs)(delayed(replay)(r, order_for(r, p), mu, sd, need_ctx) for p in PATTERNS for r in fold)
    for v in use:
        kind, n_aug, model = VARIANTS[v]
        parts = train_sets[n_aug]
        X = np.concatenate([view(kind, i, R, C, mu, sd) for i, R, C in parts])
        I = np.concatenate([i for i, _, _ in parts])
        m = clean[I] & pure[I] & (y[I] >= 0)
        pred = fit(model, X[m], y[I][m], seed=k, groups=rec[I][m])
        for j, (i, R, C) in enumerate(tests):
            p = PATTERNS[j // len(fold)]
            q = clean[i] & (y[i] >= 0)
            store[v][p][0].append(y[i][q])
            store[v][p][1].append(pred(view(kind, i, R, C, mu, sd)[q]))
            store[v][p][2].append(rec[i][q])
    print(f"fold {k} done {time.time() - t0:.0f}s", flush=True)

os.makedirs("results/order20", exist_ok=True)
for v in use:
    flat = [(PATTERNS.index(p), *(np.concatenate(x) for x in store[v][p])) for p in PATTERNS]
    np.savez(f"results/order20/{a.tag}_{v}_pred.npz", pat=np.concatenate([np.full(len(f[1]), f[0]) for f in flat]), y=np.concatenate([f[1] for f in flat]), p=np.concatenate([f[2] for f in flat]), rec=np.concatenate([f[3] for f in flat]))
for v in use:
    auc, bacc, rauc = {}, {}, {}
    for p in PATTERNS:
        yy, pp, rr = (np.concatenate(x) for x in store[v][p])
        auc[p], bacc[p] = roc_auc_score(yy, pp), balanced_accuracy_score(yy, pp >= 0.5)
        rauc[p] = np.mean([roc_auc_score(yy[rr == r], pp[rr == r]) for r in np.unique(rr)])
    A, B = np.array([auc[p] for p in PATTERNS]), np.array([bacc[p] for p in PATTERNS])
    score = (A + B) / 2
    res = dict(
        order20_mean_auc=float(A.mean()), order20_mean_bacc=float(B.mean()), order20_min_auc=float(A.min()), order20_min_bacc=float(B.min()),
        order20_max_auc=float(A.max()), order20_score=float(score.mean()), orig_auc=auc["rtrtrt"], orig_bacc=bacc["rtrtrt"],
        worst_pattern=PATTERNS[int(np.argmin(score))], order20_rec_auc_mean=float(np.mean([rauc[p] for p in PATTERNS])),
        pooled_auc=float(A.mean()), pooled_bacc=float(B.mean()),
    )
    kind, n_aug, model = VARIANTS[v]
    log_experiment(f"{a.tag}_{v}", model, "order20_cv5_dev46", 0, dict(view=kind, n_aug=n_aug, model=model, features=a.features, extra=a.extra), res, compliant="yes-online-adapt" if kind != "abs" else "yes")
    json.dump(dict(auc=auc, bacc=bacc, rec_auc=rauc, config=dict(view=kind, n_aug=n_aug, model=model, features=a.features, extra=a.extra, params=a.params), **res), open(f"results/order20/{a.tag}_{v}.json", "w"), indent=1)
    print(f"{a.tag}_{v:5s} {kind:13s} aug{n_aug} {model:6s} | mean AUC {A.mean():.4f} bACC {B.mean():.4f} score {score.mean():.4f} | min AUC {A.min():.4f} ({PATTERNS[int(np.argmin(A))]}) max {A.max():.4f} | rtrtrt {auc['rtrtrt']:.4f}/{bacc['rtrtrt']:.4f}", flush=True)
