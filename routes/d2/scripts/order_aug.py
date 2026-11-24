"""How much of the adaptive score comes from the relax-first protocol?

Rank histories are replayed under reordered trials. Training can be augmented with
random trial orders so a model cannot learn order. Dev recordings only.
"""
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
from src.preprocessing.rank_norm import rank_recording
from src.utils.exp_log import log_experiment

z = np.load("cache/feat2_w6_s1.npz")
F0, y0, rec0, clean0, pure0, end0 = np.nan_to_num(z["F"]), z["y"], z["rec"], z["clean"], z["pure"], z["end"]
sp = load_splits()
DEV = sum(sp["folds"], [])
recs = {r.rec_id: r for r in load_all("EEG_dataset") if r.rec_id in DEV}
P = {**json.load(open("configs/lgb_best.json")), "subsample_freq": 1}
EVAL = {"rtrtrt": [0, 1, 2, 3, 4, 5], "trtrtr": [1, 0, 3, 2, 5, 4], "rrrttt": [0, 2, 4, 1, 3, 5], "tttrrr": [1, 3, 5, 0, 2, 4], "ttrrtr": [1, 3, 0, 2, 5, 4], "rttrrt": [0, 1, 3, 2, 4, 5]}


def slices(rid):
    """Row indices of each trial within the recording's rows, plus the rows before the first trial."""
    idx = np.where(rec0 == rid)[0]
    e = end0[idx]
    r = recs[rid]
    pre = idx[e < r.trials[0].start]
    return pre, [idx[(e >= t.start) & (e < t.end)] for t in r.trials]


SL = {r: slices(r) for r in DEV}


def replay(rid, order, mu, sd, keep_pre=True):
    pre, tr = SL[rid]
    ids = np.concatenate(([pre] if keep_pre else []) + [tr[k] for k in order])
    return ids, rank_recording(F0[ids], clean0[ids], mu, sd)


def build(ids_list, mu, sd, n_aug, seed):
    rng = np.random.default_rng(seed)
    jobs = []
    for rid in ids_list:
        jobs.append((rid, [0, 1, 2, 3, 4, 5]) if n_aug == 0 else None)
        if n_aug:
            jobs = [j for j in jobs if j is not None]
            for _ in range(n_aug):
                jobs.append((rid, rng.permutation(6).tolist()))
    res = Parallel(n_jobs=5)(delayed(replay)(r, o, mu, sd, True) for r, o in jobs)
    I = np.concatenate([a for a, _ in res])
    R = np.concatenate([b for _, b in res])
    m = clean0[I] & pure0[I] & (y0[I] >= 0)
    return R[m], y0[I][m]


def fit(kind, X, Y, seed=0):
    if kind == "lgb":
        return lgb.LGBMClassifier(**P, n_jobs=6, verbose=-1, random_state=seed).fit(X, Y).predict_proba
    s = StandardScaler().fit(X)
    m = LogisticRegression(C=0.01, max_iter=300).fit(s.transform(X), Y)
    return lambda A: m.predict_proba(s.transform(A))


def run(n_aug, kinds=("lgb", "lr")):
    out = {k: {o: ([], [], []) for o in EVAL} for k in kinds}
    for k, fold in enumerate(sp["folds"]):
        pool = sum([f for j, f in enumerate(sp["folds"]) if j != k], [])
        d = np.isin(rec0, pool) & clean0
        mu, sd = F0[d].mean(0), F0[d].std(0) + 1e-6
        Xtr, Ytr = build(pool, mu, sd, n_aug, seed=k)
        for kind in kinds:
            pred = fit(kind, Xtr, Ytr)
            for name, order in EVAL.items():
                keep = name == "rtrtrt"
                for rid in fold:
                    ids, R = replay(rid, order, mu, sd, keep_pre=keep)
                    q = clean0[ids] & (y0[ids] >= 0)
                    out[kind][name][0].append(y0[ids][q])
                    out[kind][name][1].append(pred(R[q])[:, 1])
                    out[kind][name][2].append(rid)
        print(f" aug={n_aug} fold {k} train rows {len(Ytr)}", flush=True)
    return out


def report(tag, out):
    summary = {}
    for kind, res in out.items():
        cells = []
        for name, (ys, ps, _) in res.items():
            yy, pp = np.concatenate(ys), np.concatenate(ps)
            ra = np.mean([roc_auc_score(a, b) for a, b in zip(ys, ps)])
            auc, bacc = roc_auc_score(yy, pp), balanced_accuracy_score(yy, pp >= 0.5)
            summary[(kind, name)] = (auc, bacc, ra)
            cells.append(f"{name} {auc:.3f}/{bacc:.3f}")
        print(f"{tag:10s} {kind:4s} " + "  ".join(cells), flush=True)
    return summary


if __name__ == "__main__":
    for n_aug in (0, 4):
        t0 = time.time()
        out = run(n_aug)
        s = report(f"aug{n_aug}", out)
        for (kind, name), (auc, bacc, ra) in s.items():
            log_experiment(f"order_{kind}_aug{n_aug}_{name}", kind, f"order_test_{name}", 0, dict(n_aug=n_aug, eval_order=name), dict(pooled_auc=auc, pooled_bacc=bacc, rec_auc_mean=ra), compliant="yes-online-adapt", train_s=round(time.time() - t0))
        print(f"aug{n_aug} done {time.time() - t0:.0f}s", flush=True)
