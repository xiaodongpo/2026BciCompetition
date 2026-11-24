"""Train and export a decoder model directory: LightGBM (+ optional LR) on online-rank features.

Usable as a script or through train_model_dir() for CV and lockbox runs. Only the
recordings in `ids` touch the model, including the feature prior (mu, sd).
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import lightgbm as lgb
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.io import load_all
from src.evaluation.splits import load_splits
from src.preprocessing.rank_norm import prior_view, rank_recording

DEFAULT_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.5, subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0)
FEATURES = "cache/feat2_w6_s1.npz"


def build_rank(F, rec, clean, mu, sd, adapt=True, n_jobs=6):
    if not adapt:
        return (0.5 + 0.5 * np.tanh((F - mu) / sd)).astype(np.float32)
    ids = np.unique(rec)
    res = Parallel(n_jobs=n_jobs)(delayed(rank_recording)(F[rec == r], clean[rec == r], mu, sd) for r in ids)
    R = np.zeros_like(F)
    for r, a in zip(ids, res):
        R[rec == r] = a
    return R


def _replay(F, clean, idx, mu, sd):
    return idx, rank_recording(F[idx], clean[idx], mu, sd)


def order_augmented(F, y, rec, clean, pure, end, ids, mu, sd, n_aug, seed, recs, n_jobs=6):
    """Replay each training recording under random trial orders."""
    rng = np.random.default_rng(seed)
    jobs = []
    for rid in ids:
        i = np.where(rec == rid)[0]
        e = end[i]
        r = recs[rid]
        pre = i[e < r.trials[0].start]
        tr = [i[(e >= t.start) & (e < t.end)] for t in r.trials]
        for _ in range(n_aug):
            order = rng.permutation(len(tr))
            jobs.append(np.concatenate([pre] + [tr[k] for k in order]))
    if os.environ.get('D2_LOW_MEMORY') == '1':
        # Same replay order and float32 values, stored on disk rather than keeping
        # all augmented arrays and another concatenated copy in RAM.
        key = hashlib.sha256((str(sorted(ids)) + str(seed) + str(n_aug) + str(F.shape)).encode()).hexdigest()[:16]
        os.makedirs('cache/augmented', exist_ok=True)
        count = sum(int((clean[j] & pure[j] & (y[j] >= 0)).sum()) for j in jobs)
        X = np.lib.format.open_memmap(f'cache/augmented/{key}_X.npy', mode='w+', dtype=np.float32, shape=(count,F.shape[1]))
        Y = np.lib.format.open_memmap(f'cache/augmented/{key}_y.npy', mode='w+', dtype=y.dtype, shape=(count,))
        pos = 0
        for j in jobs:
            _, R = _replay(F, clean, j, mu, sd)
            m = clean[j] & pure[j] & (y[j] >= 0)
            n = int(m.sum())
            X[pos:pos+n], Y[pos:pos+n] = R[m], y[j][m]
            pos += n
        X.flush(); Y.flush()
        return X, Y
    res = Parallel(n_jobs=n_jobs)(delayed(_replay)(F, clean, j, mu, sd) for j in jobs)
    I = np.concatenate([a for a, _ in res])
    R = np.concatenate([b for _, b in res])
    m = clean[I] & pure[I] & (y[I] >= 0)
    return R[m], y[I][m]


def train_model_dir(ids, out, params=None, seeds=5, use_lr=False, lr_weight=0.5, n_aug=0, adapt=True, threshold=0.5, data=None, recs=None, n_jobs=6, lgb_jobs=8):
    F, y, rec, clean, pure, end, names = data or load_features()
    d = np.isin(rec, ids) & clean
    mu, sd = F[d].mean(0), F[d].std(0) + 1e-6
    if n_aug:
        recs = recs or {r.rec_id: r for r in load_all("EEG_dataset")}
        keep = np.isin(rec, ids)
        Xtr, ytr = order_augmented(F, y, rec, clean, pure, end, ids, mu, sd, n_aug, 0, recs, n_jobs)
        del keep
    else:
        m = np.isin(rec, ids)
        R = np.zeros((int(m.sum()), F.shape[1]), np.float32)
        R[:] = build_rank(F[m], rec[m], clean[m], mu, sd, adapt=adapt, n_jobs=n_jobs)
        tr = clean[m] & pure[m] & (y[m] >= 0)
        Xtr, ytr = R[tr], y[m][tr]
    os.makedirs(out, exist_ok=True)
    p = {**DEFAULT_PARAMS, **(params or {})}
    p["subsample_freq"] = 1
    models = []
    for s in range(seeds):
        b = lgb.LGBMClassifier(**p, n_jobs=lgb_jobs, verbose=-1, random_state=1000 + s).fit(Xtr, ytr)
        fn = f"lgb_{s}.txt"
        b.booster_.save_model(os.path.join(out, fn))
        models.append(fn)
    cfg = dict(models=models, mu=mu.tolist(), sd=sd.tolist(), feature_names=names, columns=None, threshold=threshold, adapt=adapt, params=p, trained_on=sorted(ids), n_aug=n_aug, lr=None)
    if use_lr:
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=0.01, max_iter=500).fit(sc.transform(Xtr), ytr)
        cfg["lr"] = dict(mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), weight=lr_weight)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"))
    return len(ytr), len(names)


def _fit_member(Xtr, ytr, spec, out, tag, lgb_jobs):
    p = {**DEFAULT_PARAMS, **(json.load(open(spec["params"])) if isinstance(spec.get("params"), str) else spec.get("params") or {})}
    p["subsample_freq"] = 1
    models = []
    for s in range(spec.get("seeds", 5)):
        b = lgb.LGBMClassifier(**p, n_jobs=lgb_jobs, verbose=-1, random_state=1000 + s).fit(Xtr, ytr)
        fn = f"{tag}_lgb_{s}.txt"
        b.booster_.save_model(os.path.join(out, fn))
        models.append(fn)
    m = dict(view=spec["view"], weight=spec.get("weight", 1.0), models=models, columns=None, lr=None, n_aug=spec.get("n_aug", 0), params=p)
    if spec.get("lr"):
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=0.01, max_iter=500).fit(sc.transform(Xtr), ytr)
        m["lr"] = dict(mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]), weight=0.5)
    return m


def train_recipe(ids, out, recipe, data=None, recs=None, threshold=0.5, seeds=None, n_jobs=6, lgb_jobs=8):
    """Export a multi-member model directory. Only recordings in `ids` are used, including for mu and sd."""
    feats = recipe.get("features", [FEATURES])
    F, y, rec, clean, pure, end, names = data or load_features(feats)
    assert len(names) == F.shape[1]
    d = np.isin(rec, ids) & clean
    mu, sd = F[d].mean(0), F[d].std(0) + 1e-6
    os.makedirs(out, exist_ok=True)
    members = []
    for k, spec in enumerate(recipe["members"]):
        spec = {**spec, **({"seeds": seeds} if seeds else {})}
        if spec["view"] == "abs":
            m = np.isin(rec, ids) & clean & pure & (y >= 0)
            Xtr, ytr = prior_view(F[m], mu, sd), y[m]
        elif spec.get("n_aug", 0) > 0:
            recs = recs or {r.rec_id: r for r in load_all("EEG_dataset")}
            Xtr, ytr = order_augmented(F, y, rec, clean, pure, end, ids, mu, sd, spec["n_aug"], k, recs, n_jobs)
        else:
            m = np.isin(rec, ids)
            R = build_rank(F[m], rec[m], clean[m], mu, sd, n_jobs=n_jobs)
            tr = clean[m] & pure[m] & (y[m] >= 0)
            Xtr, ytr = R[tr], y[m][tr]
        cols = None
        if spec.get("columns") == "first":  # only the first feature cache (the 6-s view)
            cols = list(range(len(np.load(feats[0])["names"])))
            Xtr = Xtr[:, cols]
        mem = _fit_member(Xtr, ytr, spec, out, f"m{k}", lgb_jobs)
        mem["columns"] = cols
        members.append(mem)
        del Xtr
    windows = [WINDOW_OF[p] for p in feats]
    cfg = dict(members=members, mu=mu.tolist(), sd=sd.tolist(), feature_names=names, windows=windows, threshold=threshold, trained_on=sorted(ids), recipe=recipe)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"))
    return cfg


WINDOW_OF = {"cache/feat2_w6_s1.npz": 1500, "cache/feat2_w4_s1.npz": 1000, "cache/feat2_w3_s1.npz": 750, "cache/feat2_w2_s1.npz": 500}


def load_features(paths=None):
    """Feature caches aligned on (recording, decision sample); the first one defines the rows."""
    paths = paths or [FEATURES]
    z = np.load(paths[0])
    F, rec, end = [np.nan_to_num(z["F"])], z["rec"], z["end"]
    names = [str(n) for n in z["names"]]
    for p in paths[1:]:
        e = np.load(p)
        key = {(r, t): i for i, (r, t) in enumerate(zip(e["rec"], e["end"]))}
        sel = np.array([key[(r, t)] for r, t in zip(rec, end)])
        F.append(np.nan_to_num(e["F"][sel]))
        names += [f"w{WINDOW_OF[p] // 250}s_{n}" for n in e["names"]]
    return np.concatenate(F, axis=1).astype(np.float32), z["y"], rec, z["clean"], z["pure"], end, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", default="dev", help="dev: 46 dev recordings; all: every recording")
    ap.add_argument("--params", default="{}", help="JSON overriding LightGBM parameters, or a path to a JSON file")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--lr", action="store_true", help="also blend logistic regression")
    ap.add_argument("--order_aug", type=int, default=0)
    ap.add_argument("--no_adapt", action="store_true")
    ap.add_argument("--recipe", default=None, help="name in configs/recipes.json; overrides the single-model options")
    a = ap.parse_args()
    params = json.load(open(a.params)) if os.path.exists(a.params) else json.loads(a.params)
    sp = load_splits()
    recipe = json.load(open("configs/recipes.json"))[a.recipe] if a.recipe else None
    data = load_features(recipe.get("features") if recipe else None)
    ids = sum(sp["folds"], []) if a.which == "dev" else sorted(np.unique(data[2]).tolist())
    if a.recipe:
        cfg = train_recipe(ids, a.out, recipe, data, threshold=a.threshold, seeds=a.seeds)
        print("saved", a.out, "recipe", a.recipe, "members", [(m["view"], m["weight"], len(m["models"])) for m in cfg["members"]], "recordings", len(ids))
        return
    n, nf = train_model_dir(ids, a.out, params, a.seeds, a.lr, 0.5, a.order_aug, not a.no_adapt, a.threshold, data)
    print("saved", a.out, "train rows", n, "features", nf, "recordings", len(ids))


if __name__ == "__main__":
    main()
