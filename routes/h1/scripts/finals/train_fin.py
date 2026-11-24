"""Export a finals model directory (config.json + model files) for a recipe in configs/fin_recipes.json.

--which fold<k>: train on the dev recordings outside CV fold k (for evaluation)
--which dev / all: train on the 46 dev recordings / all 57 recordings
Hybrid members are copied from experiments/<source>/fold<k|dev|all>/seed*.pt (trained on the training host).
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.getcwd())
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.finals.sim import instance, lagged_labels, load_recs, rank_vs_ref
from src.evaluation.splits import load_splits
from src.preprocessing.rank_norm import prior_view

ap = argparse.ArgumentParser()
ap.add_argument("--recipe", required=True)
ap.add_argument("--which", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--jobs", type=int, default=8)
a = ap.parse_args()
recipe = json.load(open("configs/fin_recipes.json"))[a.recipe]
sp = load_splits()
dev = sum(sp["folds"], [])
if a.which.startswith("fold"):
    k = int(a.which[4:])
    ids, tag = [r for r in dev if r not in sp["folds"][k]], k
else:
    ids, tag = (dev if a.which == "dev" else dev + sp["holdout"]), a.which
R = load_recs(ids=set(ids))
RB = load_recs("cache/fin/featbpn.npz", ids=set(ids)) if any(m.get("bpn") for m in recipe["members"]) else None
os.makedirs(a.out, exist_ok=True)
Fall = np.concatenate([R[r].F[R[r].clean] for r in ids])
mu, sd = Fall.mean(0), Fall.std(0) + 1e-6
cfg = dict(windows=[1500, 500], mu=mu.tolist(), sd=sd.tolist(), min_ref=20, min_len=750, fallback=0.5, members=[], trained_on=sorted(ids), recipe=a.recipe)
for i, m in enumerate(recipe["members"]):
    mem = dict(view=m["view"], weight=m["weight"])
    if m["view"] in ("rank", "abs"):
        P = {**json.load(open(m["params"])), "subsample_freq": 1}
        for rec in R.values():  # training labels, optionally shifted (15-s guard), as for the networks
            rec.yl, rec.keep = lagged_labels(rec, m["lag"], 15.0) if m.get("lag", 0) > 0 else (rec.y, rec.clean & rec.pure)
        if m["view"] == "rank":
            rng = np.random.default_rng(4242 + i)
            X, Y = [], []
            for rid in ids:
                rec = R[rid]
                for _ in range(m["draws"]):
                    ins = instance(rec, rng, rng.permutation(6).tolist() if m["train_orders"] == "aug" else None, train_ok=rec.keep)
                    if len(ins["ref_rows"]) < 20:
                        continue
                    tr = ins["train_rows"][rng.integers(m.get("row_step", 2)) :: m.get("row_step", 2)]
                    X.append(rank_vs_ref(rec.F[tr], rec.F[ins["ref_rows"]])); Y.append(rec.yl[tr])
            X, Y = np.concatenate(X), np.concatenate(Y)
        else:
            src = RB if m.get("bpn") else R
            Fa = np.concatenate([src[r].F[R[r].clean] for r in ids])
            amu, asd = Fa.mean(0), Fa.std(0) + 1e-6
            X = np.concatenate([prior_view(src[r].F[R[r].keep], amu, asd) for r in ids])
            Y = np.concatenate([R[r].yl[R[r].keep] for r in ids])
            mem.update(mu=amu.tolist(), sd=asd.tolist(), bpn=bool(m.get("bpn")))
        mem["models"] = []
        for s in range(m["seeds"]):
            b = lgb.LGBMClassifier(**P, n_jobs=a.jobs, verbose=-1, random_state=1000 + s).fit(X, Y)
            fn = f"m{i}_lgb_{s}.txt"
            b.booster_.save_model(os.path.join(a.out, fn))
            mem["models"].append(fn)
        mem["lr"] = None
        if m.get("lr"):
            sc = StandardScaler().fit(X)
            lr = LogisticRegression(C=0.01, max_iter=500).fit(sc.transform(X), Y)
            mem["lr"] = dict(mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]))
        print(f"member {i} {m['view']}: {len(Y)} rows", flush=True)
        del X
    else:
        srcdir = os.path.join("experiments", m["source"], f"fold{tag}")
        mem.update(wave=m["wave"], nets=[])
        for s in range(m.get("n_nets", 3)):
            fn = f"m{i}_seed{s}.pt"
            shutil.copy(os.path.join(srcdir, f"seed{s}.pt"), os.path.join(a.out, fn))
            mem["nets"].append(fn)
    cfg["members"].append(mem)
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"))
print("saved", a.out, [(m["view"], m["weight"]) for m in cfg["members"]], "recordings", len(ids))
