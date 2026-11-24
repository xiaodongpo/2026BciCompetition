"""Paired comparison of two 20-order prediction files (same rows): per-recording and per-fold AUC deltas."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def group_auc(y, p, keys):
    df = pd.DataFrame(dict(y=y, p=p, **keys))
    g = list(keys)
    df["r"] = df.groupby(g)["p"].rank()
    s = df.groupby(g).apply(lambda d: pd.Series(dict(n1=(d.y == 1).sum(), n0=(d.y == 0).sum(), rs=d.r[d.y == 1].sum())), include_groups=False)
    return (s.rs - s.n1 * (s.n1 + 1) / 2) / (s.n1 * s.n0)


def main(a_path, b_path, weight_strict=None):
    a, b = np.load(a_path), np.load(b_path)
    assert (a["y"] == b["y"]).all() and (a["pat"] == b["pat"]).all()
    y, pat, rec = a["y"], a["pat"], a["rec"]
    sp = json.load(open("configs/splits.json"))
    fold = np.zeros(len(rec), int)
    for k, f in enumerate(sp["folds"]):
        fold[np.isin(rec, f)] = k
    ra = group_auc(y, a["p"], dict(rec=rec, pat=pat)).groupby(level="rec").mean()
    rb = group_auc(y, b["p"], dict(rec=rec, pat=pat)).groupby(level="rec").mean()
    d = (ra - rb).values
    fa = group_auc(y, a["p"], dict(fold=fold, pat=pat)).groupby(level="fold").mean()
    fb = group_auc(y, b["p"], dict(fold=fold, pat=pat)).groupby(level="fold").mean()
    print(f"{a_path.split('/')[-1]} vs {b_path.split('/')[-1]}: per-recording AUC delta mean {d.mean():+.4f}, better on {(d > 0).sum()}/{len(d)}, wilcoxon p {wilcoxon(d).pvalue:.4f} | per-fold {np.round((fa - fb).values, 4).tolist()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
