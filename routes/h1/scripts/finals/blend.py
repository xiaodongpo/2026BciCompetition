"""Blend member predictions from the finals simulation (rows matched by recording, seed and trial order).

usage: blend.py name=path[:key] ... --w name=weight,... [--w ...]
  path: a bench rows pickle (results/finals/<tag>_rows.pkl) or a hybrid experiment dir (experiments/<name>)
"""
import glob
import pickle
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

members, combos = {}, []
args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--w":
        combos.append({kv.split("=")[0]: float(kv.split("=")[1]) for kv in args[i + 1].split(",")})
        i += 2
        continue
    name, spec = args[i].split("=")
    path, key = (spec.split(":") + [None])[:2]
    files = [path] if path.endswith(".pkl") else sorted(glob.glob(f"{path}/fold*/rows.pkl"))
    rows = sum([pickle.load(open(f, "rb")) for f in files], [])
    key = key or ("hyb" if "hyb" in rows[0] else name)
    members[name] = {(r["rec"], r["seed"], r["pat"]): (r["y"], r[key]) for r in rows}
    i += 1
keys = sorted(set.intersection(*[set(m) for m in members.values()]))
first = next(iter(members.values()))
for m in members.values():
    for k in keys:
        assert np.array_equal(m[k][0], first[k][0]), "rows do not align"
print(f"{len(keys)} instances, {len({k[0] for k in keys})} recordings, orders {len({k[2] for k in keys})}")


def score(w):
    by = {}
    for k in keys:
        y = first[k][0]
        if len(np.unique(y)) < 2:
            continue
        p = sum(wt * members[n][k][1] for n, wt in w.items()) / sum(w.values())
        by.setdefault((k[0], k[2]), []).append((roc_auc_score(y, p), np.mean((p >= 0.5) == y), balanced_accuracy_score(y, p >= 0.5)))
    res = {}
    for pat in sorted({k[1] for k in by}):
        res[pat] = np.array([np.mean(v, axis=0) for (rid, pt), v in by.items() if pt == pat]).mean(0)
    M = np.array(list(res.values()))
    worst = min(res, key=lambda p: res[p][0] + res[p][1])
    s = f"AUC {M[:, 0].mean():.4f} ACC {M[:, 1].mean():.4f} bACC {M[:, 2].mean():.4f} final {(M[:, 0].mean() + M[:, 1].mean()) / 2:.4f}"
    if len(res) > 1:
        s += f" | worst {worst} {res[worst][0]:.4f}/{res[worst][1]:.4f} | rtrtrt {res['rtrtrt'][0]:.4f}/{res['rtrtrt'][1]:.4f}"
    return s


for n in members:
    print(f"{n:28s}", score({n: 1.0}))
for w in combos:
    print(f"{','.join(f'{k}={v:g}' for k, v in w.items()):28s}", score(w))
