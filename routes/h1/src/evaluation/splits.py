"""Recording-level data partitions. A recording never spans train and validation."""
from __future__ import annotations

import json
import os

import numpy as np

SPLIT_PATH = "configs/splits.json"


def _rid_batch(rid: str) -> str:
    return "A" if int(rid) <= 39 else "B"


def make_splits(rec_ids: list[str], groups: dict | None = None, n_folds: int = 5, n_holdout: int = 11, seed: int = 20261105) -> dict:
    """Lockbox plus k folds of whole groups, balanced over the two id batches.

    groups maps rec_id to a group label for recordings that must stay together
    (default: every recording is its own group).
    """
    groups = groups or {r: r for r in rec_ids}
    rng = np.random.default_rng(seed)
    by_group: dict = {}
    for r in rec_ids:
        by_group.setdefault(groups[r], []).append(r)
    glist = list(by_group)
    order = rng.permutation(len(glist))
    holdout, dev = [], []
    n_a = {"A": 0, "B": 0}
    total = {b: sum(_rid_batch(r) == b for r in rec_ids) for b in "AB"}
    target = {b: round(n_holdout * total[b] / len(rec_ids)) for b in "AB"}
    for i in order:
        g = glist[i]
        b = _rid_batch(by_group[g][0])
        if n_a[b] + len(by_group[g]) <= target[b]:
            holdout.extend(by_group[g])
            n_a[b] += len(by_group[g])
        else:
            dev.append(g)
    # Greedy fold assignment keeps the sizes and batch mix close.
    folds = [[] for _ in range(n_folds)]
    for g in sorted(dev, key=lambda g: (-len(by_group[g]), _rid_batch(by_group[g][0]))):
        b = _rid_batch(by_group[g][0])
        j = min(range(n_folds), key=lambda k: (sum(_rid_batch(r) == b for r in folds[k]), len(folds[k]), rng.random()))
        folds[j].extend(by_group[g])
    return dict(seed=seed, holdout=sorted(holdout), folds=[sorted(f) for f in folds], groups=groups)


def load_splits(path: str = SPLIT_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def save_splits(s: dict, path: str = SPLIT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(s, f, indent=1)
