"""Run the real submission (finals/submission/solution.py) the way the organisers will:
filter_eeg on the BDF, organiser-style reference draw, shuffled windows, one predict call each.
Then score per recording with the dataset's labels (a window is scored when it lies inside one trial).

Usage: FIN_MODEL_DIR is set per run; --recs lists recording ids; several seeds give several draws.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import csv

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--recs", required=True, help="comma-separated ids, or fold<k>, lockbox, dev")
ap.add_argument("--seeds", default="14")
ap.add_argument("--out", default=None)
ap.add_argument("--predictions-dir", type=Path, default=None, help="Optional PRIVATE directory for per-window labels and scores; keep under cache/")
a = ap.parse_args()
if a.predictions_dir:
    a.predictions_dir.mkdir(parents=True,exist_ok=True)
os.environ["FIN_MODEL_DIR"] = a.model
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "finals", "submission"))
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import solution as S
from src.data.io import list_recordings, pair_trials, parse_t0, read_events
from src.evaluation.splits import load_splits

sp = load_splits()
if a.recs.startswith("fold"):
    ids = sp["folds"][int(a.recs[4:])]
elif a.recs == "lockbox":
    ids = sp["holdout"]
elif a.recs == "dev":
    ids = sum(sp["folds"], [])
else:
    ids = a.recs.split(",")
paths = {parse_t0(b)[0]: (b, c) for b, c in list_recordings("EEG_dataset")}
W, width = S.WINDOW, S.REFERENCE // S.WINDOW
rows, t_filter, t_pred, n_pred = [], 0.0, 0.0, 0
for rid in ids:
    bdf, csv = paths[rid]
    t0 = time.perf_counter()
    eeg = S.filter_eeg(bdf)
    t_filter += time.perf_counter() - t0
    ev = read_events(csv, parse_t0(bdf)[1])
    trials, _ = pair_trials(ev, len(eeg))
    lab = np.full(len(eeg), -1, np.int8)
    tid = np.full(len(eeg), -1, np.int32)
    for i, t in enumerate(trials):
        lab[t.start : t.end] = t.label
        tid[t.start : t.end] = i
    for seed in [int(s) for s in a.seeds.split(",")]:
        # Same random stream as the organisers' example, keeping the mapping for scoring.
        rng = np.random.default_rng(seed)
        ref_slots, n_windows = min(eeg.size, S.HORIZON) // W, eeg.size // W
        compact = np.sort(rng.choice(ref_slots - S.REFERENCE_COUNT * (width - 1), S.REFERENCE_COUNT, replace=False))
        ref_starts = rng.permutation(compact + np.arange(S.REFERENCE_COUNT) * (width - 1)) * W
        keep = np.ones(n_windows, dtype=bool)
        for first in ref_starts // W:
            keep[int(first) : int(first) + width] = False
        order = np.flatnonzero(keep)[rng.permutation(int(keep.sum()))]
        reference, shuffled = S.segment_and_shuffle(eeg, seed)
        assert np.array_equal(shuffled, eeg[: n_windows * W].reshape(n_windows, W)[order])
        t0 = time.perf_counter()
        p = np.array([S.checked_predict(w, reference) for w in shuffled])
        t_pred += time.perf_counter() - t0
        n_pred += len(p)
        if seed == int(a.seeds.split(",")[0]):
            S.check_order_independence(shuffled, reference)
        starts = order * W
        scored = (tid[starts] >= 0) & (tid[starts] == tid[starts + W - 1])
        y = lab[starts][scored]
        q = p[scored]
        if a.predictions_dir:
            # Write all scheduled queries and mark scoring eligibility, never silently drop them.
            path=a.predictions_dir/f'{rid}_seed{seed}.csv'
            with path.open('x',encoding='utf8',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['shuffled_index','start_sample','y_endpoint','p_task','scored_pure_phase'])
                writer.writerows((j,int(starts[j]),int(lab[starts[j]+W-1]),float(p[j]),int(scored[j])) for j in range(len(p)))
        rows.append(dict(rec=rid, seed=seed, auc=roc_auc_score(y, q), acc=float(np.mean((q >= 0.5) == y)), bacc=balanced_accuracy_score(y, q >= 0.5), n=int(scored.sum()), n_half=int((q == 0.5).sum())))
        print(rid, seed, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items() if k not in ("rec", "seed")}, flush=True)
auc, acc, bacc = (np.mean([np.mean([r[k] for r in rows if r["rec"] == rid]) for rid in ids]) for k in ("auc", "acc", "bacc"))
res = dict(model=a.model, recs=a.recs, seeds=a.seeds, n_rec=len(ids), auc=float(auc), acc=float(acc), bacc=float(bacc), final=float((auc + acc) / 2), ms_per_predict=1000 * t_pred / n_pred, filter_s_per_rec=t_filter / len(ids), rows=rows)
print(f"RESULT {a.model} on {a.recs}: AUC {auc:.4f} ACC {acc:.4f} bACC {bacc:.4f} final {(auc + acc) / 2:.4f} | {res['ms_per_predict']:.1f} ms/predict, filter {res['filter_s_per_rec']:.2f} s/rec")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
