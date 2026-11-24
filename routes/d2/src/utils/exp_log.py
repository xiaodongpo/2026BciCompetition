"""Append-only experiment log and leaderboard. Every row comes from a real run."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess

import pandas as pd

EXP = "results/experiments.csv"
BOARD = "results/leaderboard.csv"


def git_rev() -> str:
    """Commit id when available, plus a content hash of the code so uncommitted runs stay traceable."""
    h = hashlib.sha1()
    for root in ("src", "scripts", "configs"):
        for d, _, files in sorted(os.walk(root)):
            for f in sorted(files):
                if f.endswith((".py", ".json", ".sh")):
                    h.update(open(os.path.join(d, f), "rb").read())
    try:
        c = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        c = "uncommitted"
    return f"{c}+tree-{h.hexdigest()[:10]}"


def log_experiment(exp_id: str, model: str, split: str, seed, config: dict, metrics: dict, **extra) -> None:
    os.makedirs("results", exist_ok=True)
    row = dict(
        exp_id=exp_id,
        time=dt.datetime.now().isoformat(timespec="seconds"),
        git=git_rev(),
        seed=seed,
        split=split,
        model=model,
        config=json.dumps(config, sort_keys=True, default=str),
    )
    row.update(extra)
    row.update(metrics)
    new = pd.DataFrame([row])
    if os.path.exists(EXP):
        old = pd.read_csv(EXP, dtype=str)
        new = pd.concat([old, new.astype(str)], ignore_index=True)
    new.to_csv(EXP, index=False)
    update_leaderboard()


def update_leaderboard() -> None:
    d = pd.read_csv(EXP)
    cols = [c for c in ["exp_id", "model", "split", "compliant", "pooled_auc", "pooled_bacc", "rec_auc_mean", "rec_auc_std", "rec_bacc_mean", "rec_bacc_std", "fold_auc_std", "train_s", "infer_ms", "n_params", "config"] if c in d.columns]
    d = d[d["split"].astype(str).str.startswith("cv")]
    d = d.sort_values(["pooled_auc"], ascending=False)
    d[cols].to_csv(BOARD, index=False)
