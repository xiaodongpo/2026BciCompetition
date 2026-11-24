"""Metrics. relax=0, task=1; score = P(task), higher means more likely task."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return dict(
        bacc=balanced_accuracy_score(y, yhat),
        auc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        sens=tp / max(tp + fn, 1),
        spec=tn / max(tn + fp, 1),
        f1=f1_score(y, yhat, zero_division=0),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        n=len(y),
    )


def best_bacc_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximizing balanced accuracy; use on training/inner data only."""
    order = np.argsort(p)
    ps, ys = p[order], y[order]
    n1 = max(int(ys.sum()), 1)
    n0 = max(len(ys) - n1, 1)
    # threshold between i and i+1: predicted task for indices > i
    tn = np.cumsum(ys == 0)
    fn = np.cumsum(ys == 1)
    bacc = 0.5 * (tn / n0 + (n1 - fn) / n1)
    i = int(np.argmax(bacc))
    if i + 1 < len(ps):
        return float((ps[i] + ps[i + 1]) / 2)
    return float(ps[i])


def evaluate(df_y: np.ndarray, df_p: np.ndarray, df_rec: np.ndarray, thr: float = 0.5) -> dict:
    """Pooled and per-recording metrics for the out-of-fold predictions."""
    pooled = binary_metrics(df_y, df_p, thr)
    per = []
    for r in np.unique(df_rec):
        m = df_rec == r
        per.append(binary_metrics(df_y[m], df_p[m], thr))
    out = {f"pooled_{k}": v for k, v in pooled.items()}
    for k in ("bacc", "auc", "sens", "spec", "f1"):
        v = np.array([d[k] for d in per], dtype=float)
        out[f"rec_{k}_mean"] = float(np.nanmean(v))
        out[f"rec_{k}_std"] = float(np.nanstd(v))
    return out
