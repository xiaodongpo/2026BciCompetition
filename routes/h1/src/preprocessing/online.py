"""Causal per-recording running statistics (the official rule's "online adaptation").

Values at time t depend only on samples at or before t. They use only unlabeled
signal, and they have memory longer than 6 s: anything using them must be
reported as using online adaptation rather than a pure 6-s function.
"""
from __future__ import annotations

import bisect

import numpy as np

from src.data.io import FS

DEFAULT_SCALE = 15.0  # dev-set median of per-second MAD, used until history is available


class RunningScale:
    """Expanding median of the per-second MAD of the detrended signal."""

    def __init__(self, prior: float = DEFAULT_SCALE, min_hist: int = 30):
        self.prior, self.min_hist = prior, min_hist
        self.sorted: list[float] = []

    def update(self, sec: np.ndarray) -> None:
        d = sec - np.median(sec)
        bisect.insort(self.sorted, float(np.median(np.abs(d))))

    @property
    def value(self) -> float:
        n = len(self.sorted)
        if n == 0:
            return self.prior
        med = self.sorted[n // 2]
        w = min(1.0, n / self.min_hist)
        return float(np.exp(w * np.log(max(med, 1e-3)) + (1 - w) * np.log(self.prior)))


def scale_trace(x: np.ndarray, valid: np.ndarray, prior: float = DEFAULT_SCALE) -> np.ndarray:
    """Per-sample running scale; scale[t] uses whole seconds fully completed by t."""
    n = len(x)
    out = np.empty(n, dtype=np.float32)
    rs = RunningScale(prior)
    for s in range(0, n, FS):
        e = min(n, s + FS)
        out[s:e] = rs.value
        if e - s == FS and valid[s:e].all():
            rs.update(x[s:e])
    return out
