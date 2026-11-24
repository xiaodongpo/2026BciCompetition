"""Causal per-recording percentile normalization of features."""
from __future__ import annotations

import bisect

import numpy as np

NORMAL_IQR = 1.349  # IQR of a standard normal, in sd units


class OnlineRank:
    """Percentile of each feature within the recording's past clean windows.

    The value for the current window is computed before the window enters the
    history. Early on, the percentile is blended with a prior derived from
    training-set mean and std, so the first minutes remain usable.

    context() summarizes the history itself (median and IQR relative to the
    training prior). It also uses only past, unlabeled windows.
    """

    def __init__(self, mu: np.ndarray, sd: np.ndarray, min_hist: int = 30):
        self.mu, self.sd, self.min_hist = np.asarray(mu, float), np.asarray(sd, float), min_hist
        self.hist: list[list[float]] = [[] for _ in range(len(self.mu))]

    def transform(self, f: np.ndarray) -> np.ndarray:
        f = np.nan_to_num(np.asarray(f, float))
        prior = 0.5 + 0.5 * np.tanh((f - self.mu) / self.sd)
        n = len(self.hist[0])
        if n == 0:
            return prior
        q = np.array([bisect.bisect_left(h, v) for h, v in zip(self.hist, f)]) / n
        w = min(n / self.min_hist, 1.0)
        return w * q + (1 - w) * prior

    def context(self) -> np.ndarray:
        n = len(self.hist[0])
        k = len(self.mu)
        if n == 0:
            return np.concatenate([np.zeros(k), np.full(k, NORMAL_IQR)])
        med = np.array([h[n // 2] for h in self.hist])
        iqr = np.array([h[(3 * n) // 4] - h[n // 4] for h in self.hist])
        w = min(n / self.min_hist, 1.0)
        return np.concatenate([w * (med - self.mu) / self.sd, w * iqr / self.sd + (1 - w) * NORMAL_IQR])

    def update(self, f: np.ndarray) -> None:
        f = np.nan_to_num(np.asarray(f, float))
        for h, v in zip(self.hist, f):
            bisect.insort(h, float(v))

    @property
    def n(self) -> int:
        return len(self.hist[0])


def prior_view(F: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Non-adaptive view: features squashed with the training prior only."""
    return (0.5 + 0.5 * np.tanh((np.nan_to_num(F) - mu) / sd)).astype(np.float32)


def rank_recording(F: np.ndarray, clean: np.ndarray, mu: np.ndarray, sd: np.ndarray, min_hist: int = 30, context: bool = False):
    """Offline replay of the streaming behavior, one recording's rows in time order."""
    rk = OnlineRank(mu, sd, min_hist)
    out = np.zeros_like(F, dtype=np.float32)
    ctx = np.zeros((len(F), 2 * F.shape[1]), dtype=np.float32) if context else None
    for t in range(len(F)):
        out[t] = rk.transform(F[t])
        if context:
            ctx[t] = rk.context()
        if clean[t]:
            rk.update(F[t])
    return (out, ctx) if context else out
