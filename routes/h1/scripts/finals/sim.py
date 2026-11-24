"""Feature-level simulation of the finals protocol.

A recording is a sequence of 1-s steps; row i is the 6-s window starting at second i.
The organisers' sampling is reproduced on that timeline: ten non-overlapping 30-s reference
segments on the 6-s grid inside the first 10 minutes; all other grid windows are predicted.
Trial order can be rearranged (the pre-experiment block stays first) to test order dependence.
"""
from __future__ import annotations

import itertools
import os

import numpy as np

from src.data.io import load_all

WINDOW_S, REF_S, REF_COUNT, HORIZON_S = 6, 30, 10, 600
PATTERNS = sorted({"".join(p) for p in itertools.permutations("rrrttt")})


class Rec:
    def __init__(self, rid, F, y, clean, clean_sig, pure, blocks, labels, n_sec):
        self.rid, self.F, self.y, self.clean, self.clean_sig, self.pure = rid, F, y, clean, clean_sig, pure
        self.blocks, self.labels, self.n_sec = blocks, labels, n_sec  # blocks: [pre, t0..t5, post] as (a, b) seconds

    def timeline(self, order=None) -> np.ndarray:
        """Virtual second -> row index (-1: no complete window inside the block)."""
        n_rows = len(self.F)
        ident = order is None or list(order) == list(range(len(self.blocks) - 2))
        if ident:
            v = np.arange(self.n_sec)
            return np.where(v < n_rows, v, -1)
        seq = [self.blocks[0]] + [self.blocks[1 + k] for k in order] + [self.blocks[-1]]
        out = []
        for a, b in seq:
            v = np.arange(a, b)
            v[v > b - WINDOW_S] = -1
            out.append(v)
        v = np.concatenate(out)
        return np.where(v < n_rows, v, -1)

    def order_for(self, pattern: str) -> list[int]:
        rr = [k for k, l in enumerate(self.labels) if l == 0]
        tt = [k for k, l in enumerate(self.labels) if l == 1]
        return [rr.pop(0) if c == "r" else tt.pop(0) for c in pattern]


def load_recs(feat_path="cache/fin/feat.npz", ids=None, extra=None) -> dict:
    z = np.load(feat_path)
    F = np.nan_to_num(z["F"]).astype(np.float32)
    if extra:
        F = np.concatenate([F, np.nan_to_num(np.load(extra)["F"]).astype(np.float32)], axis=1)
    out = {}
    for r in load_all("EEG_dataset"):
        if ids is not None and r.rec_id not in ids:
            continue
        m = z["rec"] == r.rec_id
        assert (np.diff(z["start"][m]) == 250).all()
        n_sec = r.n // 250
        a = [int(np.ceil(t.start / 250)) for t in r.trials]
        b_last = int(np.ceil(r.trials[-1].end / 250))
        edges = [0] + a + [b_last, n_sec]
        blocks = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        out[r.rec_id] = Rec(r.rec_id, F[m], z["y"][m], z["clean"][m], z["clean_sig"][m], z["pure"][m], blocks, [t.label for t in r.trials], n_sec)
        out[r.rec_id].lab = r.label_array()
    return out


def lagged_labels(rec: Rec, lag_s: float, guard_s: float):
    """Training labels taken lag_s before the window end; eligible when (window + guard) lies in one lagged phase."""
    end = np.arange(len(rec.F)) * 250 + 1499
    t = end - int(round(lag_s * 250))
    yl = np.where(t >= 0, rec.lab[np.maximum(t, 0)], -1).astype(np.int8)
    span = int(round(6 + guard_s))
    keep = np.zeros(len(yl), bool)
    run = 0
    for i in range(len(yl)):
        run = run + 1 if (i > 0 and yl[i] == yl[i - 1] and yl[i] >= 0) else 0
        keep[i] = yl[i] >= 0 and run >= span
    return yl, keep & rec.clean


def draw_reference(n_sec: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """The organisers' sampler on a second-based timeline: (reference start slots, keep mask over 6-s slots)."""
    ref_slots = min(n_sec, HORIZON_S) // WINDOW_S
    n_windows = n_sec // WINDOW_S
    width = REF_S // WINDOW_S
    compact = np.sort(rng.choice(ref_slots - REF_COUNT * (width - 1), REF_COUNT, replace=False))
    starts = rng.permutation(compact + np.arange(REF_COUNT) * (width - 1))
    keep = np.ones(n_windows, dtype=bool)
    for s in starts:
        keep[int(s) : int(s) + width] = False
    return starts, keep


def instance(rec: Rec, rng, order=None, ref_step=1, train_ok=None):
    """One simulated evaluation of a recording.

    Returns dict with ref_rows (usable reference windows), test_rows/test_scored (grid windows to
    predict; -1 when no window), train_rows (all non-reference clean labeled 1-s rows)."""
    v = rec.timeline(order)
    starts, keep = draw_reference(len(v), rng)
    ref = []
    for s in starts:
        t = np.arange(6 * s, 6 * s + REF_S - WINDOW_S + 1, ref_step)
        r = v[t]
        ref.append(r[(r >= 0)])
    ref = np.concatenate(ref)
    ref = ref[rec.clean_sig[ref]]
    slots = np.flatnonzero(keep)
    test = v[6 * slots]
    sec = np.arange(len(v))
    free = np.ones(len(v), bool)
    for s in starts:
        free[max(0, 6 * s - WINDOW_S + 1) : 6 * s + REF_S] = False
    tr = v[sec[free]]
    tr = tr[tr >= 0]
    tr = tr[(rec.clean[tr] & rec.pure[tr]) if train_ok is None else train_ok[tr]]
    return dict(ref_rows=ref, test_rows=test, train_rows=tr, ref_starts=starts)


def rank_vs_ref(F: np.ndarray, Fref: np.ndarray) -> np.ndarray:
    """Fraction of reference windows whose feature is below the window's, per feature."""
    S = np.sort(Fref, axis=0)
    out = np.empty(F.shape, np.float32)
    n = len(S)
    for j in range(F.shape[1]):
        out[:, j] = np.searchsorted(S[:, j], F[:, j], side="left") / n
    return out


def z_vs_ref(F: np.ndarray, Fref: np.ndarray) -> np.ndarray:
    mu, sd = Fref.mean(0), Fref.std(0) + 1e-6
    return np.clip((F - mu) / sd, -8, 8).astype(np.float32)
