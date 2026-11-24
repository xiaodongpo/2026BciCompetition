"""Sample-quality masks computable from the raw signal alone.

No annotations are required, so the same logic runs in streaming inference.
"""
from __future__ import annotations

import numpy as np

RAIL_UV = 400_000.0  # BDF physical range is +/-403333 uV; values at it are saturated.
ZERO_FILL_UV = 0.1  # zero-filled gaps decode to 0.024 uV because of the BDF offset.
MIN_RUN = 3


def _dilate_runs(flat: np.ndarray, min_run: int) -> np.ndarray:
    """Keep only True runs of at least min_run samples."""
    if not flat.any():
        return flat
    d = np.diff(np.concatenate([[0], flat.astype(np.int8), [0]]))
    out = np.zeros_like(flat)
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if e - s >= min_run:
            out[s:e] = True
    return out


def invalid_mask(x: np.ndarray) -> np.ndarray:
    """True where a sample is missing or not a usable EEG measurement.

    Covers NaN, zero-fill (exact gap marker) and rail saturation. Deliberately
    causal-friendly: each sample's flag depends on values inside the given array.
    """
    x = np.asarray(x, dtype=np.float64)
    nan = ~np.isfinite(x)
    v = np.where(nan, 0.0, x)
    rail = np.abs(v) >= RAIL_UV
    zero = _dilate_runs(np.abs(v) < ZERO_FILL_UV, MIN_RUN)
    return nan | rail | zero


def make_valid_mask(x: np.ndarray, annotated_bad: np.ndarray | None = None, guard: int = 125) -> np.ndarray:
    """Training-time mask: valid samples, with a guard region after each bad run.

    The guard is applied only *after* a bad run (0.5 s default) since a
    saturated amplifier needs time to recover; applying it before would be fine
    offline but could not be reproduced causally.
    """
    bad = invalid_mask(x)
    if annotated_bad is not None:
        bad = bad | annotated_bad
    if guard > 0 and bad.any():
        d = np.diff(np.concatenate([[0], bad.astype(np.int8), [0]]))
        for e in np.where(d == -1)[0]:
            bad[e : e + guard] = True
    return ~bad
