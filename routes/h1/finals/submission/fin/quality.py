"""Unusable-sample detection on the filtered signal (inside predict only; filter_eeg never calls this).

The low-pass in filter_eeg has unit DC gain, so after it settles a zero-filled gap sits near
0 uV and amplifier saturation sits at the BDF rail (+/-403333 uV). Both are found from the
samples alone; BDF annotations are not used.
"""
from __future__ import annotations

import numpy as np

RAIL_UV = 400_000.0
ZERO_UV = 0.1
MIN_RUN = 5  # samples
SETTLE = 40  # samples; the filter settles within 34


def _runs(mask: np.ndarray):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def invalid_mask(x: np.ndarray) -> np.ndarray:
    """True where a sample of the filtered signal is not a usable EEG measurement."""
    x = np.asarray(x, dtype=np.float64)
    bad = ~np.isfinite(x)
    v = np.where(bad, 0.0, x)
    bad |= np.abs(v) >= RAIL_UV
    s, e = _runs(np.abs(v) < ZERO_UV)
    for a, b in zip(s, e):
        if b - a >= MIN_RUN:
            bad[a:b] = True
    if bad.any():
        s, e = _runs(bad)
        for a, b in zip(s, e):
            bad[max(0, a - SETTLE) : b + SETTLE] = True
    return bad


def longest_clean_run(bad: np.ndarray) -> tuple[int, int]:
    """(start, end) of the longest run without bad samples; (0, 0) when there is none."""
    s, e = _runs(~bad)
    if len(s) == 0:
        return 0, 0
    k = int(np.argmax(e - s))
    return int(s[k]), int(e[k])
