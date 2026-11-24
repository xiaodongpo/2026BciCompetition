"""Window bookkeeping: decision times, validity and labels, one recording at a time."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.io import FS, Recording
from src.preprocessing.quality import make_valid_mask


@dataclass
class WindowIndex:
    rec_id: str
    ends: np.ndarray  # decision sample t; the window covers [t-win+1, t]
    y: np.ndarray  # label at the decision sample (-1 = no state label)
    clean: np.ndarray  # whole window free of bad samples
    pure: np.ndarray  # whole window inside one trial
    valid_t: np.ndarray  # decision sample itself is not bad


def clean_signal(rec: Recording, guard: int = 125) -> tuple[np.ndarray, np.ndarray]:
    return rec.x, make_valid_mask(rec.x, rec.bad, guard=guard)


def index_windows(rec: Recording, win: int, step: int, guard: int = 125) -> WindowIndex:
    valid = make_valid_mask(rec.x, rec.bad, guard=guard)
    y = rec.label_array()
    trial_id = np.full(rec.n, -1, np.int32)
    for i, t in enumerate(rec.trials):
        trial_id[t.start : t.end] = i
    ends = np.arange(win - 1, rec.n, step, dtype=np.int64)
    cum = np.concatenate([[0], np.cumsum(~valid)])
    n_bad = cum[ends + 1] - cum[ends + 1 - win]
    tid_end = trial_id[ends]
    tid_start = trial_id[ends - win + 1]
    pure = (tid_end == tid_start) & (tid_end >= 0)
    return WindowIndex(rec.rec_id, ends, y[ends].astype(np.int8), n_bad == 0, pure, valid[ends])


def extract_windows(x: np.ndarray, ends: np.ndarray, win: int) -> np.ndarray:
    idx = ends[:, None] - win + 1 + np.arange(win)
    return x[idx]
