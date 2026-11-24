"""Stateless per-window preparation for SingLEM: forward-only filters, then 128 Hz tokens."""
from __future__ import annotations

import numpy as np
from scipy import signal

from src.data.io import FS

_BP = signal.butter(4, [0.5, 50.0], "bandpass", fs=FS, output="sos")
_N = signal.tf2sos(*signal.iirnotch(50.0, 30.0, fs=FS))
_SOS = np.vstack([_BP, _N])


def prep_tokens(w: np.ndarray, token_len: int = 128, gain: float = 0.01, clip: float = 3.0) -> np.ndarray:
    """(n, samples) in uV -> (n, tokens, 128). Only samples inside the window are used."""
    n, T = w.shape
    w = w.astype(np.float64)
    w = w - w[:, :1]
    y = signal.sosfilt(_SOS, w, axis=-1)
    tokens = int(round(T / FS))
    m = tokens * token_len
    idx = np.clip(np.arange(m) * FS / token_len, 0, T - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    a = idx - i0
    z = y[:, i0] * (1 - a) + y[:, i1] * a
    z = np.clip(z * gain, -clip, clip)
    return np.ascontiguousarray(z.reshape(n, tokens, token_len), dtype=np.float32)
