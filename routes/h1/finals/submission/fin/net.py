"""Two-branch network used by the "hybrid" members (inference only; CPU).

Waveform branch: the 6-s window, linearly detrended (optionally band-passed 1-45 Hz and notched at
50 Hz, forward only, inside the window), reduced to 125 Hz and divided by the window's own
standard deviation. Feature branch: the window's features z-scored against the reference EEG.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy import signal

from .features import linear_detrend

FS = 250
_BP_SOS = signal.butter(4, [1.0, 45.0], "bandpass", fs=FS, output="sos")
_NOTCH_SOS = signal.tf2sos(*signal.iirnotch(50.0, 30.0, fs=FS))
_BPN_SOS = np.vstack([_BP_SOS, _NOTCH_SOS])


def bandpass_notch(w: np.ndarray) -> np.ndarray:
    """Forward-only 1-45 Hz band-pass + 50 Hz notch along the last axis, started from each row's first sample."""
    x = np.asarray(w, dtype=np.float64)
    zi = signal.sosfilt_zi(_BPN_SOS)[:, None, :] * x[None, :, :1]
    return signal.sosfilt(_BPN_SOS, x, axis=-1, zi=zi)[0]


def wave_input(w: np.ndarray, mode: str) -> np.ndarray:
    """(n, 1500) uV -> (n, 750) network input. A pure function of each window."""
    d = linear_detrend(np.asarray(w, dtype=np.float64))
    if mode == "base":
        d = d.reshape(len(d), -1, 2).mean(axis=2)
    elif mode == "bpn":
        d = bandpass_notch(d)[:, 1::2]
    else:
        raise ValueError(mode)
    d = d / (d.std(axis=-1, keepdims=True) + 1e-6)
    return np.clip(d, -10, 10).astype(np.float32)


class HybridNet(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 25, stride=2, padding=12), nn.BatchNorm1d(16), nn.ELU(),
            nn.Conv1d(16, 32, 11, stride=2, padding=5), nn.BatchNorm1d(32), nn.ELU(), nn.Dropout(0.1),
            nn.Conv1d(32, 32, 11, stride=2, padding=5), nn.BatchNorm1d(32), nn.ELU(), nn.Dropout(0.1),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.ELU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ELU(),
        )  # fmt: skip
        self.z_norm = nn.BatchNorm1d(z_dim)
        self.head = nn.Sequential(
            nn.Linear(128 + z_dim, 128), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(128, 32), nn.ELU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )  # fmt: skip

    def forward(self, raw: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.conv(raw[:, None, :])
        emb = torch.cat([h.mean(-1), h.std(-1)], dim=1)
        return self.head(torch.cat([emb, self.z_norm(z)], dim=1)).squeeze(1)
