"""Two-branch hybrid network, ported from the user's eegcomp project (src/eegcog/models/hybrid.py).

Waveform branch: the 6-s window, linearly detrended, decimated to 125 Hz (750 points)
and divided by the window's own standard deviation, so it carries no cross-window
information. Five strided 1-D convolutions, then mean and std over time (128 dims).

Feature branch: hand-crafted features after per-recording online normalization
(the only cross-window state), through BatchNorm.

Head: 3 fully connected layers to one logit (P(task) after sigmoid).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy import signal

from src.preprocessing.features import linear_detrend
from src.preprocessing.filters import bandpass_notch, bandpass_only

RAW_LEN = 750  # 6 s at 125 Hz
FS = 250
# Waveform preprocessing variants. Every one is a pure function of the 6-s window.
WAVE_MODES = ("base", "aa", "250", "fb", "scale", "wav", "bpn", "bp")
_AA_TAPS = signal.firwin(51, 55.0, fs=FS)  # passes 50 Hz, removes 100 Hz before 2:1 decimation
_FB_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
_FB_SOS = [signal.butter(4, b, "bandpass", fs=FS, output="sos") for b in _FB_BANDS]



def _causal_fir(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Forward-only FIR along the last axis, started from the window's first sample (no cross-window state)."""
    zi = signal.lfilter_zi(taps, 1.0)[None, :] * x[:, :1]
    return signal.lfilter(taps, 1.0, x, axis=-1, zi=zi)[0]


def _causal_sos(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    zi = signal.sosfilt_zi(sos)[:, None, :] * x[None, :, :1]
    return signal.sosfilt(sos, x, axis=-1, zi=zi)[0]


def wavelet_denoise(w: np.ndarray, wavelet: str = "db4", level: int = 5) -> np.ndarray:
    """VisuShrink inside the window: soft-threshold all detail levels at sigma*sqrt(2 ln N).

    sigma is estimated from the finest detail level (median absolute deviation / 0.6745).
    Uses only the samples of the given window.
    """
    import pywt

    w = np.asarray(w, dtype=np.float64)
    coeffs = pywt.wavedec(w, wavelet, level=level, axis=-1, mode="symmetric")
    sigma = np.median(np.abs(coeffs[-1]), axis=-1, keepdims=True) / 0.6745
    thr = sigma * np.sqrt(2.0 * np.log(w.shape[-1]))
    coeffs = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet, axis=-1, mode="symmetric")[..., : w.shape[-1]]


def wave_input(w: np.ndarray, mode: str = "base") -> np.ndarray:
    """(n, 1500) uV -> network waveform input for a preprocessing variant.

    base  : detrend, average pairs to 125 Hz, divide by the window's own std          -> (n, 750)
    aa    : detrend, causal 55-Hz FIR low-pass, take every 2nd sample, window std    -> (n, 750)
    250   : detrend, keep 250 Hz (62.5-125 Hz retained), window std                   -> (n, 1500)
    fb    : anti-aliased 125-Hz signal plus 5 causal band-passed copies, shared scale -> (n, 6, 750)
    scale : like base but NOT divided by the window std (caller divides by an online scale) -> (n, 750)
    wav   : wavelet denoising (db4, 5 levels, soft universal threshold), then base    -> (n, 750)
    bpn   : forward-only 1-45 Hz band-pass + 50 Hz notch, every 2nd sample, window std -> (n, 750)
    bp    : the same band-pass without the notch (ablation)                             -> (n, 750)
    """
    d = linear_detrend(np.asarray(w, dtype=np.float64))
    if mode == "wav":
        d = linear_detrend(wavelet_denoise(d))
    if mode in ("base", "scale", "wav"):
        d = d.reshape(len(d), -1, 2).mean(axis=2)
    elif mode == "aa":
        d = _causal_fir(d, _AA_TAPS)[:, 1::2]
    elif mode == "bpn":
        d = bandpass_notch(d)[:, 1::2]
    elif mode == "bp":
        d = bandpass_only(d)[:, 1::2]
    elif mode == "fb":
        bands = [_causal_sos(d, sos) for sos in _FB_SOS]
        d = np.stack([_causal_fir(d, _AA_TAPS)] + bands, axis=1)[:, :, 1::2]
        d = d / (d[:, :1].std(axis=2, keepdims=True) + 1e-6)
        return np.clip(d, -10, 10).astype(np.float32)
    if mode != "scale":
        d = d / (d.std(axis=-1, keepdims=True) + 1e-6)
    return d.astype(np.float32) if mode == "scale" else np.clip(d, -10, 10).astype(np.float32)


class HybridNet(nn.Module):
    def __init__(self, z_dim: int, in_ch: int = 1, k1: int = 25, s1: int = 2):
        """Defaults reproduce the original network. The 250-Hz variant uses k1=49, s1=4, which
        covers the same time span per output step as the original at 125 Hz."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 16, k1, stride=s1, padding=k1 // 2), nn.BatchNorm1d(16), nn.ELU(),
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

    def wave(self, raw: torch.Tensor) -> torch.Tensor:
        h = self.conv(raw[:, None, :] if raw.dim() == 2 else raw)
        return torch.cat([h.mean(-1), h.std(-1)], dim=1)

    def head_from(self, wave_emb: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([wave_emb, self.z_norm(z)], dim=1)).squeeze(1)

    def forward(self, raw: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.head_from(self.wave(raw), z)


def raw_input(w: np.ndarray) -> np.ndarray:
    """The original waveform input (mode "base")."""
    return wave_input(w, "base")


def net_kwargs(mode: str) -> dict:
    return {"fb": dict(in_ch=6), "250": dict(k1=49, s1=4)}.get(mode, {})


def online_z(F: np.ndarray, clean: np.ndarray, mu0: np.ndarray, sd0: np.ndarray, prior_n: float = 60.0) -> np.ndarray:
    """Per-recording cumulative z-score, rows in time order; row t sees only clean rows before t.

    The training-set prior counts as prior_n pseudo-observations (as in eegcomp's OnlineNormalizer),
    so the first minutes are continuous rather than noisy.
    """
    X = np.nan_to_num(F).astype(np.float64)
    c = clean.astype(np.float64)[:, None]
    s1 = np.cumsum(X * c, axis=0) - X * c
    s2 = np.cumsum(X * X * c, axis=0) - X * X * c
    n = np.cumsum(c, axis=0) - c
    w = prior_n + n
    mean = (prior_n * mu0 + s1) / w
    ex2 = (prior_n * (sd0**2 + mu0**2) + s2) / w
    sd = np.sqrt(np.maximum(ex2 - mean**2, 1e-12))
    return np.clip((X - mean) / sd, -8, 8).astype(np.float32)
