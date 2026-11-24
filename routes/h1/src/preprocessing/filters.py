"""Window-only filters (no PyTorch). Each call filters one window forward in time, starting from the
window's first sample, so no state is kept between windows and no later sample is used."""
from __future__ import annotations

import numpy as np
from scipy import signal

FS = 250
_BP_SOS = signal.butter(4, [1.0, 45.0], "bandpass", fs=FS, output="sos")
_NOTCH_SOS = signal.tf2sos(*signal.iirnotch(50.0, 30.0, fs=FS))
_BPN_SOS = np.vstack([_BP_SOS, _NOTCH_SOS])


def causal_sos(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Forward-only IIR along the last axis of (n, T), initial state set from each row's first sample."""
    x = np.asarray(x, dtype=np.float64)
    zi = signal.sosfilt_zi(sos)[:, None, :] * x[None, :, :1]
    return signal.sosfilt(sos, x, axis=-1, zi=zi)[0]


def bandpass_notch(w: np.ndarray) -> np.ndarray:
    """1-45 Hz 4th-order Butterworth band-pass followed by a 50 Hz notch (Q = 30)."""
    return causal_sos(w, _BPN_SOS)


def bandpass_only(w: np.ndarray) -> np.ndarray:
    """1-45 Hz band-pass without the notch (ablation: about -6 dB at 50 Hz only)."""
    return causal_sos(w, _BP_SOS)


FILTERS = {"bpn": bandpass_notch, "bp": bandpass_only}
