"""The causal filter used by filter_eeg: 2nd-order Butterworth low-pass at 100 Hz, forward only.

Only filtering happens here: no normalization, no feature extraction, no gap handling.
The initial state is the filter's steady state for the first sample, so the output does not
start with an artificial step from zero.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

FS = 250
SOS = butter(2, 100.0, btype="lowpass", fs=FS, output="sos")


def causal_lowpass(eeg_uv: np.ndarray) -> np.ndarray:
    x = np.asarray(eeg_uv, dtype=np.float64)
    y, _ = sosfilt(SOS, x, zi=sosfilt_zi(SOS) * x[0])
    return y
