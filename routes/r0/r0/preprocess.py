# -*- coding: utf-8 -*-
# 文件作用：核心算法模块：每个4秒窗口独立执行因果1–50Hz带通、50Hz陷波、去均值、MAD缩放和限幅。供步骤5/6调用；不需要单独执行。
"""4秒已到达窗口的质量预处理。不依赖其它窗口、标签或完整记录。"""
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch, lfilter, lfilter_zi

CONFIG = dict(
    fs=250,
    seconds=4,
    bandpass_hz=[1.0, 50.0],
    bandpass_order=4,
    notch_hz=50.0,
    notch_q=30.0,
    causal_filter=True,
    filter_state='reset per window; first sample steady-state initialization',
    physical_output='filtered and mean-centered microvolt',
    model_output='per-window median/MAD scaling, clip to +/-8, then mean-center',
    mad_factor=1.4826,
    scale_floor_uv=1.0,
    clip_threshold=8.0,
    normalization_scope='current completed 4s window only',
    no_resampling=True,
    labels_used_in_preprocessing=False,
    tradeoffs=[
        'removes sub-1Hz and above-50Hz content',
        'removes absolute amplitude/baseline as model cues',
        'cannot eliminate task-order confounding',
    ],
)


def filter_only(x):
    z = np.asarray(x, dtype=np.float64)
    sos = butter(4, [1, 50], btype='bandpass', fs=250, output='sos')
    zi = sosfilt_zi(sos)[:, None, None, :] * z[None, ..., 0, None]
    z, _ = sosfilt(sos, z, zi=zi, axis=-1)
    b, a = iirnotch(50, 30, fs=250)
    z, _ = lfilter(
        b, a, z, zi=lfilter_zi(b, a)[None, None, :] * z[..., 0, None], axis=-1
    )
    return z


def transform(X_uv):
    x = np.asarray(X_uv, dtype=np.float64)
    assert x.ndim == 3 and x.shape[1:] == (1, 1000) and np.isfinite(x).all()
    z = filter_only(x)
    z -= z.mean(axis=-1, keepdims=True)
    center = np.median(z, axis=-1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(z - center), axis=-1, keepdims=True)
    scale = np.maximum(mad, CONFIG['scale_floor_uv'])
    unclip = (z - center) / scale
    fraction = np.mean(np.abs(unclip) > 8, axis=(1, 2))
    model = np.clip(unclip, -8, 8)
    model -= model.mean(axis=-1, keepdims=True)
    keep = np.isfinite(model).all(axis=(1, 2)) & (np.ptp(z, axis=-1).max(axis=1) > 1e-5)
    return (
        z.astype(np.float32),
        model[:, None, :, :].astype(np.float32),
        center.flatten(),
        scale.flatten(),
        fraction,
        keep,
    )


def self_test():
    t = np.arange(1000) / 250
    x = (
        40000
        + 20 * np.sin(2 * np.pi * 10 * t)
        + 100 * np.sin(2 * np.pi * 50 * t)
        + 500 * np.sin(2 * np.pi * 0.3 * t)
    )[None, None, :]
    physical, model, _, _, _, keep = transform(x)
    assert (
        physical.shape == (1, 1, 1000) and model.shape == (1, 1, 1, 1000) and keep.all()
    )
    assert abs(physical.mean()) < 1e-3 and abs(model.mean()) < 1e-5
    a = filter_only(x)
    b = filter_only(np.concatenate([x, np.ones((1, 1, 500)) * 1e9], axis=-1))
    assert np.array_equal(a, b[..., :1000])
    a50 = abs(np.mean(x[0, 0, 500:] * np.exp(-2j * np.pi * 50 * t[500:])))
    b50 = abs(np.mean(physical[0, 0, 500:] * np.exp(-2j * np.pi * 50 * t[500:])))
    assert b50 / a50 < 0.05, (b50, a50)
    assert np.allclose(transform(x + 100000)[1], model, atol=1e-5, rtol=0)
    print('QUALITY_1_50HZ_SYNTHETIC_CAUSAL_OFFSET_LINE_TEST_PASS', flush=True)
