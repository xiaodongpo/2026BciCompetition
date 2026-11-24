"""Stateless spectral features of one EEG window.

Everything here is a pure function of the samples inside the window, so the
receptive field equals the window length and is trivially causal.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import windows as sw

FS = 250

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta1": (13, 20),
    "beta2": (20, 30),
    "gamma1": (30, 45),
    "gamma2": (55, 95),
}


def linear_detrend(w: np.ndarray) -> np.ndarray:
    """Remove mean and slope along the last axis (uses only the window itself)."""
    n = w.shape[-1]
    t = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    tn = t / np.sqrt(np.sum(t**2))
    w = w - w.mean(axis=-1, keepdims=True)
    return w - (w @ tn)[..., None] * tn


def subwindow_psd(w: np.ndarray, nper: int = 2 * FS, nstep: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """Per-subwindow PSD: (n_windows, n_sub, n_freq). Detrends each subwindow."""
    n = w.shape[-1]
    starts = np.arange(0, n - nper + 1, nstep)
    sub = np.stack([w[..., s : s + nper] for s in starts], axis=-2)
    sub = linear_detrend(sub)
    h = sw.hann(nper, sym=False)
    scale = 1.0 / (FS * np.sum(h**2))
    S = np.fft.rfft(sub * h, axis=-1)
    P = 2.0 * scale * (S.real**2 + S.imag**2)
    return np.fft.rfftfreq(nper, 1 / FS), P


class FeatureConfig:
    def __init__(self, fmin=1.0, fmax=45.0, fstep=1.0, include_time=True):
        self.fmin, self.fmax, self.fstep, self.include_time = fmin, fmax, fstep, include_time


def feature_names(cfg: FeatureConfig | None = None) -> list[str]:
    cfg = cfg or FeatureConfig()
    edges = np.arange(cfg.fmin, cfg.fmax, cfg.fstep)
    names = [f"logpsd_{e:g}" for e in edges]
    names += [f"logband_{b}" for b in BANDS] + [f"rel_{b}" for b in BANDS]
    names += [f"mad_{b}" for b in BANDS]
    names += ["log_total", "theta_alpha", "theta_beta", "alpha_beta", "engage", "slope", "intercept", "sp_entropy", "sef90", "sef50", "line50"]
    if cfg.include_time:
        names += ["log_ll", "log_rms", "log_mad", "hj_mob", "hj_comp", "kurt", "skew", "zcr", "crest", "frac_large"]
    return names


def extract(w: np.ndarray, cfg: FeatureConfig | None = None) -> np.ndarray:
    """w: (n_windows, n_samples) raw microvolts -> (n_windows, n_features) float32."""
    cfg = cfg or FeatureConfig()
    w = linear_detrend(np.asarray(w, dtype=np.float64))
    f, P = subwindow_psd(w)
    eps = 1e-6
    # Median across subwindows resists short artifacts; mean keeps burst power.
    Pm = np.median(P, axis=1) / np.log(2.0)
    logP = np.log10(Pm + eps)
    feats = []
    for e in np.arange(cfg.fmin, cfg.fmax, cfg.fstep):
        m = (f >= e) & (f < e + cfg.fstep)
        feats.append(np.log10(Pm[:, m].mean(1) + eps))
    tot_m = (f >= 1) & (f < 45)
    tot = Pm[:, tot_m].sum(1) + eps
    bp = {b: Pm[:, (f >= l) & (f < h)].mean(1) + eps for b, (l, h) in BANDS.items()}
    bs = {b: Pm[:, (f >= l) & (f < h)].sum(1) + eps for b, (l, h) in BANDS.items()}
    for b in BANDS:
        feats.append(np.log10(bp[b]))
    for b in BANDS:
        feats.append(bs[b] / tot)
    # Dispersion of band power across subwindows (non-stationarity).
    logPsub = np.log10(P + eps)
    for b, (l, h) in BANDS.items():
        m = (f >= l) & (f < h)
        feats.append(logPsub[:, :, m].mean(2).std(1))
    feats.append(np.log10(tot))
    feats.append(np.log10(bp["theta"] / bp["alpha"]))
    feats.append(np.log10(bp["theta"] / (bp["beta1"] + bp["beta2"])))
    feats.append(np.log10(bp["alpha"] / (bp["beta1"] + bp["beta2"])))
    feats.append(np.log10((bp["beta1"] + bp["beta2"]) / (bp["alpha"] + bp["theta"])))
    # 1/f fit over 2-45 Hz excluding the alpha peak.
    fm = ((f >= 2) & (f < 7)) | ((f >= 14) & (f < 45))
    lf = np.log10(f[fm])
    A = np.stack([lf, np.ones_like(lf)], axis=1)
    coef = np.linalg.lstsq(A, logP[:, fm].T, rcond=None)[0]
    feats.append(coef[0])
    feats.append(coef[1])
    p = Pm[:, tot_m] / tot[:, None]
    feats.append(-(p * np.log(p + 1e-12)).sum(1) / np.log(p.shape[1]))
    cum = np.cumsum(p, axis=1)
    ftm = f[tot_m]
    feats.append(ftm[np.argmax(cum >= 0.9, axis=1)])
    feats.append(ftm[np.argmax(cum >= 0.5, axis=1)])
    feats.append(np.log10(Pm[:, (f >= 49) & (f <= 51)].mean(1) + eps))
    if cfg.include_time:
        d1 = np.diff(w, axis=1)
        d2 = np.diff(d1, axis=1)
        v0, v1, v2 = w.var(1) + eps, d1.var(1) + eps, d2.var(1) + eps
        mob = np.sqrt(v1 / v0)
        feats.append(np.log10(np.abs(d1).mean(1) + eps))
        feats.append(0.5 * np.log10(v0))
        feats.append(np.log10(np.median(np.abs(w), axis=1) + eps))
        feats.append(mob)
        feats.append(np.sqrt(v2 / v1) / mob)
        z = (w - w.mean(1, keepdims=True)) / np.sqrt(v0)[:, None]
        feats.append(np.clip((z**4).mean(1), 0, 50))
        feats.append(np.clip((z**3).mean(1), -10, 10))
        feats.append((np.signbit(w[:, 1:]) != np.signbit(w[:, :-1])).mean(1))
        feats.append(np.clip(np.abs(z).max(1), 0, 50))
        mad = np.median(np.abs(w), axis=1, keepdims=True) + eps
        feats.append((np.abs(w) > 5 * mad).mean(1))
    return np.stack(feats, axis=1).astype(np.float32)


# ---- Extended set: low-frequency resolution, transients, within-window dynamics ----
V2_EXTRA = (
    [f"lf_{a}_{b}" for a, b in [(0.17, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 4)]]
    + [f"trans_{k}" for k in ("n30", "n60", "n120", "rate2mad", "rate4mad", "max", "slope_max", "slope_p95")]
    + [f"{b}_{q}" for b in ("delta", "theta", "alpha", "beta", "gamma") for q in ("p10", "p50", "p90", "range")]
    + ["recent_delta", "recent_theta", "recent_alpha", "recent_beta", "recent_gamma", "perm_ent", "emg_ratio"]
)


def feature_names_v2() -> list[str]:
    return feature_names() + V2_EXTRA


def _ma(w: np.ndarray, k: int) -> np.ndarray:
    """Moving average over the k most recent samples, valid part only (stateless)."""
    c = np.cumsum(np.concatenate([np.zeros((w.shape[0], 1)), w], axis=1), axis=1)
    return (c[:, k:] - c[:, :-k]) / k


def extract_extra(w: np.ndarray) -> np.ndarray:
    w = linear_detrend(np.asarray(w, dtype=np.float64))
    n = w.shape[1]
    eps = 1e-6
    feats = []
    # Long periodogram: finer than the 2-s Welch used elsewhere.
    h = sw.hann(n, sym=False)
    S = np.fft.rfft(w * h, axis=1)
    f = np.fft.rfftfreq(n, 1 / FS)
    P = 2.0 * (S.real**2 + S.imag**2) / (FS * np.sum(h**2))
    for a, b in [(0.17, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 4)]:
        m = (f >= a) & (f < b)
        feats.append(np.log10(P[:, m].mean(1) + eps) if m.any() else np.zeros(len(w)))
    # Transients in a 1-8 Hz-ish band: two moving averages approximate a band-pass.
    slow = _ma(w, 8)[:, 56:] - _ma(w, 64)
    mad = np.median(np.abs(slow), axis=1, keepdims=True) + eps
    a = np.abs(slow)
    for thr in (30.0, 60.0, 120.0):
        up = (a[:, 1:] > thr) & (a[:, :-1] <= thr)
        feats.append(up.sum(1).astype(np.float64))
    for k in (2.0, 4.0):
        feats.append((a > k * mad).mean(1))
    feats.append(np.log10(a.max(1) + eps))
    d = np.abs(np.diff(_ma(w, 5), axis=1))
    feats.append(np.log10(d.max(1) + eps))
    feats.append(np.log10(np.percentile(d, 95, axis=1) + eps))
    # Band power dynamics across 0.5-s subwindows.
    nper, nstep = FS // 2, FS // 4
    f2, Q = subwindow_psd(w, nper, nstep)
    logQ = {}
    for b, (lo, hi) in {"delta": (2, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}.items():
        m = (f2 >= lo) & (f2 < hi)
        v = np.log10(Q[:, :, m].mean(2) + eps)
        logQ[b] = v
        p = np.percentile(v, [10, 50, 90], axis=1)
        feats += [p[0], p[1], p[2], p[2] - p[0]]
    # Most recent second versus the whole window (recency emphasis).
    for b in logQ:
        feats.append(logQ[b][:, -3:].mean(1) - logQ[b].mean(1))
    # Permutation entropy, order 3, lag 4 samples.
    x0, x1, x2 = w[:, :-8], w[:, 4:-4], w[:, 8:]
    code = (x0 > x1).astype(int) * 1 + (x1 > x2).astype(int) * 2 + (x0 > x2).astype(int) * 4
    pe = np.zeros(len(w))
    for c in range(8):
        pc = (code == c).mean(1)
        pe -= pc * np.log(pc + 1e-12)
    feats.append(pe / np.log(6))
    # Muscle-to-EEG ratio.
    m1 = (f2 >= 30) & (f2 < 45)
    m2 = (f2 >= 8) & (f2 < 20)
    feats.append(np.log10((Q[:, :, m1].mean(2).mean(1) + eps) / (Q[:, :, m2].mean(2).mean(1) + eps)))
    return np.stack(feats, axis=1).astype(np.float32)


def extract_v2(w: np.ndarray) -> np.ndarray:
    return np.concatenate([extract(w), extract_extra(w)], axis=1)


def extract_multi(w: np.ndarray, windows=(1500,)) -> np.ndarray:
    """Concatenate extract_v2 over the most recent `n` samples for each n in `windows`.

    All views end at the same (newest) sample and lie inside the given window, so
    the receptive field is that of the longest view. Shorter inputs (warm-up or a
    clean suffix after a gap) use whatever is available for the longer views.
    """
    w = np.asarray(w)
    return np.concatenate([extract_v2(w[:, -min(n, w.shape[1]) :]) for n in windows], axis=1)


# ---- Set v3: information that the detrended v2 features discard or average out ----
V3_NAMES = (
    ["drift_slope", "drift_abs_slope", "drift_range", "drift_curv", "slow_move", "slow_move_p90"]
    + [f"line50_{q}" for q in ("p10", "p50", "p90", "range")]
    + ["line100", "line100_ratio", "line50_prom"]
    + [f"hf_{q}" for q in ("mean", "p10", "p90", "range")]
    + ["iaf", "alpha_peak", "alpha_peak_ratio", "flatness", "flat_lo", "flat_hi"]
)


def feature_names_v3() -> list[str]:
    return list(V3_NAMES)


def extract_v3(w: np.ndarray) -> np.ndarray:
    """Pure function of the window, like extract_v2."""
    raw = np.asarray(w, dtype=np.float64)
    n = raw.shape[1]
    eps = 1e-6
    t = (np.arange(n) - (n - 1) / 2) / FS
    raw0 = raw - raw.mean(1, keepdims=True)
    # Quadratic fit of the raw window: drift level, slope and curvature (uV, uV/s, uV/s^2).
    V = np.stack([np.ones(n), t, t**2], 1)
    coef = np.linalg.lstsq(V, raw0.T, rcond=None)[0]
    fit = (V @ coef).T
    feats = [coef[1], np.log10(np.abs(coef[1]) + eps), np.log10(np.ptp(fit, axis=1) + eps), coef[2]]
    # Slow movement: 0.5-s moving average of the detrended signal, then its step size.
    w2 = linear_detrend(raw)
    slow = _ma(w2, FS // 2)
    d = np.abs(np.diff(slow[:, :: FS // 10], axis=1))
    feats += [np.log10(d.mean(1) + eps), np.log10(np.percentile(d, 90, axis=1) + eps)]
    # Line noise over 0.5-s subwindows; 100 Hz harmonic; 50 Hz prominence over neighbours.
    f2, Q = subwindow_psd(w2, FS // 2, FS // 4)
    l50 = np.log10(Q[:, :, (f2 >= 48) & (f2 <= 52)].mean(2) + eps)
    p = np.percentile(l50, [10, 50, 90], axis=1)
    feats += [p[0], p[1], p[2], p[2] - p[0]]
    f, P = subwindow_psd(w2)
    Pm = np.median(P, axis=1)
    b = lambda lo, hi: Pm[:, (f >= lo) & (f < hi)].mean(1) + eps
    feats += [np.log10(b(98, 102)), np.log10(b(98, 102) / b(48, 52)), np.log10(b(49, 51) / (0.5 * (b(44, 47) + b(53, 56))))]
    # High-frequency muscle band 100-124 Hz (above the 100 Hz harmonic).
    hf = np.log10(Q[:, :, (f2 >= 104) & (f2 < 124)].mean(2) + eps)
    q = np.percentile(hf, [10, 90], axis=1)
    feats += [hf.mean(1), q[0], q[1], q[1] - q[0]]
    # Alpha peak from a 4-s Welch (0.25 Hz bins), relative to a 1/f line through 3-7 and 14-20 Hz.
    f4, P4 = subwindow_psd(w2, 4 * FS, FS // 2)
    L4 = np.log10(np.median(P4, axis=1) + eps)
    am = (f4 >= 7) & (f4 <= 14)
    fm = ((f4 >= 3) & (f4 < 7)) | ((f4 > 14) & (f4 <= 20))
    lf = np.log10(f4[fm])
    A = np.stack([lf, np.ones_like(lf)], 1)
    c = np.linalg.lstsq(A, L4[:, fm].T, rcond=None)[0]
    resid = L4[:, am] - (np.log10(f4[am])[:, None] * c[0] + c[1]).T
    k = np.argmax(resid, axis=1)
    feats += [f4[am][k], resid[np.arange(len(k)), k], resid.mean(1)]
    # Spectral flatness (Wiener entropy) overall, low and high.
    def flat(lo, hi):
        x = Pm[:, (f >= lo) & (f < hi)] + eps
        return np.exp(np.log(x).mean(1)) / x.mean(1)
    feats += [flat(1, 45), flat(1, 13), flat(13, 45)]
    return np.stack(feats, axis=1).astype(np.float32)
