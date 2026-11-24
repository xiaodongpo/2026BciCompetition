# 来源：用户result1.zip手工特征；本版关闭高频gamma、工频和慢漂移特征。
"""单窗口手工特征。这是训练和推理共用的唯一路径，且是窗口的纯函数。

约束：
- 不使用双向滤波，只有线性去趋势和分段 FFT，且只用窗口内的样本。
- 绝对电平（直流偏置）不进入特征，其数值在 ±10^5 µV 量级，只反映电极状态。
- 补零段长度、有效分段数不进入特征。部分记录 relax 的缺失比例明显高于 task，
  这些量会泄漏标签，但换一批数据就失效。
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from types import SimpleNamespace as Config
FS = 250
def sample_validity(window, cfg):
    # 质量检查在原始微伏6秒窗完成，不能对去均值信号使用近零缺失规则。
    return np.ones(len(window), dtype=bool)

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta_lo": (13.0, 20.0),
    "beta_hi": (20.0, 30.0),
    "gamma_lo": (30.0, 45.0),
}
_EPS = 1e-12


def _causal_ma(segs: np.ndarray, width: int) -> np.ndarray:
    """逐行的因果移动平均，只用当前及之前的样本。前 width-1 个样本窗口不满。"""
    c = np.cumsum(segs, axis=1)
    out = c.copy()
    out[:, width:] = c[:, width:] - c[:, :-width]
    out[:, width:] /= width
    out[:, :width] /= np.arange(1, width + 1)
    return out


def _shape_stats(x: np.ndarray) -> np.ndarray:
    """跨分段中位数的 log RMS、log p99、峰均比，以及分段最大 log RMS。"""
    rms = np.sqrt((x**2).mean(axis=1)) + _EPS
    p99 = np.percentile(np.abs(x), 99, axis=1) + _EPS
    return np.array(
        [
            np.median(np.log10(rms)),
            np.median(np.log10(p99)),
            np.median(p99 / rms),
            np.max(np.log10(rms)),
        ]
    )


class HandcraftedFeaturizer:
    def __init__(self, cfg: Config):
        self.quality = cfg.quality
        self.groups = list(cfg.features.groups)
        self.min_segments = cfg.features.min_segments
        self.nperseg = int(round(cfg.features.segment_sec * FS))
        self.hop = int(round(cfg.features.segment_hop_sec * FS))
        self.max_segments = (cfg.stream.window_samples - self.nperseg) // self.hop + 1

        self.freqs = np.fft.rfftfreq(self.nperseg, 1 / FS)
        self.hann = np.hanning(self.nperseg)
        self.psd_scale = 2.0 / (FS * np.sum(self.hann**2))
        self.t = np.arange(self.nperseg) - (self.nperseg - 1) / 2
        self.t_norm = float(self.t @ self.t)

        f = self.freqs
        self.band_masks = np.stack([(f >= lo) & (f < hi) for lo, hi in BANDS.values()])
        self.total_mask = (f >= 1) & (f < 45)
        self.shape_mask = (f >= 2) & (f < 45)
        self.line_mask = (f >= 49) & (f <= 51)
        self.line_side_mask = ((f >= 45) & (f < 48)) | ((f > 52) & (f <= 55))
        self.alpha_mask = (f >= 8) & (f < 13)
        self.log_f = np.log10(f[self.shape_mask])
        self.names = self._names()

    def _names(self) -> list[str]:
        table = {
            "band_abs": [f"abs_{b}" for b in BANDS],
            "band_rel": [f"rel_{b}" for b in BANDS],
            "ratios": ["theta_alpha", "theta_beta", "alpha_beta", "engagement"],
            "shape": ["spec_entropy", "sef95", "aperiodic_slope", "aperiodic_offset", "alpha_peak"],
            "nonstat": [f"std_{b}" for b in BANDS],
            "time": ["log_var", "hjorth_mobility", "hjorth_complexity", "log_line_length", "kurtosis", "log_ptp"],
            "line_noise": ["line_abs", "line_prominence"],
            "band_tail": [f"mean_{b}" for b in BANDS] + [f"max_{b}" for b in BANDS],
            "transient": [
                "lf_log_rms",
                "lf_log_p99",
                "lf_crest",
                "lf_skew",
                "lf_event_rate",
                "lf_log_rms_max",
                "hf_log_rms",
                "hf_log_p99",
                "hf_crest",
                "hf_log_rms_max",
            ],
            "slow": ["slow_log_std", "slow_log_range", "slow_log_slope"],
        }
        unknown = set(self.groups) - set(table)
        if unknown:
            raise KeyError(f"未知特征组: {sorted(unknown)}")
        return [n for g in self.groups for n in table[g]]

    def _segments(self, window: np.ndarray) -> np.ndarray | None:
        """取出不含坏样本的分段，从判断时刻向前对齐，所以结果不依赖窗口起点。"""
        n = len(window)
        if n < self.nperseg:
            return None
        valid = sample_validity(window, self.quality)
        count = (n - self.nperseg) // self.hop + 1
        starts = n - self.nperseg - self.hop * np.arange(count)[::-1]
        views = sliding_window_view(window, self.nperseg)[starts]
        if valid.all():
            return views
        ok = sliding_window_view(valid, self.nperseg)[starts].all(axis=1)
        return views[ok] if ok.any() else None

    def n_valid_segments(self, window: np.ndarray) -> int:
        segs = self._segments(window)
        return 0 if segs is None else len(segs)

    def __call__(self, window: np.ndarray) -> np.ndarray | None:
        segs = self._segments(window)
        if segs is None or len(segs) < self.min_segments:
            return None

        # 去趋势前的分段均值和斜率保留下来，供慢波特征使用（只用差值和斜率，不用绝对电平）
        self._seg_means = segs.mean(axis=1)
        self._seg_slopes = (segs @ self.t) / self.t_norm
        # 线性去趋势（同时去掉直流）
        segs = segs - segs.mean(axis=1, keepdims=True)
        slope = (segs @ self.t) / self.t_norm
        segs = segs - slope[:, None] * self.t

        psd = np.abs(np.fft.rfft(segs * self.hann, axis=1)) ** 2 * self.psd_scale
        band_seg = np.log10(psd @ self.band_masks.T.astype(float) / self.band_masks.sum(axis=1) + _EPS)
        band = np.median(band_seg, axis=0)  # 中位数跨分段，抵抗瞬时伪迹
        psd_med = np.median(psd, axis=0)

        out: list[np.ndarray] = []
        for group in self.groups:
            out.append(getattr(self, f"_f_{group}")(segs, psd_med, band, band_seg))
        return np.concatenate(out)

    # ---- 各特征组 ----
    def _f_band_abs(self, segs, psd, band, band_seg):
        return band

    def _f_band_rel(self, segs, psd, band, band_seg):
        return band - np.log10(psd[self.total_mask].mean() + _EPS)

    def _f_ratios(self, segs, psd, band, band_seg):
        delta, theta, alpha, b_lo, b_hi, *_ = band
        beta = np.log10((10**b_lo + 10**b_hi) / 2 + _EPS)
        engagement = beta - np.log10((10**theta + 10**alpha) / 2 + _EPS)
        return np.array([theta - alpha, theta - beta, alpha - beta, engagement])

    def _f_shape(self, segs, psd, band, band_seg):
        p = psd[self.shape_mask] + _EPS
        prob = p / p.sum()
        entropy = -(prob * np.log(prob)).sum() / np.log(len(prob))
        sef95 = self.freqs[self.shape_mask][np.searchsorted(np.cumsum(prob), 0.95)]
        log_p = np.log10(p)
        slope, offset = np.polyfit(self.log_f, log_p, 1)
        fit_alpha = offset + slope * np.log10(self.freqs[self.alpha_mask])
        alpha_peak = np.max(np.log10(psd[self.alpha_mask] + _EPS) - fit_alpha)
        return np.array([entropy, sef95, slope, offset, alpha_peak])

    def _f_nonstat(self, segs, psd, band, band_seg):
        return band_seg.std(axis=0)

    def _f_time(self, segs, psd, band, band_seg):
        var0 = segs.var(axis=1) + _EPS
        d1 = np.diff(segs, axis=1)
        var1 = d1.var(axis=1) + _EPS
        var2 = np.diff(d1, axis=1).var(axis=1) + _EPS
        mobility = np.sqrt(var1 / var0)
        complexity = np.sqrt(var2 / var1) / mobility
        line_length = np.abs(d1).mean(axis=1) + _EPS
        z = (segs - segs.mean(axis=1, keepdims=True)) / np.sqrt(var0)[:, None]
        kurt = (z**4).mean(axis=1) - 3
        ptp = np.ptp(segs, axis=1) + _EPS
        return np.array(
            [
                np.median(np.log10(var0)),
                np.median(mobility),
                np.median(complexity),
                np.median(np.log10(line_length)),
                np.median(kurt),
                np.median(np.log10(ptp)),
            ]
        )

    def _f_band_tail(self, segs, psd, band, band_seg):
        """中位数会抹掉眨眼和动作这类瞬态，这里补充均值和最大值。"""
        linear = 10.0**band_seg
        return np.concatenate([np.log10(linear.mean(axis=0) + _EPS), band_seg.max(axis=0)])

    def _f_transient(self, segs, psd, band, band_seg):
        """因果移动平均差分出低频（约 1–8 Hz，眼动和眨眼）和高频（>15 Hz，肌电）分量。"""
        low = _causal_ma(segs, 8) - _causal_ma(segs, 62)
        low = low[:, 62:]
        high = segs[:, 8:] - _causal_ma(segs, 8)[:, 8:]
        lf = _shape_stats(low)
        hf = _shape_stats(high)
        mad = np.median(np.abs(low), axis=1, keepdims=True) + _EPS
        events = (np.abs(low) > 4.0 * mad).mean()
        skew = np.median(((low - low.mean(axis=1, keepdims=True)) ** 3).mean(axis=1) / (low.std(axis=1) ** 3 + _EPS))
        return np.array([lf[0], lf[1], lf[2], skew, events, lf[3], hf[0], hf[1], hf[2], hf[3]])

    def _f_slow(self, segs, psd, band, band_seg):
        """1 Hz 以下的漂移：电极和头部运动。"""
        m = self._seg_means
        centered = m - m.mean()
        return np.array(
            [
                np.log10(centered.std() + 1.0),
                np.log10(np.ptp(m) + 1.0),
                np.log10(np.median(np.abs(self._seg_slopes)) * FS + 1.0),
            ]
        )

    def _f_line_noise(self, segs, psd, band, band_seg):
        line = np.log10(psd[self.line_mask].mean() + _EPS)
        side = np.log10(psd[self.line_side_mask].mean() + _EPS)
        return np.array([line, line - side])
