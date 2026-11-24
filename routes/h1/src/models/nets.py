"""Small single-channel networks. Input (B, 1, T) at 250 Hz, output one logit."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogPowerNet(nn.Module):
    """Learned filterbank -> squared -> smoothed -> log power, then temporal statistics."""

    def __init__(self, n_filt=32, k=125, pool_s=0.5, dropout=0.5, ds=2):
        super().__init__()
        self.ds = ds
        self.conv = nn.Conv1d(1, n_filt, k // ds | 1, bias=False)
        self.bn = nn.BatchNorm1d(n_filt)
        self.pool = int(pool_s * 250 / ds)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(n_filt * 2, 1)

    def features(self, x):
        x = F.avg_pool1d(x, self.ds)
        h = self.bn(self.conv(x)) ** 2
        h = torch.log(F.avg_pool1d(h, self.pool, self.pool // 2) + 1e-4)
        return torch.cat([h.mean(-1), h.std(-1)], 1)

    def forward(self, x):
        return self.head(self.drop(self.features(x))).squeeze(1)


class EEGNet1D(nn.Module):
    """EEGNet (Lawhern 2018) reduced to one channel: the spatial layer degenerates and is dropped."""

    def __init__(self, f1=16, d=2, f2=32, k1=64, k2=16, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.AvgPool1d(2),
            nn.Conv1d(1, f1, k1 | 1, padding=(k1 | 1) // 2, bias=False),
            nn.BatchNorm1d(f1),
            nn.Conv1d(f1, f1 * d, 1, bias=False),
            nn.BatchNorm1d(f1 * d),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(dropout),
            nn.Conv1d(f1 * d, f1 * d, k2 | 1, padding=(k2 | 1) // 2, groups=f1 * d, bias=False),
            nn.Conv1d(f1 * d, f2, 1, bias=False),
            nn.BatchNorm1d(f2),
            nn.ELU(),
            nn.AvgPool1d(8),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(f2, 1)

    def features(self, x):
        return self.net(x).mean(-1)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class _CausalBlock(nn.Module):
    def __init__(self, ch, k, dil, dropout):
        super().__init__()
        self.pad = (k - 1) * dil
        self.c1 = nn.Conv1d(ch, ch, k, dilation=dil)
        self.c2 = nn.Conv1d(ch, ch, k, dilation=dil)
        self.n1, self.n2 = nn.BatchNorm1d(ch), nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.relu(self.n1(self.c1(F.pad(x, (self.pad, 0))))))
        h = self.drop(F.relu(self.n2(self.c2(F.pad(h, (self.pad, 0))))))
        return x + h


class CausalTCN(nn.Module):
    """Left-padded dilated TCN. receptive_field() is asserted <= 6 s at construction."""

    def __init__(self, ch=32, k=6, levels=5, ds=4, dropout=0.2):
        super().__init__()
        self.ds = ds
        self.inp = nn.Conv1d(1, ch, 15, padding=0)
        self.blocks = nn.ModuleList(_CausalBlock(ch, k, 2**i, dropout) for i in range(levels))
        self.k, self.levels = k, levels
        self.head = nn.Linear(ch, 1)
        assert self.receptive_field() <= 1500, self.receptive_field()

    def receptive_field(self) -> int:
        rf = 15
        for i in range(self.levels):
            rf += 2 * (self.k - 1) * 2**i
        return rf * self.ds

    def features(self, x):
        x = F.avg_pool1d(x, self.ds)
        h = self.inp(F.pad(x, (14, 0)))
        for b in self.blocks:
            h = b(h)
        return h.mean(-1)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class MultiScaleNet(nn.Module):
    """Inception-style stacked 1-D CNN with both learned-power and shape pathways."""

    def __init__(self, ch=24, dropout=0.3):
        super().__init__()
        self.pre = nn.AvgPool1d(2)
        self.branches = nn.ModuleList(nn.Conv1d(1, ch, k | 1, padding=(k | 1) // 2, bias=False) for k in (15, 31, 63, 125))
        self.bn = nn.BatchNorm1d(ch * 4)
        self.stack = nn.Sequential(
            nn.Conv1d(ch * 4, ch * 4, 9, stride=2, padding=4, bias=False), nn.BatchNorm1d(ch * 4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(ch * 4, ch * 4, 9, stride=2, padding=4, bias=False), nn.BatchNorm1d(ch * 4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(ch * 4, ch * 4, 5, padding=2, bias=False), nn.BatchNorm1d(ch * 4), nn.ReLU(),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(ch * 4 * 3, 1)

    def features(self, x):
        x = self.pre(x)
        h = self.bn(torch.cat([b(x) for b in self.branches], 1))
        lp = torch.log(F.avg_pool1d(h**2, 62, 31) + 1e-4).mean(-1)
        s = self.stack(F.relu(h))
        return torch.cat([lp, s.mean(-1), s.amax(-1)], 1)

    def forward(self, x):
        return self.head(self.drop(self.features(x))).squeeze(1)


def build_model(name: str, **kw) -> nn.Module:
    return {"logpower": LogPowerNet, "eegnet": EEGNet1D, "tcn": CausalTCN, "msnet": MultiScaleNet}[name](**kw)
