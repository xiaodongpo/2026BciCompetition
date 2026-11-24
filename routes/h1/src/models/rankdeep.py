"""Deep models with the same per-recording percentile adaptation as the LightGBM pipeline.

A window embedding z (learned by a backbone, optionally concatenated with the 126
hand-crafted features) is compared with the embeddings of the recording's earlier
windows through a differentiable percentile:

    q_d = mean_j sigmoid((z_d - h_jd) / tau)        over history windows j
    r_d = w * q_d + (1 - w) * prior_d,   w = min(n_hist / 30, 1),  prior_d = 0.5 + 0.5 tanh(z_d)

which mirrors OnlineRank (rank_norm.py) with a soft step. Training estimates the
mean from a random sample of the history; inference uses the whole history with the
same formula. Two heads: an adaptive one on r and a strict one on the prior, which
depends only on the current 6-s window. Output: 0.8 * p_adaptive + 0.2 * p_strict.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.nets import CausalTCN, EEGNet1D, LogPowerNet, MultiScaleNet

MIN_HIST = 30


def make_backbone(name: str):
    if name == "none":
        return None, 0
    if name == "logpower":
        return LogPowerNet(n_filt=32), 64
    if name == "tcn":
        return CausalTCN(ch=32), 32
    if name == "eegnet":
        return EEGNet1D(), 32
    if name == "msnet":
        return MultiScaleNet(ch=16), 16 * 4 * 3
    raise ValueError(name)


def _mlp(d, hidden, dropout):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))


class RankDeep(nn.Module):
    def __init__(self, backbone: str, n_hand: int, tau: float = 0.1, hidden: int = 64, dropout: float = 0.3, w_adapt: float = 0.8):
        super().__init__()
        self.backbone, d_emb = make_backbone(backbone)
        self.n_hand = n_hand
        self.dim = d_emb + n_hand
        assert self.dim > 0
        # Per-dimension standardization learned from training batches (running stats at inference).
        self.norm = nn.BatchNorm1d(self.dim, affine=False, momentum=0.01)
        self.tau, self.w_adapt = tau, w_adapt
        self.head_rank = _mlp(self.dim, hidden, dropout)
        self.head_abs = _mlp(self.dim, hidden, dropout)

    def embed(self, x: torch.Tensor | None, hand: torch.Tensor | None) -> torch.Tensor:
        parts = []
        if self.backbone is not None:
            parts.append(self.backbone.features(x))
        if self.n_hand:
            parts.append(hand)
        return self.norm(torch.cat(parts, 1))

    def soft_rank(self, z_t: torch.Tensor, z_ref: torch.Tensor, mask: torch.Tensor, n_hist: torch.Tensor) -> torch.Tensor:
        """z_t (T, D); z_ref (R, D); mask (T, R) True where ref j precedes target t; n_hist (T,) true history size."""
        s = torch.sigmoid((z_t[:, None, :] - z_ref[None, :, :]) / self.tau)
        m = mask.to(z_t.dtype)
        cnt = m.sum(1, keepdim=True)
        q = (s * m[:, :, None]).sum(1) / cnt.clamp(min=1.0)
        w = (n_hist.to(z_t.dtype) / MIN_HIST).clamp(max=1.0)[:, None] * (cnt > 0).to(z_t.dtype)
        return w * q + (1 - w) * self.prior(z_t)

    @staticmethod
    def prior(z: torch.Tensor) -> torch.Tensor:
        return 0.5 + 0.5 * torch.tanh(z)

    def heads(self, r: torch.Tensor, z: torch.Tensor):
        return self.head_rank(r).squeeze(1), self.head_abs(self.prior(z)).squeeze(1)

    def fuse(self, l_rank: torch.Tensor, l_abs: torch.Tensor) -> torch.Tensor:
        return self.w_adapt * torch.sigmoid(l_rank) + (1 - self.w_adapt) * torch.sigmoid(l_abs)


@torch.no_grad()
def exact_sequence_rank(model: RankDeep, z_seq: torch.Tensor, clean_seq: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Inference-time adaptation over a whole sequence in time order.

    Row t is compared with every earlier clean row (its full history), as the
    streaming decoder would. Returns r for every row.
    """
    n = len(z_seq)
    hist_mask = clean_seq.bool()
    n_hist = torch.cumsum(hist_mask.int(), 0) - hist_mask.int()  # clean rows strictly before t
    out = torch.empty_like(z_seq)
    idx = torch.arange(n, device=z_seq.device)
    for s in range(0, n, block):
        t = idx[s : s + block]
        mask = (idx[None, :] < t[:, None]) & hist_mask[None, :]
        out[s : s + block] = model.soft_rank(z_seq[t], z_seq, mask, n_hist[t])
    return out
