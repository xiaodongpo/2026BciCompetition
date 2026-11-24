"""SingLEM encoder (Sukhbaatar et al., arXiv:2509.17920), written for this project.

Implements the published architecture so that the MIT-licensed checkpoints from
github.com/ttlabtuat/SingLEM load strictly as plain tensors; no code from that
repository is imported or executed. Attention is bidirectional, but only over
tokens inside the input window, which never extends past the decision time.

Parameter names mirror the checkpoint keys.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class BandFeat(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv1d(1, c.filter_num, c.filter1_size, padding=c.filter1_size // 2)
        self.conv2 = nn.Conv1d(c.filter_num, c.filter_num, c.filter2_size, padding=c.filter2_size // 2)
        self.conv3 = nn.Conv1d(c.filter_num, 1, 1)
        self.LN1 = nn.LayerNorm(c.filter_num)
        self.LN2 = nn.LayerNorm(c.filter_num)
        self.dropout = nn.Dropout(c.cnn_dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = F.elu(self.dropout(self.LN1(x.transpose(1, 2)).transpose(1, 2)))
        x = self.conv2(x)
        x = F.elu(self.dropout(self.LN2(x.transpose(1, 2)).transpose(1, 2)))
        return F.elu(self.conv3(x))


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.concat_attn = nn.Linear(d_model, 3 * d_model)
        self.concat_proj = nn.Linear(d_model, d_model)
        self.n_heads, self.d_model = n_heads, d_model

    def forward(self, x):
        b, s, d = x.size()
        q, k, v = self.concat_attn(x).split(d, dim=2)
        shape = (b, s, self.n_heads, d // self.n_heads)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous().view(b, s, d)
        return self.concat_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_1 = nn.Linear(d_model, 4 * d_model)
        self.W_2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.W_2(F.gelu(self.W_1(x), approximate="tanh"))


class Layer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.self_att_heads = SelfAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.self_att_heads(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GlobalTransformer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pos_embedding = nn.Embedding(c.max_seq_len, c.d_model)
        self.Transformer_layers = nn.Sequential(*[Layer(c.d_model, c.n_heads) for _ in range(c.n_layers)])
        self.ln_final = nn.LayerNorm(c.d_model)

    def forward(self, x):
        pos = self.pos_embedding(torch.arange(x.size(1), device=x.device))
        return self.ln_final(self.Transformer_layers(x + pos))


class TokenEmbedding(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c.emb_d_model
        self.emb_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embedding = nn.Embedding(c.emb_seq_len + 1, d)
        self.linear_proj = nn.Linear(c.cnn_features, d)
        self.Emb_Transformer_layers = nn.Sequential(*[Layer(d, c.emb_n_heads) for _ in range(c.emb_n_layers)])
        self.ln = nn.LayerNorm(d)
        self.proj = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, c.emb_dim), nn.GELU(), nn.Linear(c.emb_dim, c.d_model))

    def forward(self, x):
        x = self.linear_proj(x)
        x = torch.cat([self.emb_token.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.pos_embedding(torch.arange(x.size(1), device=x.device))
        return self.proj(self.ln(self.Emb_Transformer_layers(x)[:, 0]))


class SingLEMEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        c = SimpleNamespace(**cfg)
        self.window_len = c.emb_seq_len
        self.cnn = BandFeat(c)
        self.Embedding = TokenEmbedding(c)
        self.TransformerEncoder = GlobalTransformer(c)
        self.DimRedLayer = nn.Linear(c.d_model, c.rep_dim)

    def forward(self, x):
        """(batch, tokens, 128) -> (batch, tokens, rep_dim)."""
        b, s, t = x.shape
        feat = self.cnn(x.reshape(b * s, 1, t)).squeeze(1).view(b, s, -1)
        pad = self.window_len - 1
        feat = F.pad(feat.transpose(1, 2), (pad // 2, pad - pad // 2), mode="replicate").transpose(1, 2)
        ctx = feat.unfold(1, self.window_len, 1).permute(0, 1, 3, 2).reshape(b * s, self.window_len, -1)
        emb = self.Embedding(ctx).view(b, s, -1)
        return self.DimRedLayer(self.TransformerEncoder(emb))


DEFAULTS = dict(
    token_len=128, n_heads=8, d_model=128, n_layers=12, max_seq_len=16, filter_num=32, filter1_size=3, filter2_size=61,
    cnn_dropout=0.3, cnn_features=128, emb_n_heads=4, emb_n_layers=4, emb_seq_len=5, emb_dim=32, emb_d_model=128, rep_dim=16,
)


def load_singlem(ckpt_path: str) -> SingLEMEncoder:
    """Tensors only: weights_only=True refuses any pickled code in the file."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = {**DEFAULTS, **{k: v for k, v in ck.get("model_config", {}).items() if k in DEFAULTS}}
    m = SingLEMEncoder(cfg)
    m.load_state_dict(ck["encoder_state_dict"], strict=True)
    return m.eval()
