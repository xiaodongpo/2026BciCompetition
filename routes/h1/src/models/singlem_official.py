"""Load SingLEM through the official repository code (github.com/ttlabtuat/SingLEM, MIT).

Follows the repository README: Config from the checkpoint's model_config,
mask_prob = 0.0, EEGEncoder, load_state_dict, eval. The encoder returns
(features, mask_idx, seq_len); callers use features with shape (batch, tokens, 16).
"""
from __future__ import annotations

import os
import sys

import torch

REPO = os.environ.get("SINGLEM_REPO", os.path.join(os.path.dirname(__file__), "..", "..", "pretrained", "singlem"))
CKPT = os.path.join(REPO, "SingLEM", "checkpoints", "singlem_downstream_included.pt")


def _import():
    repo = os.path.abspath(REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from SingLEM.model import Config, EEGEncoder  # official code

    return Config, EEGEncoder


class OfficialSingLEM(torch.nn.Module):
    """Thin wrapper so the official encoder returns only the features tensor."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x.contiguous())[0]  # the official forward uses .view()


def load_official(ckpt: str = CKPT) -> OfficialSingLEM:
    Config, EEGEncoder = _import()
    config = Config()
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    for name, value in payload["model_config"].items():
        setattr(config, name, value)
    config.mask_prob = 0.0
    model = EEGEncoder(config)
    model.load_state_dict(payload["encoder_state_dict"])
    model.eval()
    return OfficialSingLEM(model)
