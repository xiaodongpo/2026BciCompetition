# -*- coding: utf-8 -*-
"""以E为基线：多尺度首层卷积、轻量残差扩张卷积，两项独立比较。"""
import torch
from torch import nn
from model_def import GroupNormStatsCNN


class MultiScaleStem(nn.Module):
    """25/63/125点并行卷积，各4个特征，再以1×1卷积融合到原16维。

    参数量仅比原首层增加644，不额外改变后续归一化、池化和分类头。
    卷积核对应不同时间范围，并非预先指定的频带滤波器。
    """

    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(1, 4, k, padding=k // 2, bias=False) for k in [25, 63, 125]
        ])
        self.fusion = nn.Conv1d(12, 16, 1, bias=False)

    def forward(self, x):
        return self.fusion(torch.cat([branch(x) for branch in self.branches], dim=1))


class MultiScaleCNN(GroupNormStatsCNN):
    """E_MS：仅替换E首层，完整保留4秒连续输入和均值/标准差池化。"""

    def __init__(self):
        super().__init__()
        self.features[0] = MultiScaleStem()


class ResidualDilatedBlock(nn.Module):
    """在下采样特征序列上以膨胀率2/4扩大感受野，深度卷积限制参数量。"""

    def __init__(self):
        super().__init__()
        layers = []
        for dilation in [2, 4]:
            layers.extend([
                nn.Conv1d(64, 64, 3, padding=dilation, dilation=dilation,
                          groups=64, bias=False),
                nn.Conv1d(64, 64, 1, bias=False),
                nn.GroupNorm(4, 64), nn.ELU(), nn.Dropout(0.15),
            ])
        self.branch = nn.Sequential(*layers)
        self.activation = nn.ELU()

    def forward(self, x):
        return self.activation(x + self.branch(x))


class ResidualDilatedCNN(GroupNormStatsCNN):
    """E_RD：仅在E最后卷积之后、统计池化之前添加残差扩张块。"""

    def __init__(self):
        super().__init__()
        self.features.add_module('residual_dilated', ResidualDilatedBlock())


def build_model(variant):
    return {'E': GroupNormStatsCNN, 'E_MS': MultiScaleCNN,
            'E_RD': ResidualDilatedCNN}[variant]()
