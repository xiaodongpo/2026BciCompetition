# -*- coding: utf-8 -*-
"""D/E/F结构对照；复用B/C代码，不覆盖既有模型及训练结果。"""
import torch
from torch import nn
from model import EEGCNN
from model_bc import MeanStdCNN, PatchTemporalCNN


def mean_std(z, dim):
    """拼接沿指定维度的均值与总体标准差。"""
    return torch.cat((z.mean(dim), z.std(dim, unbiased=False)), dim=1)


class PatchTemporalStatsCNN(PatchTemporalCNN):
    """D：保留C的共享1秒编码器和残差时间卷积，只修改最终池化及分类头输入。"""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(128, 32), nn.ELU(), nn.Dropout(0.3), nn.Linear(32, 2)
        )

    def forward(self, x):
        assert x.ndim == 3 and x.shape[1:] == (1, 1000)
        batch = x.shape[0]
        patches = x.reshape(batch * 4, 1, 250)
        z = self.encoder(patches).reshape(batch, 4, 64).transpose(1, 2)
        z = self.activation(z + self.temporal(z))
        return self.classifier(mean_std(z, -1))


class GroupNormStatsCNN(MeanStdCNN):
    """E：以B为基线，仅将三处BatchNorm替换为4组GroupNorm。"""

    def __init__(self):
        super().__init__()
        for index, layer in enumerate(self.features):
            if isinstance(layer, nn.BatchNorm1d):
                self.features[index] = nn.GroupNorm(4, layer.num_features)


class PatchAttentionStatsCNN(nn.Module):
    """F：4个1秒CNN token，加窗口内位置编码及1层4头注意力，再做均值/标准差池化。

    为明确比较关系：与D共享分段、卷积编码器和池化方案，时间建模由TCN改为注意力。
    不使用绝对实验时间、跨窗口状态或窗口结束后的信号。
    """

    def __init__(self):
        super().__init__()
        self.encoder = EEGCNN().features
        self.time_position = nn.Parameter(torch.zeros(1, 4, 64))
        nn.init.normal_(self.time_position, std=0.02)
        self.attention = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.15,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.output_norm = nn.LayerNorm(64)
        self.classifier = nn.Sequential(
            nn.Linear(128, 32), nn.ELU(), nn.Dropout(0.3), nn.Linear(32, 2)
        )

    def forward(self, x):
        assert x.ndim == 3 and x.shape[1:] == (1, 1000)
        batch = x.shape[0]
        z = self.encoder(x.reshape(batch * 4, 1, 250)).reshape(batch, 4, 64)
        z = self.output_norm(self.attention(z + self.time_position))
        return self.classifier(mean_std(z, 1))


def build_model(variant):
    return {'D': PatchTemporalStatsCNN, 'E': GroupNormStatsCNN,
            'F': PatchAttentionStatsCNN}[variant]()
