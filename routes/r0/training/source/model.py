# -*- coding: utf-8 -*-
"""传统一维CNN：三个卷积块提取时间特征，全局平均池化后二分类，无注意力模块。"""
import torch
from torch import nn


class EEGCNN(nn.Module):
    """输入[B,1,1000]，输出[B,2]的logits；0为relax，1为task。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=25, padding=12, bias=False),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(0.15),
            nn.Conv1d(16, 32, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(0.15),
            nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        assert x.ndim == 3 and x.shape[1:] == (1, 1000)
        return self.classifier(self.features(x))
