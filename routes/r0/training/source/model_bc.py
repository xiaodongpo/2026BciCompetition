# -*- coding: utf-8 -*-
"""CNN结构对照：B仅修改池化；C使用共享的1秒CNN与轻量片段时间卷积。"""
import torch
from torch import nn
from model import EEGCNN


class MeanStdCNN(nn.Module):
    """B：保留原三层卷积，拼接时间均值、总体标准差，送入128→32→2分类头。"""

    def __init__(self):
        super().__init__()
        baseline = EEGCNN()
        self.features = nn.Sequential(*list(baseline.features.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(128, 32), nn.ELU(), nn.Dropout(0.3), nn.Linear(32, 2)
        )

    def forward(self, x):
        assert x.ndim == 3 and x.shape[1:] == (1, 1000)
        z = self.features(x)
        # unbiased=False计算总体标准差，避免短序列的样本校正。
        pooled = torch.cat((z.mean(-1), z.std(-1, unbiased=False)), dim=1)
        return self.classifier(pooled)


class PatchTemporalCNN(nn.Module):
    """C：4×250点共享原CNN编码器，再以残差卷积整合4个片段；最终只用均值池化。"""

    def __init__(self):
        super().__init__()
        self.encoder = EEGCNN().features
        # 输入[B,64,4]。卷积仅访问当前4秒窗口内的片段，不跨窗口传递状态。
        self.temporal = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ELU(),
            nn.Dropout(0.15),
            nn.Conv1d(64, 64, 3, padding=1),
        )
        self.activation = nn.ELU()
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ELU(), nn.Dropout(0.3), nn.Linear(32, 2)
        )

    def forward(self, x):
        assert x.ndim == 3 and x.shape[1:] == (1, 1000)
        batch = x.shape[0]
        # 250Hz下1秒=250点，保持原4秒窗口与标签，按时间顺序折叠到批维共享权重。
        patches = x.reshape(batch, 4, 250).reshape(batch * 4, 1, 250)
        z = self.encoder(patches).reshape(batch, 4, 64).transpose(1, 2)
        z = self.activation(z + self.temporal(z))
        return self.classifier(z.mean(-1))


def build_model(variant):
    return {'B': MeanStdCNN, 'C': PatchTemporalCNN}[variant]()
