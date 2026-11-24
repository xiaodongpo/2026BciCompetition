# -*- coding: utf-8 -*-
"""输入表示对照：频谱MLP、多尺度波形CNN与频谱/可学习能量分支融合。"""
import torch
from torch import nn
from model_priority import MultiScaleCNN


def unpack(x):
    assert x.ndim == 2 and x.shape[1] == 1006
    return x[:,:1000].unsqueeze(1), x[:,1000:]


class SpectralMLP(nn.Module):
    """只读取6个频谱特征，6→16→2；不读取波形，用于检验简单频谱可分性。"""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(6,16), nn.ELU(), nn.Dropout(.3), nn.Linear(16,2))

    def forward(self, x):
        _, spectrum = unpack(x)
        return self.classifier(spectrum)


class FusionBase(nn.Module):
    """复用现有多尺度CNN卷积骨干，统计池化128维，再融合16维辅助特征。"""

    def __init__(self):
        super().__init__()
        self.features = MultiScaleCNN().features
        self.classifier = nn.Sequential(nn.Linear(144,32), nn.ELU(), nn.Dropout(.3), nn.Linear(32,2))

    def waveform_features(self, wave):
        z = self.features(wave)
        return torch.cat((z.mean(-1),z.std(-1,unbiased=False)),dim=1)


class FrequencyFusionCNN(FusionBase):
    """频谱分支：标准化后的5个相对功率与谱熵投影至16维，再同波形特征拼接。"""

    def __init__(self):
        super().__init__()
        self.spectral = nn.Sequential(nn.Linear(6,16), nn.ELU(), nn.Dropout(.15))

    def forward(self, x):
        wave, spectrum = unpack(x)
        return self.classifier(torch.cat((self.waveform_features(wave),self.spectral(spectrum)),dim=1))


class EnergyFusionCNN(FusionBase):
    """可学习能量分支：8个线性时间滤波器→平方→整窗均值→对数→归一化/投影。

    能量先于该分支的归一化与非线性计算，避免先归一化再计算能量。
    LayerNorm在8个对数能量间操作，偏重相对模式；输入本身已经逐窗口MAD缩放。
    """

    def __init__(self):
        super().__init__()
        self.energy_filters = nn.Conv1d(1,8,125,padding=62,bias=False)
        self.energy_projection = nn.Sequential(nn.LayerNorm(8),nn.Linear(8,16),nn.ELU(),nn.Dropout(.15))

    def forward(self, x):
        wave, _ = unpack(x)
        log_power = torch.log(self.energy_filters(wave).square().mean(-1).clamp_min(1e-8))
        auxiliary = self.energy_projection(log_power)
        return self.classifier(torch.cat((self.waveform_features(wave),auxiliary),dim=1))


def build_model(variant):
    return {'SPEC': SpectralMLP, 'FREQ': FrequencyFusionCNN, 'ENERGY': EnergyFusionCNN}[variant]()
