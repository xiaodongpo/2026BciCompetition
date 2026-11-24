"""R0 ENERGY CNN。参数名称与训练检查点一致，推理无需训练工程。"""
import torch
from torch import nn


class MultiScaleStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches=nn.ModuleList([nn.Conv1d(1,4,k,padding=k//2,bias=False) for k in [25,63,125]])
        self.fusion=nn.Conv1d(12,16,1,bias=False)

    def forward(self,x):
        return self.fusion(torch.cat([branch(x) for branch in self.branches],1))


class EnergyFusionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(MultiScaleStem(),nn.GroupNorm(4,16),nn.ELU(),nn.AvgPool1d(4),nn.Dropout(.15),
            nn.Conv1d(16,32,9,padding=4,bias=False),nn.GroupNorm(4,32),nn.ELU(),nn.AvgPool1d(4),nn.Dropout(.15),
            nn.Conv1d(32,64,7,padding=3,bias=False),nn.GroupNorm(4,64),nn.ELU())
        self.energy_filters=nn.Conv1d(1,8,125,padding=62,bias=False)
        self.energy_projection=nn.Sequential(nn.LayerNorm(8),nn.Linear(8,16),nn.ELU(),nn.Dropout(.15))
        self.classifier=nn.Sequential(nn.Linear(144,32),nn.ELU(),nn.Dropout(.3),nn.Linear(32,2))

    def embedding(self,wave):
        fm=self.features(wave)
        lp=self.energy_filters(wave).square().mean(-1).clamp_min(1e-8).log()
        return torch.cat([fm.mean(-1),fm.std(-1,unbiased=False),self.energy_projection(lp)],1)

    def forward(self,wave):
        return self.classifier(self.embedding(wave))
