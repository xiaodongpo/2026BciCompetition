import torch
from torch import nn
import torch.nn.functional as F

def normalize_window(x):
    # A completed trailing window is available at decision time. No external history.
    x=x-x.mean(dim=-1,keepdim=True)
    scale=x.square().mean(dim=-1,keepdim=True).sqrt().clamp_min(1)
    return (x/scale).clamp(-10,10),scale.log().squeeze(-1)

class SmallCNN(nn.Module):
    def __init__(self,width=16,dropout=.25):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv1d(1,width,31,stride=5,padding=15),nn.BatchNorm1d(width),nn.ELU(),
            nn.Conv1d(width,width*2,15,stride=5,padding=7),nn.BatchNorm1d(width*2),nn.ELU(),
            nn.Conv1d(width*2,width*2,7,padding=3),nn.ELU(),nn.AdaptiveAvgPool1d(1))
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(width*2+1,1))
    def forward(self,x):
        z,scale=normalize_window(x);h=self.encoder(z).squeeze(-1)
        return self.head(torch.cat((h,scale/10),dim=1)).squeeze(-1)

class CausalConv(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__();self.pad=4*dilation
        self.conv=nn.Conv1d(channels,channels,5,dilation=dilation)
    def forward(self,x):return F.elu(self.conv(F.pad(x,(self.pad,0))))+x

class CausalTCN(nn.Module):
    def __init__(self,width=16,dropout=.25,downsample=5):
        super().__init__()
        self.downsample=downsample
        self.front=nn.Conv1d(1,width,15,stride=5)
        self.blocks=nn.Sequential(*[CausalConv(width,d) for d in (1,2,4,8,16)])
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(width+1,1))
    def forward(self,x):
        # Fixed scaling makes every TCN node causal; no temporal normalization within the network.
        z=torch.asinh((x-x[:,:,:1])/100)
        z=F.avg_pool1d(z,self.downsample,self.downsample)
        h=self.blocks(F.elu(self.front(F.pad(z,(14,0)))))
        return self.head(torch.cat((h.mean(-1),z.square().mean(-1).clamp_min(1e-8).log()),dim=1)).squeeze(-1)

def build_model(c):
    if c['model']=='cnn':return SmallCNN(c.get('width',16),c.get('dropout',.25))
    if c['model']=='tcn':return CausalTCN(c.get('width',16),c.get('dropout',.25),c.get('downsample',5))
    raise ValueError(c)
