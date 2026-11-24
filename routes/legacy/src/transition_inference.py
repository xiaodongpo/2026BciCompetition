"""EEG-only CPU CNN adapter. Event-derived training policies are not consulted."""
import numpy as np
import torch
from src.models import SmallCNN
from src.inference import StreamingPredictor


class TransitionCNNPredictor(StreamingPredictor):
    def __init__(self,path,hop_samples=125,units='uV'):
        if not isinstance(hop_samples,int) or hop_samples<1:raise ValueError('Invalid hop')
        if units not in ('uV','V'):raise ValueError('units must be uV or V')
        ck=torch.load(path,map_location='cpu',weights_only=False);self.model=SmallCNN(16,.25);self.model.load_state_dict(ck['model']);self.model.eval()
        self.config=dict(seconds=6,features='cnn');self.size=1500;self.hop=hop_samples;self.units=units;self.threshold=.5;self.reset()

    def predict_window(self,window):
        x=np.asarray(window,dtype=np.float32)
        if x.shape!=(1500,) or not np.isfinite(x).all():raise ValueError('Expected 1500 finite samples')
        with torch.no_grad():p=float(torch.sigmoid(self.model(torch.from_numpy(x.copy())[None,None,:])).item())
        if not 0<=p<=1:raise ValueError('Invalid probability')
        return p
