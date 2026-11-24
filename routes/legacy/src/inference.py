"""Bounded-memory EEG streaming inference. No phase/event information is accepted."""
from pathlib import Path
import numpy as np
import joblib
from src.features import extract_features

class StreamingPredictor:
    def __init__(self, model_path, hop_samples=125, units='uV'):
        bundle=joblib.load(model_path)
        self.model=bundle['model']; self.config=bundle['config'];self.threshold=float(bundle['threshold'])
        self.size=int(self.config['seconds']*250)
        if not 250<=self.size<=1500:raise ValueError('Window must be 1-6 s')
        if not isinstance(hop_samples,int) or hop_samples<1:raise ValueError('Invalid hop')
        if units not in ('uV','V'):raise ValueError('units must be uV or V')
        self.hop=hop_samples;self.units=units;self.reset()

    def reset(self,start_sample=0):
        if not isinstance(start_sample,(int,np.integer)) or start_sample<0:raise ValueError('Invalid start index')
        self.next_sample=int(start_sample)
        self.buffer=np.empty(0,dtype=np.float32)

    def predict_window(self,window):
        z=np.asarray(window,dtype=np.float32)
        if z.ndim!=1 or len(z)!=self.size or not np.isfinite(z).all():raise ValueError('Invalid full window')
        f=extract_features(z,mode=self.config.get('features','full'))
        p=float(self.model.predict_proba(f)[0,1])
        if not np.isfinite(p) or not 0<=p<=1:raise ValueError('Model returned invalid probability')
        return p

    def process(self,eeg,start_sample=None,valid=None):
        x=np.asarray(eeg,dtype=np.float32)
        if x.ndim!=1:raise ValueError('Supply one-dimensional single-channel samples')
        if self.units=='V':x=x*1e6
        ok=np.isfinite(x)
        if valid is not None:
            mask=np.asarray(valid,dtype=bool)
            if mask.shape!=x.shape:raise ValueError('Invalid validity-mask shape')
            ok &= mask
        if start_sample is not None:
            if not isinstance(start_sample,(int,np.integer)) or start_sample<self.next_sample:
                raise ValueError('Out-of-order sample index')
            if start_sample!=self.next_sample:self.reset(start_sample)
        outputs=[]
        # Split at emission boundaries; never inspect the rest of an arriving block for a past output.
        pos=0
        while pos<len(x):
            take=min(self.hop-self.next_sample%self.hop,len(x)-pos)
            segment=x[pos:pos+take]; mask=ok[pos:pos+take]
            bad=np.flatnonzero(~mask)
            if len(bad):self.buffer=segment[bad[-1]+1:].copy()
            else:self.buffer=np.concatenate((self.buffer,segment))[-self.size:]
            self.buffer=self.buffer[-self.size:]
            pos+=take;self.next_sample+=take
            if self.next_sample%self.hop==0:
                ready=len(self.buffer)==self.size
                p=self.predict_window(self.buffer) if ready else .5
                outputs.append(dict(sample_index=self.next_sample-1,time_s=(self.next_sample-1)/250,
                                    label=int(p>=self.threshold) if ready else None,p_task=p,valid=ready,
                                    status='ok' if ready else 'warming_up_or_gap'))
        return outputs
