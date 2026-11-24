"""Third-round models share the already-tested bounded streaming scheduler."""
import numpy as np
from src.inference import StreamingPredictor
from src.round3_features import get_features


class RoundThreePredictor(StreamingPredictor):
    def predict_window(self,window):
        x=np.asarray(window,dtype=np.float32)
        if x.ndim!=1 or x.shape!=(1500,) or not np.isfinite(x).all():raise ValueError('Expected one finite 1500-sample window.')
        p=float(self.model.predict_proba(get_features(x,self.config['features']))[0,1])
        if not np.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid P(task).')
        return p
