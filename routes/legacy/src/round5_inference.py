"""Round-five CPU adapter; inherited buffer is limited to 1500 raw samples."""
import numpy as np
from src.inference import StreamingPredictor
from src.round5_features import get_features


class RoundFivePredictor(StreamingPredictor):
    def predict_window(self, window):
        x = np.asarray(window, dtype=np.float32).reshape(-1)
        feature = get_features(x, self.config['features'])
        p = float(self.model.predict_proba(feature)[0, 1])
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Invalid task probability.')
        return p
