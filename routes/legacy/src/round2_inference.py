"""Endpoint-only, six-second inference for the frozen round-two candidates."""
import numpy as np
from src.inference import StreamingPredictor
from src.round2_features import get_features, singlem_input


class RoundTwoPredictor(StreamingPredictor):
    def __init__(self, model_path, hop_samples=125, units='uV',
                 singlem_weights='third_party/singlem_downstream_excluded.pt'):
        super().__init__(model_path, hop_samples=hop_samples, units=units)
        self.kind = self.config['features']
        self.encoder = None
        if self.kind in ('singlem', 'singlem_random', 'fusion'):
            import torch
            from third_party.singlem_model import Config, EEGEncoder
            torch.manual_seed(20261103)
            config = Config()
            config.mask_prob = 0
            self.encoder = EEGEncoder(config)
            if self.kind != 'singlem_random':
                state = torch.load(singlem_weights, map_location='cpu', weights_only=True)
                if 'encoder_state_dict' in state:
                    assert all(getattr(config, k) == v for k, v in state['model_config'].items())
                    state = state['encoder_state_dict']
                elif 'model_state_dict' in state: state = state['model_state_dict']
                if 'state_dict' in state: state = state['state_dict']
                state = {k.removeprefix('encoder.'): v for k, v in state.items()}
                self.encoder.load_state_dict(state, strict=True)
            self.encoder.eval()

    def predict_window(self, window):
        x = np.asarray(window, dtype=np.float32).reshape(-1)
        if x.shape != (1500,) or not np.isfinite(x).all():
            raise ValueError('Expected 1500 finite raw samples.')
        if self.encoder is None:
            feature = get_features(x, self.kind)
        else:
            import torch
            with torch.no_grad():
                rep = self.encoder(torch.from_numpy(singlem_input(x)))[0]
                feature = torch.cat((rep.mean(1), rep.std(1), rep[:, -1]), 1).numpy()
            if self.kind == 'fusion':
                feature = np.concatenate((get_features(x, 'robust'), feature), axis=1)
        p = float(self.model.predict_proba(feature)[0, 1])
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Invalid task probability.')
        return p
