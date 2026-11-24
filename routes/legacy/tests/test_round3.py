import json,unittest
from pathlib import Path
import numpy as np
from src.round3_features import get_features


class RoundThreeTests(unittest.TestCase):
    def test_frozen_groups(self):
        p=json.loads(Path('configs/round3/protocol.json').read_text())
        self.assertEqual(len(p['candidates']),15)
        flattened=sum(p['outer_folds'],[])
        self.assertEqual(len(flattened),len(set(flattened)))
        self.assertEqual(set(flattened),set(p['development']))
        self.assertFalse(set(flattened)&set(p['excluded_previous_holdout']))

    def test_features_are_finite_and_local(self):
        x=np.random.default_rng(5).normal(0,50,(2,1500));edge=np.stack([np.zeros(1500),np.full(1500,403333),np.linspace(-403333,403333,1500)])
        for kind,dim in [('base',125),('normalized',125),('multiscale',217)]:
            self.assertEqual(get_features(x,kind).shape,(2,dim))
            self.assertTrue(np.isfinite(get_features(edge,kind)).all())
            np.testing.assert_allclose(get_features(x[:1],kind)[0],get_features(x,kind)[0],atol=1e-6,rtol=1e-5)
            for n in (1499,1501):
                with self.assertRaises(ValueError):get_features(np.zeros(n),kind)

    def test_gain_and_offset_invariance(self):
        x=np.random.default_rng(7).normal(0,50,(3,1500))
        f=get_features(x,'normalized')
        for gain in (.1,.5,2,10):np.testing.assert_allclose(f,get_features(x*gain+5000,'normalized'),rtol=1e-5,atol=2e-5)


if __name__=='__main__':unittest.main()
