import unittest,json
from pathlib import Path
import numpy as np
from src.round2_features import get_features,singlem_input

class RoundTwoTests(unittest.TestCase):
    def test_frozen_record_isolation(self):
        p=json.loads(Path('configs/round2/protocol.json').read_text())
        self.assertFalse(set(p['development'])&set(p['excluded_previous_holdout']))
        self.assertEqual(len(p['development']),45)
        self.assertEqual(set().union(*map(set,p['outer_folds'])),set(p['development']))
        for i,a in enumerate(p['outer_folds']):
            for b in p['outer_folds'][i+1:]:self.assertFalse(set(a)&set(b))

    def test_finite_constant_and_extreme_windows(self):
        x=np.stack([np.zeros(1500),np.full(1500,403333),np.linspace(-400000,400000,1500)])
        for k in ('legacy','robust','filterbank'):self.assertTrue(np.isfinite(get_features(x,k)).all())
        self.assertEqual(singlem_input(x).shape,(3,5,128));self.assertTrue(np.isfinite(singlem_input(x)).all())

    def test_no_cross_example_statistics(self):
        a=np.random.default_rng(3).normal(size=(1,1500))*50;b=np.random.default_rng(4).normal(size=(2,1500))*10000
        for k in ('legacy','robust','filterbank'):
            np.testing.assert_allclose(get_features(a,k)[0],get_features(np.concatenate([a,b]),k)[0],rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(singlem_input(a)[0],singlem_input(np.concatenate([a,b]))[0],atol=1e-6)

    def test_raw_support_limit(self):
        for n in (1250,1501,1750):
            with self.assertRaises(ValueError):get_features(np.zeros(n),'robust')

if __name__=='__main__':unittest.main()
