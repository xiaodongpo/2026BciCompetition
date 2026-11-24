import collections,json,math,unittest
from pathlib import Path
import numpy as np
from src.round5_features import entropy_pair,ordinal_features


class RoundFiveTests(unittest.TestCase):
    def test_independent_count_reference_with_ties(self):
        z=np.random.default_rng(17).integers(-3,4,size=80).astype(float)
        for m in (3,4):
            for d in (1,4,16):
                counts=collections.Counter();weights=collections.Counter()
                for start in range(len(z)-(m-1)*d):
                    v=z[start:start+m*d:d]
                    if len(set(v))<m:continue
                    pattern=tuple(sorted(range(m),key=lambda j:v[j]))
                    counts[pattern]+=1;weights[pattern]+=float(np.var(v))
                expected=[]
                for c in (counts,weights):
                    total=sum(c.values());ps=[value/total for value in c.values() if value>0]
                    expected.append(-sum(p*math.log(p) for p in ps)/math.log(math.factorial(m)))
                np.testing.assert_allclose(entropy_pair(z[None],m,d)[0],expected,atol=1e-14)

    def test_affine_and_polarity_invariance_with_ties(self):
        z=np.random.default_rng(3).integers(-10,11,size=(4,1250)).astype(float)
        for m in (3,4):
            for d in (1,2,4,8,16):
                ref=entropy_pair(z,m,d)
                np.testing.assert_allclose(ref,entropy_pair(2*z+30,m,d),atol=1e-14)
                np.testing.assert_allclose(ref,entropy_pair(-z,m,d),atol=1e-14)
                np.testing.assert_allclose(ref,entropy_pair(z[:,::-1],m,d),atol=1e-14)

    def test_degenerate_and_input_boundaries(self):
        np.testing.assert_array_equal(ordinal_features(np.zeros((2,1500))),np.zeros((2,20)))
        for n in (1499,1501):
            with self.assertRaises(ValueError):ordinal_features(np.zeros((1,n)))
        with self.assertRaises(ValueError):ordinal_features(np.full((1,1500),np.nan))
        x=np.random.default_rng(4).normal(size=(3,1500))*30
        f=ordinal_features(x);self.assertEqual(f.shape,(3,20));self.assertTrue(((f>=0)&(f<=1)).all())
        np.testing.assert_allclose(f,np.concatenate([ordinal_features(r) for r in x]),atol=1e-7)

    def test_frozen_record_exclusion(self):
        p=json.loads(Path('configs/round5/protocol.json').read_text());rows=sum(p['outer_folds'],[])
        self.assertEqual(len(rows),len(set(rows)));self.assertEqual(set(rows),set(p['development']))
        self.assertFalse(set(rows)&set(p['excluded_previous_holdout']));self.assertEqual(len(p['candidates']),8)


if __name__=='__main__':unittest.main()
