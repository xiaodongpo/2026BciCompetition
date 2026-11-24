import json,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.round4_model import training_arrays,fit,predict
from scripts.round2_train import sample_weights

class RoundFourTests(unittest.TestCase):
    def test_augmentation_weight_and_label_conservation(self):
        meta=pd.DataFrame({'record':['a']*6+['b']*4,'y':[0,0,0,0,1,1,0,0,1,1]})
        x=np.arange(10*375).reshape(10,375);xx,y,w=training_arrays(x,meta)
        np.testing.assert_array_equal(xx[:10],x[:,:125]);np.testing.assert_array_equal(y,np.tile(meta.y,3))
        np.testing.assert_allclose(w.reshape(3,10).sum(0),sample_weights(meta));self.assertAlmostEqual(w.sum(),10)

    def test_scaler_sees_original_train_only_and_predictions_ignore_extra_views(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(100,375)).astype('float32');x[:,125:]+=1000
        meta=pd.DataFrame({'record':['a']*50+['b']*50,'y':np.tile([0,1],50)})
        model=fit({'C':1},x,meta)
        np.testing.assert_allclose(model[0].mean_,x[:,:125].mean(0,dtype=np.float64));self.assertEqual(model[0].n_samples_seen_,100)
        other=x.copy();other[:,125:]=-5000
        np.testing.assert_array_equal(predict(model,x),predict(model,other))

    def test_frozen_split(self):
        p=json.loads(Path('configs/round4/protocol.json').read_text());all_rows=sum(p['outer_folds'],[])
        self.assertEqual(len(all_rows),len(set(all_rows)));self.assertEqual(set(all_rows),set(p['development']))
        self.assertFalse(set(all_rows)&set(p['excluded_previous_holdout']));self.assertEqual(p['threshold'],.46)

if __name__=='__main__':unittest.main()
