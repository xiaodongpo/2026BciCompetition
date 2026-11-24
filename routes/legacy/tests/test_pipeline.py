import tempfile,unittest,os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from src.data import filename_time,ceil_sample,pair_events,windows
from src.features import extract_features
from src.inference import StreamingPredictor

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.path=Path(cls.tmp.name)/'model.joblib'
        if os.environ.get('BCI_MODEL_PATH'):
            cls.path=Path(os.environ['BCI_MODEL_PATH']);return
        rng=np.random.default_rng(5); x=rng.normal(size=(40,1500)); y=np.arange(40)%2
        model=LogisticRegression(max_iter=1000).fit(extract_features(x),y)
        joblib.dump(dict(model=model,config=dict(seconds=6,features='full'),threshold=.5),cls.path)

    def stream(self,x,chunk,valid=None,path=None):
        s=StreamingPredictor(path or self.path);out=[]
        for a in range(0,len(x),chunk):out.extend(s.process(x[a:a+chunk],start_sample=a,valid=None if valid is None else valid[a:a+chunk]))
        return out

    def test_time_and_pairing(self):
        t=filename_time('001_20261017_170137.984_UTC+0800.bdf')
        self.assertEqual(ceil_sample(t+pd.Timedelta(milliseconds=1),t),1)
        self.assertEqual(ceil_sample(t+pd.Timedelta(milliseconds=4),t),1)
        self.assertEqual(ceil_sample(t-pd.Timedelta(milliseconds=1),t),0)
        e=pd.DataFrame([[(t+pd.Timedelta(seconds=s)).isoformat(),event,phase,''] for s,event,phase in
                        [(0,'trial_start','relax'),(7.2,'trial_end','relax'),(7.3,'trial_start','task'),(8,'key_press','task'),(12.1,'trial_end','task')]],columns=['local_time','event','phase','key'])
        trials,issues=pair_events(e,t,4000);self.assertEqual(len(trials),2);self.assertEqual(trials[0]['end'],1800)

    def test_chunk_and_reference(self):
        x=np.random.default_rng(10).normal(size=5125).astype('float32')
        reference=self.stream(x,1)
        for chunk in (125,250,777,5125):self.assertEqual(reference,self.stream(x,chunk))
        s=StreamingPredictor(self.path)
        for row in reference:
            if row['valid']:
                end=row['sample_index']+1;self.assertEqual(row['p_task'],s.predict_window(x[end-1500:end]))

    def test_future_perturbation(self):
        x=np.random.default_rng(11).normal(size=6000).astype('float32');z=x.copy();z[3000:]*=100000
        a=self.stream(x,6000);b=self.stream(z,6000)
        self.assertEqual([r for r in a if r['sample_index']<3000],[r for r in b if r['sample_index']<3000])

    def test_finite_support(self):
        x=np.random.default_rng(12).normal(size=5000).astype('float32');z=x.copy();z[:-1500]+=90000
        self.assertEqual(self.stream(x,250)[-1],self.stream(z,250)[-1])

    def test_gap_reset_and_units(self):
        x=np.random.default_rng(3).normal(size=6000).astype('float32');valid=np.ones(len(x),bool);valid[2000:2250]=False
        a=self.stream(x,250,valid)
        self.assertTrue(all(not r['valid'] for r in a if 2000<=r['sample_index']<3749))
        self.assertTrue(next(r for r in a if r['sample_index']==3749)['valid'])
        s=StreamingPredictor(self.path);s.process(x[:2000]);o=s.process(x[:250],start_sample=3000);self.assertFalse(o[-1]['valid'])
        with self.assertRaises(ValueError):s.process(x[:1],start_sample=0)
        s.reset();self.assertEqual(s.next_sample,0);self.assertEqual(len(s.buffer),0)
        v=StreamingPredictor(self.path,units='V').process(x[:1500]/1e6)[-1]['p_task']
        u=StreamingPredictor(self.path).process(x[:1500])[-1]['p_task'];self.assertAlmostEqual(u,v,places=6)

    def test_windows_exclude_transitions_and_gaps(self):
        x=np.ones(4000);valid=np.ones(4000,bool);valid[2200]=False;y=np.zeros(4000,dtype=int);y[2000:]=1
        for _,label,end in windows(x,valid,y,6,.5):
            self.assertTrue((y[end-1500:end]==label).all());self.assertTrue(valid[end-1500:end].all())

    def test_feature_modes_are_finite(self):
        for mode in ('full','relative','fir','detrend','fir_2_40','fir_4_40'):
            for x in (np.zeros(1500),np.full(1500,403333),np.random.default_rng(2).normal(size=1500)):
                self.assertTrue(np.isfinite(extract_features(x,mode=mode)).all())

if __name__=='__main__':unittest.main()
