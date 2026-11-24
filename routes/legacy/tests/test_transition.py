import json,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.transition_training import policy
from src.transition_evaluation import measures


class TransitionTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(record=['a']*8+['b']*8,y=[0,0,0,0,1,1,1,1]*2,t=[0,5,20,50]*4,valid_switch=[True]*16,pure=[False,False,True,True]*4))

    def test_direction_independence_and_initial_phase(self):
        m=self.frame();q,w=policy(m,dict(kind='drop',T_task=15,T_relax=0))
        self.assertTrue((w[m.y==0]>0).all());self.assertTrue((w[(m.y==1)&(m.t<15)]==0).all())
        m['valid_switch']=False;_,w=policy(m,dict(kind='drop',T_task=30,T_relax=30));self.assertTrue((w>0).all())

    def test_equal_record_class_totals_and_soft_targets(self):
        m=self.frame();q,w=policy(m,dict(kind='soft',T_task=30,T_relax=15));m['w']=w
        totals=m.groupby(['record','y']).w.sum().to_numpy();np.testing.assert_allclose(totals,totals[0])
        self.assertEqual(q[0],.5);self.assertEqual(q[4],.5);self.assertEqual(q[3],0);self.assertEqual(q[7],1)
        self.assertNotEqual(q[1],1-q[5])

    def test_soft_bce_equivalence(self):
        m=self.frame();q,w=policy(m,dict(kind='soft',T_task=30,T_relax=15));p=np.linspace(.1,.9,len(m))
        direct=np.sum(w*(-q*np.log(p)-(1-q)*np.log(1-p)))
        copies=np.sum(w*q*(-np.log(p)))+np.sum(w*(1-q)*(-np.log(1-p)))
        self.assertAlmostEqual(direct,copies,places=12)

    def test_curriculum_and_no_mutation(self):
        m=self.frame();original=m.copy(deep=True);_,early=policy(m,dict(kind='curriculum',T_task=30,T_relax=30),0.)
        _,late=policy(m,dict(kind='curriculum',T_task=30,T_relax=30),1.)
        self.assertTrue((early[m.t<30]==0).all());self.assertTrue((late>0).all());pd.testing.assert_frame_equal(m,original)

    def test_single_class_auc_and_folds(self):
        self.assertIsNone(measures([1,1],[.2,.8])['auc'])
        p=json.loads(Path('configs/round6/protocol.json').read_text());rows=sum(p['outer_folds'],[])
        self.assertEqual(len(rows),len(set(rows)));self.assertEqual(set(rows),set(p['development']));self.assertFalse(set(rows)&set(p['excluded_previous_holdout']))


if __name__=='__main__':unittest.main()
