import json,hashlib
from pathlib import Path
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from scripts.train_classical import build,load

def run():
    s=json.loads(Path('configs/splits.json').read_text());bundle=joblib.load('submission/model.joblib');c=bundle['config']
    x,m=load(c);tr=m.record.isin(s['development']);ho=m.record.isin(s['holdout'])
    assert not set(m[tr].record)&set(m[ho].record)
    assert set(bundle['training_records'])==set(s['development'])
    assert len(set().union(*map(set,s['folds'])))==45
    for i,a in enumerate(s['folds']):
        for b in s['folds'][i+1:]:assert not set(a)&set(b)
    np.testing.assert_allclose(bundle['model'][0].mean_,x[tr].mean(axis=0,dtype=np.float64),rtol=1e-12,atol=1e-12)
    with threadpool_limits(limits=4):fresh=build(c).fit(x[tr],m.y[tr])
    np.testing.assert_array_equal(fresh[-1].coef_,bundle['model'][-1].coef_)
    np.testing.assert_array_equal(fresh[-1].intercept_,bundle['model'][-1].intercept_)
    np.testing.assert_array_equal(fresh.predict_proba(x[tr][:100]),bundle['model'].predict_proba(x[tr][:100]))
    result=dict(record_disjointness=True,development_only_scaler=True,refit_coefficients_bitwise_equal=True,refit_probabilities_bitwise_equal=True,
                model_sha256=hashlib.sha256(Path('submission/model.joblib').read_bytes()).hexdigest())
    Path('reports/reproduction_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':run()
