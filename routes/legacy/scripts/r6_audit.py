"""Read-only model/split/scaler audit, final LR refit and original byte integrity."""
import json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from src.transition_training import policy,fit_lr,predict_lr


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    p=json.loads(Path('configs/round6/protocol.json').read_text());dev=set(p['development']);excluded=set(p['excluded_previous_holdout']);meta=pd.read_csv('cache/round6/windows.csv',dtype={'record':str});x=np.load('cache/round6/base.npy')
    for name in ('lr','cnn'):
        m=json.loads(Path(f'experiments/round6/{name}_manifest.json').read_text())
        for source,sha in m['sources'].items():assert digest(source)==sha,source
        for source,sha in m['data']['digests'].items():assert digest(Path('cache/round6')/source)==sha,source
    count=0;max_error=0.
    for path in Path('experiments/round6/lr').rglob('*.joblib'):
        b=joblib.load(path);train=set(b['training_records']);assert train<=dev and not train&excluded;c=b['config']['training_policy']
        if path.parent.name in ('inner','final_selection'):
            s=json.loads((path.parent/'splits.json').read_text());fold=int(path.stem.rsplit('inner',1)[1]);valid=set(s['validation_folds'][fold]);assert train==set(s['training_records'])-valid
            logged=pd.read_csv(path.parent/(c['id']+'.csv'),dtype={'record':str});logged=logged[logged.inner_fold==fold]
            details=json.loads((path.parent/(c['id']+'.json')).read_text());assert digest(path)==details['fits'][fold]['sha256']
        else:
            s=json.loads((path.parent/'split.json').read_text());valid=set(s['validation']);assert train==set(s['train']) and not train&valid
            logged=pd.read_csv(path.with_suffix('.csv'),dtype={'record':str})
        tr=meta.record.isin(train);va=meta.record.isin(valid);_,w=policy(meta[tr],c);kept=x[tr][w>0]
        assert b['model'][0].n_samples_seen_==len(kept);np.testing.assert_allclose(b['model'][0].mean_,kept.mean(0,dtype=np.float64),rtol=0,atol=1e-12)
        assert logged[['record','end','y']].reset_index(drop=True).equals(meta.loc[va,['record','end','y']].reset_index(drop=True))
        prob=predict_lr(b['model'],x[va]);err=float(np.max(abs(prob-logged.p.to_numpy())));assert err<2e-6;np.testing.assert_array_equal(prob>=.46,logged.p.to_numpy()>=.46);max_error=max(max_error,err);count+=1
    assert count==682,count
    final=joblib.load('checkpoints/round6/lr_candidate.joblib');assert set(final['training_records'])==dev
    start=time.perf_counter();refit=fit_lr(x,meta,final['config']['training_policy']);seconds=time.perf_counter()-start
    np.testing.assert_array_equal(refit[1].coef_,final['model'][1].coef_);np.testing.assert_array_equal(refit[1].intercept_,final['model'][1].intercept_)
    error=float(np.max(abs(predict_lr(refit,x)-predict_lr(final['model'],x))));assert error<2e-6
    old=json.loads(Path('reports/round6/prior_archives.json').read_text())
    for name,sha in old.items():assert digest(name)==sha
    originals=pd.read_csv('reports/original_manifest.csv')
    for r in originals.itertuples(index=False):assert Path(r.path).stat().st_size==r.bytes and digest(r.path)==r.sha256
    result=dict(lr_checkpoints=count+1,sources_and_inputs_match=True,all_training_splits_verified=True,all_validation_rows_preserved=True,all_scaler_means_match_retained_training_rows=True,max_logged_probability_error=max_error,all_lr_logged_decisions_equal=True,final_refit_coefficients_bitwise=True,final_refit_max_probability_error=error,refit_seconds=seconds,original_files_unchanged=len(originals),original_bytes=int(originals.bytes.sum()),old_archives_unchanged=old,final_lr_checkpoint_sha256=digest('checkpoints/round6/lr_candidate.joblib'))
    Path('reports/round6/lr_integrity_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
