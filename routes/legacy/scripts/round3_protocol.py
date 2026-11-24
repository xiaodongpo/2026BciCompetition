"""Freeze hypotheses, models and evaluation before third-round results exist."""
import hashlib,json
from pathlib import Path


def ranked(records,salt):
    return sorted(records,key=lambda r:hashlib.sha256((salt+r).encode()).hexdigest())


def run():
    for p in ['configs','cache','reports','results','experiments','checkpoints']:(Path(p)/'round3').mkdir(exist_ok=True)
    previous=json.loads(Path('configs/round2/protocol.json').read_text())
    dev=previous['development'];order=ranked(dev,'round3-record-2026110303-')
    candidates=[]
    for features in ('base','normalized','multiscale'):
        for C in (1,10):candidates.append(dict(id=f'{features}_lr_C{C}',features=features,model='lr',C=C))
        for leaves in (8,15):candidates.append(dict(id=f'{features}_hgb_L{leaves}',features=features,model='hgb',leaves=leaves,max_iter=160,learning_rate=.05,l2=10,min_leaf=100,max_bins=63))
        candidates.append(dict(id=f'{features}_rff512_C1',features=features,model='rff',components=512,gamma_factor=.5,C=1))
    config=dict(round=3,seed=2026110303,development=dev,excluded_previous_holdout=previous['excluded_previous_holdout'],
        outer_folds=[sorted(order[k::3]) for k in range(3)],inner_folds=3,inner_split_salt='round3-inner-2026110303-',candidates=candidates,
        fixed_reference=dict(id='base_lr_C10',features='base',model='lr',C=10,threshold=.46),
        hypotheses=['Within-window gain normalization may reduce amplitude nuisance without identity metadata.','Recent 2s and 3s spectral features may complement whole-window descriptors.','Fixed low-capacity nonlinear classifiers may improve on linear decoding.'],
        representations=dict(base='Unchanged round-two 125-dimensional filterbank features.',normalized='Scale the centered 1500 raw samples using MAD of their own 1-45Hz filtered samples; map MAD to 20uV, floor 1e-6uV; then apply unchanged filterbank features.',multiscale='125 base features plus 46 features each from the last 500 and 750 filtered points. All FIR history stays inside the same 1500 raw samples.'),
        selection='Maximize mean record macro bACC@0.5 and AUC from 3 inner OOF folds; first candidate breaks ties.',threshold='Record/class weighted inner OOF bACC, grid 0.35:0.01:0.65, ties closest to 0.5.',
        weighting='Equal total weight per (record,class) for every model; scalers fit training rows only; no outer validation early stopping.',
        date_stress='Four acquisition-date outer folds, independently nested model selection on remaining dates.',
        promotion='Paired-record bootstrap 95% lower bound of (bACC+AUC)/2 gain >0; all three record outer-fold score gains >0; date macro score and overall record macro bACC do not decrease versus fixed round-two configuration.',
        supplemental='After selection is locked, evaluate synthetic gain 0.5/2 and offset +5000uV on every 32nd window per record; never use this stress result to choose a model. Actual continuous OOF streaming and new package only if promotion passes.',
        bootstrap=dict(seed=2026110303,replicates=10000,unit='record, not verified participant'),raw_support_samples=1500,fs=250,stride_samples=500,
        limitations='Same previously studied development data. Fresh grouping is a sensitivity analysis, not a fresh independent test. No participant/electrode/device assumptions. No new original-holdout evaluation.')
    path=Path('configs/round3/protocol.json')
    if path.exists():assert json.loads(path.read_text())==config,'Frozen protocol cannot change.'
    else:path.write_text(json.dumps(config,indent=2))
    archives={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path('submission_20261103.zip'),Path('submission_round2_20261103.zip'))}
    Path('reports/round3/prior_archives.json').write_text(json.dumps(archives,indent=2))
    print('Frozen',len(candidates),'candidates; development records',len(dev))


if __name__=='__main__':run()
