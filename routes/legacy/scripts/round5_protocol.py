"""Freeze the ordinal-complexity question before observing round-five metrics."""
import hashlib, json
from pathlib import Path
from scripts.round3_protocol import ranked


def run():
    for root in ('configs','cache','reports','results','experiments','checkpoints'):
        (Path(root)/'round5').mkdir(exist_ok=True)
    old=json.loads(Path('configs/round4/protocol.json').read_text());dev=old['development']
    order=ranked(dev,'round5-20261104-outer-')
    candidates=[dict(id=f'{kind}_C{C}',features=kind,model='lr',C=C)
                for kind in ('base','ordinal','fusion') for C in ((1,10) if kind=='base' else (.1,1,10))]
    p=dict(round=5,seed=20261104,development=dev,excluded_previous_holdout=old['excluded_previous_holdout'],
        outer_folds=[sorted(order[k::3]) for k in range(3)],inner_split_salt='round5-inner-20261104-',
        candidates=candidates,fixed_reference=dict(id='base_C10',features='base',model='lr',C=10,threshold=.46),
        hypothesis='Permutation and variance-weighted permutation entropies may complement the established spectral/covariance features.',
        representations=dict(base=125,ordinal=20,fusion=145),
        ordinal='Unchanged within-window 1-45Hz FIR and robust clipping; embedding orders 3/4, delays 1/2/4/8/16 at 250Hz; PE and variance-weighted PE normalized by log(m!). Exact-tie vectors excluded, all-tied returns zero. No fitted feature parameters.',
        selection='3 inner OOF folds, record-macro (bACC@0.46+AUC)/2; deterministic candidate order breaks ties.',
        threshold=.46,weighting='Same equal total weight per record/class as round two; scaler fitted on training rows only.',
        date_stress='Nested leave-one-acquisition-date-out; dates are grouping variables only.',
        promotion=old['promotion'],raw_support_samples=1500,fs=250,stride_samples=500,
        supplemental='After lock: verify all saved representation adapters; paired time-reversal sensitivity on every 32nd dev window, no selection from this diagnostic. New package and pure-phase continuous-stream comparison only if promotion passes.',
        limitations=old['limitations'])
    path=Path('configs/round5/protocol.json')
    if path.exists():assert json.loads(path.read_text())==p
    else:path.write_text(json.dumps(p,indent=2))
    archive_path=Path('reports/round5/prior_archives.json')
    prior={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('submission_20261103.zip'),Path('submission_round2_20261103.zip')]}
    if archive_path.exists():assert json.loads(archive_path.read_text())==prior
    else:archive_path.write_text(json.dumps(prior,indent=2))
    print('Frozen',len(candidates),'configurations; 207 planned research fits; threshold 0.46')


if __name__=='__main__':run()
