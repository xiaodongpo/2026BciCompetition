import hashlib,json
from pathlib import Path
from scripts.round3_protocol import ranked

def run():
    for root in ('configs','cache','reports','results','experiments','checkpoints'):(Path(root)/'round4').mkdir(exist_ok=True)
    old=json.loads(Path('configs/round3/protocol.json').read_text());dev=old['development'];order=ranked(dev,'round4-20261104-outer-')
    candidates=[dict(id=f'w{tag}_C{C}',features=f'w{tag}',model='lr',C=C,width=width) for tag,width in [('0',0),('015',.15),('035',.35)] for C in (1,10)]
    p=dict(round=4,seed=20261104,development=dev,excluded_previous_holdout=old['excluded_previous_holdout'],outer_folds=[sorted(order[k::3]) for k in range(3)],inner_split_salt='round4-inner-20261104-',candidates=candidates,fixed_reference=dict(id='w0_C10',features='w0',model='lr',C=10,width=0,threshold=.46),
        selection='Inner OOF mean of record-macro bACC@0.46 and AUC; deterministic candidate order breaks ties.',threshold=.46,
        augmentation='For width w>0 use raw gains exp(-w),1,exp(w) on TRAINING records only; same labels. Per-copy weights are original record/class weights divided by 3; sum of effective weights unchanged. Scaler fits original training features only. Validation uses unchanged signal.',
        hypotheses='Small gain augmentation may improve generalization while retaining absolute power; fixing the established threshold avoids another inner threshold search.',
        date_stress='Nested leave-one-acquisition-date-out on the same 45 development records; dates are grouping variables only.',
        promotion='Record paired-bootstrap composite gain lower 95% bound >0; all 3 record outer-fold gains >0; date macro composite and record macro bACC nondecreasing.',
        supplemental='After lock: every 32nd window per record, gains 0.5/2 and offset +5000uV. These scores do not select models. No new package unless promotion passes; then continuous-stream and package verification required.',
        raw_support_samples=1500,fs=250,stride_samples=500,limitations='Previously studied development records; no new pristine holdout. Participants/electrodes/devices unknown. Record bootstrap is not participant-level inference.')
    path=Path('configs/round4/protocol.json')
    if path.exists():assert json.loads(path.read_text())==p
    else:path.write_text(json.dumps(p,indent=2))
    prior={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('submission_20261103.zip'),Path('submission_round2_20261103.zip')]}
    Path('reports/round4/prior_archives.json').write_text(json.dumps(prior,indent=2))
    print('Frozen',len(candidates),'configurations; threshold 0.46')

if __name__=='__main__':run()
