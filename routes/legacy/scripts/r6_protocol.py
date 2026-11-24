import json,hashlib
from pathlib import Path
from scripts.round3_protocol import ranked


def axes():
    out=[dict(id='pure_reference',kind='pure',T_task=0,T_relax=0),dict(id='full_hard',kind='hard',T_task=0,T_relax=0)]
    for state in ('task','relax'):
        for T in (6,15,30):out.append(dict(id=f'drop_{state}_{T}',kind='drop',T_task=T if state=='task' else 0,T_relax=T if state=='relax' else 0))
    return out


def additions(T_task,T_relax):
    out=[]
    for tag,a,b in [('independent',T_task,T_relax),('same15',15,15),('task15_relax30',15,30),('task30_relax15',30,15)]:out.append(dict(id='drop_'+tag,kind='drop',T_task=a,T_relax=b))
    for tag,a,b,ma,mb in [('task15',15,0,.25,.25),('relax15',0,15,.25,.25),('same15',15,15,.25,.25),('independent',T_task,T_relax,.25,.25),('task_low',15,15,.25,.5),('relax_low',15,15,.5,.25)]:out.append(dict(id='weight_'+tag,kind='weight',T_task=a,T_relax=b,min_task=ma,min_relax=mb))
    for tag,a,b in [('task15',15,0),('relax15',0,15),('same15',15,15),('independent',T_task,T_relax)]:out.append(dict(id='soft_'+tag,kind='soft',T_task=a,T_relax=b))
    return out


def run():
    for root in ('configs','reports','results','experiments','checkpoints','cache'):(Path(root)/'round6').mkdir(exist_ok=True)
    old=json.loads(Path('configs/round5/protocol.json').read_text());order=ranked(old['development'],'round6-record-20261104-')
    p=dict(round=6,seed=20261104,development=old['development'],excluded_previous_holdout=old['excluded_previous_holdout'],outer_folds=[sorted(order[k::3]) for k in range(3)],inner_salt='round6-inner-20261104-',inner_folds=3,
        primary='All 2-second-grid windows with 1500 valid raw samples and official endpoint label. Do not exclude mixed or early windows from evaluation.',
        training_scope='45 development records. Event timing only in offline labels/weights/diagnostics. First phase is not a known transition.',
        lr_axes=axes(),lr_joint_rule='Choose T_task and T_relax separately among 0/6/15/30 by three-fold inner full-validation score; then evaluate 14 predeclared joint/weight/soft candidates on the same inner folds. No outer metrics affect this process.',lr_addition_template=additions(15,30),
        lr=dict(features='unchanged round2 filterbank 125',C=10,threshold=.46,selection='record macro (bACC+AUC)/2',record_class_weights=True),
        neural=dict(model='existing SmallCNN width16 dropout0.25 raw1500',max_epochs=150,early_stop_patience=25,learning_rate=.001,weight_decay=.01,batch=128,clip_norm=1.,scheduler='ReduceLROnPlateau factor0.5 patience8 min_lr1e-5; hard validation BCE, inner records only',
        conditions=['legacy6_pure','long150_pure','long150_full','scheduled150_full','scheduled150_transition','scheduled150_curriculum'],
        original_control='legacy6 and long150_pure retain pure training windows, original global pos_weight and constant LR. Only training duration / inner checkpoint selection differs.',
        other_control='long150_full versus scheduled150_full isolates scheduler; use identical global pos_weight and train rows. transition/curriculum add record-class-balanced supervision, disclosed as a strategy bundle.',
        epoch_selection='One fixed whole-record inner validation split (one third of outer-training records), all valid labeled windows retained. Select best record composite; early stop on no score improvement. Refit outer training for selected epoch with fixed LR schedule replay, never monitor outer validation.',
        curriculum='stable first 15 epochs (T_task=T_relax=30), linear introduction epochs16-30, full supervision thereafter; best epoch and early stopping eligible only from epoch31.',
        checkpoint='Save best and last per epoch, optimizer/scheduler/RNG, training and validation loss and full record bACC/AUC. Save epoch6 snapshot for long-run control.',threshold=.5),
        evaluation='All fixed ablations reported on identical outer records. LR selected procedure has record/date nested validation. Neural conditions use record outer folds and internal epoch tuning; no date-generalization claim for neural models without subsequent date tests.',
        promotion=old['promotion'],neural_promotion='Must first pass record bootstrap, all record folds and bACC gate against fixed LR; only then run nested date confirmation and full-package verification. Otherwise research only.',
        diagnostics='Baseline cross-fitted predictions; bins 0/2/4/6/10/15/20/30/60/inf, initial phases separate, pure/mixed separate; trial then record averaging; paired record bootstrap for directional early-versus-stable differences. No individual physiological transition time forced.',
        future='No transition-aware inference or external event input. No paid resources. Preserve old archives.',limitations='Previously used development data; participant/electrode/device unavailable; no untouched independent evaluation.')
    path=Path('configs/round6/protocol.json')
    if path.exists():assert json.loads(path.read_text())==p
    else:path.write_text(json.dumps(p,indent=2))
    prior={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ('submission_20261103.zip','submission_round2_20261103.zip')}
    target=Path('reports/round6/prior_archives.json')
    if target.exists():assert json.loads(target.read_text())==prior
    else:target.write_text(json.dumps(prior,indent=2))
    print('Frozen: 8 axis/control + 14 joint/weight/soft policies; 6 CNN conditions, max 150 epochs.')


if __name__=='__main__':run()
