"""Audit neural provenance, matched controls, and exact interrupted-training replay."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.r6_cnn import fit_run,conditions


def run():
    p=json.loads(Path('configs/round6/protocol.json').read_text());dev=set(p['development']);excluded=set(p['excluded_previous_holdout']);rows=[];control_checks=[]
    for outer in sorted(Path('experiments/round6/cnn').glob('outer*')):
        for folder in sorted(outer.iterdir()):
            r=json.loads((folder/'complete.json').read_text());inner=r['inner'];refit=r['refit'];valid=set(r['validation_records']);train=set(refit['train']);assert train==dev-valid and not(train|valid)&excluded
            assert not set(inner['train'])&set(inner['validation']);assert set(inner['train'])|set(inner['validation'])==train
            assert refit['actual_epochs']==inner['best_epoch'];assert refit['lr_replay']==inner['lr_replay']
            for stage in ('inner','refit'):
                ck=torch.load(folder/stage/'best.pt',map_location='cpu',weights_only=False);assert set(ck['train'])==set(r[stage]['train']) and set(ck['valid'])==set(r[stage]['validation'])
                assert hashlib.sha256((folder/stage/'best.pt').read_bytes()).hexdigest()==r[stage]['checkpoint_sha256']
                hist=pd.read_json(folder/stage/'history.jsonl',lines=True);assert hist.epoch.tolist()==list(range(1,r[stage]['actual_epochs']+1));assert np.isfinite(hist.training_objective).all()
            rows.append(dict(outer=outer.name,condition=folder.name,train=len(train),validation=len(valid),best_epoch=inner['best_epoch'],actual_epochs=inner['actual_epochs']))
        a=torch.load(outer/'legacy6_pure/inner/epoch6.pt',map_location='cpu',weights_only=False);b=torch.load(outer/'long150_pure/inner/epoch6.pt',map_location='cpu',weights_only=False)
        equal=all(torch.equal(a['model'][k],b['model'][k]) for k in a['model']);assert equal;control_checks.append(dict(outer=outer.name,first_six_epochs_bitwise_equal=True))
    assert len(rows)==18
    meta=pd.read_csv('cache/round6/windows.csv',dtype={'record':str});raw=np.load('cache/round6/raw.npy',mmap_mode='r');x=torch.empty((len(raw),1,1500),device='cuda')
    for a in range(0,len(raw),512):x[a:a+512,0]=torch.as_tensor(np.array(raw[a:a+512]),device='cuda')
    root=Path('cache/round6/resume_check');root.mkdir(exist_ok=True);c=conditions(dict(kind='hard',T_task=0,T_relax=0))[0]
    fit_run(x,meta,['002','003'],[],c,root/'fresh',20261104,refit_epochs=8,lr_replay=[.001]*8)
    resumed=root/'resumed';resumed.mkdir(exist_ok=True)
    if not (resumed/'last.pt').exists():shutil.copy2(root/'fresh/epoch6.pt',resumed/'last.pt')
    fit_run(x,meta,['002','003'],[],c,resumed,20261104,refit_epochs=8,lr_replay=[.001]*8)
    a=torch.load(root/'fresh/best.pt',map_location='cpu',weights_only=False);b=torch.load(resumed/'best.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    result=dict(neural_outer_models=18,neural_training_runs=36,all_splits_verified=True,all_refit_epochs_and_lr_schedules_inner_only=True,original_and_long_training_controls=control_checks,interrupted_epoch6_to8_replay_bitwise=True,resume_test_extra_epochs=10,rows=rows)
    Path('reports/round6/cnn_integrity_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
