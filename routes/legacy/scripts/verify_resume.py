"""Check real saved model/optimizer/RNG restoration without creating a new experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.models import build_model

def run():
    torch.set_num_threads(1)
    p=next(Path('experiments').glob('model-cnn*/fold0_resume.pt'))
    c=json.loads((p.parent/'config.json').read_text());ck=torch.load(p,map_location='cpu',weights_only=False)
    split=json.loads(Path('configs/splits.json').read_text());meta=pd.read_csv('cache/windows_6s.csv',dtype={'record':str})
    ix=np.flatnonzero(meta.record.isin(set(split['development'])-set(split['folds'][0])))[:16]
    x=torch.from_numpy(np.load('cache/raw_6s.npy',mmap_mode='r')[ix].copy())[:,None,:]
    y=torch.from_numpy(meta.y.to_numpy()[ix].astype('float32'));results=[];steps=[]
    for repeat in range(2):
        # Optimizer state tensors are mutable; each independent recovery reads the checkpoint anew.
        ck=torch.load(p,map_location='cpu',weights_only=False)
        model=build_model(c);model.load_state_dict(ck['model'])
        opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=.01);opt.load_state_dict(ck['optimizer'])
        torch.set_rng_state(ck['rng'].cpu());model.train();opt.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y);loss.backward();opt.step()
        results.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        steps.append(float(next(iter(opt.state.values()))['step']))
    assert all(torch.equal(results[0][k],results[1][k]) for k in results[0])
    r=dict(checkpoint=str(p),saved_epoch=ck['epoch']+1,restored_next_update_bitwise_equal=True,optimizer_step_after_resume=steps,
           scope='Real checkpoint restoration and identical next CPU update on fold0 training records only; model discarded, no new model-selection experiment')
    Path('reports/checkpoint_resume_verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

if __name__=='__main__':run()
