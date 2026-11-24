import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,time,random
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from src.models import build_model
from src.evaluation import metrics,record_metrics,choose_threshold
from scripts.train_classical import source_digest,update_tables

class WindowsDataset(Dataset):
    def __init__(self,x,y,indices):self.x=x;self.y=y;self.indices=np.asarray(indices)
    def __len__(self):return len(self.indices)
    def __getitem__(self,k):
        i=self.indices[k];return torch.from_numpy(self.x[i].copy())[None,:],torch.tensor(self.y[i],dtype=torch.float32)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def predict(model,loader,device):
    model.eval();out=[]
    with torch.no_grad():
        for x,_ in loader:out.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.concatenate(out)

def run(c):
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    split=json.loads(Path('configs/splits.json').read_text());X=np.load('cache/raw_6s.npy',mmap_mode='r')
    meta=pd.read_csv('cache/windows_6s.csv',dtype={'record':str});y=meta.y.to_numpy()
    eid='_'.join(f'{k}-{v}' for k,v in c.items());out=Path('experiments')/eid;out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():return
    (out/'config.json').write_text(json.dumps(c,indent=2));start=time.perf_counter();pred=[];folds=[];peaks=[]
    with zipfile.ZipFile(out/'source_snapshot.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in [*Path('src').rglob('*.py'),*Path('scripts').rglob('*.py'),Path('configs/splits.json')]:archive.write(p)
    for fold,valid in enumerate(split['folds']):
        seed_all(c['seed']+fold)
        torch.backends.cudnn.deterministic=c.get('cudnn_deterministic',True)
        tr=np.flatnonzero(meta.record.isin(set(split['development'])-set(valid)));va=np.flatnonzero(meta.record.isin(valid))
        tl=DataLoader(WindowsDataset(X,y,tr),batch_size=c['batch'],shuffle=True,num_workers=0)
        vl=DataLoader(WindowsDataset(X,y,va),batch_size=256,num_workers=0)
        model=build_model(c).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=.01)
        loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y[tr]==0).sum()/max(1,(y[tr]==1).sum())),device=device))
        last=out/f'fold{fold}_resume.pt';epoch0=0
        if last.exists():
            ck=torch.load(last,map_location=device,weights_only=False);model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);epoch0=ck['epoch']+1
            torch.set_rng_state(ck['rng'].cpu())
            if device=='cuda':torch.cuda.set_rng_state_all(ck['cuda_rng'])
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        for epoch in range(epoch0,c['epochs']):
            model.train();losses=[]
            for x,t in tl:
                x,t=x.to(device),t.to(device);optimizer.zero_grad(set_to_none=True)
                loss=loss_fn(model(x),t);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(loss.item())
            p=predict(model,vl,device);m=metrics(y[va],p)
            log=dict(fold=fold,epoch=epoch+1,loss=float(np.mean(losses)),**m)
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
            ck=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [])
            torch.save(ck,last);print(eid,json.dumps(log),flush=True)
        # Fixed epoch count per candidate; avoid selecting epochs on each outer validation fold.
        p=predict(model,vl,device);part=meta.iloc[va].copy();part['p']=p;part['fold']=fold;pred.append(part);folds.append(dict(fold=fold,**metrics(y[va],p)))
        torch.save(dict(model=model.cpu().state_dict(),config=c),out/f'fold{fold}.pt')
        peaks.append(torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0)
    oof=pd.concat(pred);oof.to_csv(out/'oof.csv',index=False);pd.DataFrame(folds).to_csv(out/'fold_metrics.csv',index=False)
    rm=record_metrics(oof);rm.to_csv(out/'record_metrics.csv',index=False);threshold=choose_threshold(oof.y,oof.p)
    summary=dict(id=eid,config=c,seed=c['seed'],source_sha256=source_digest(),split='configs/splits.json',**metrics(oof.y,oof.p),threshold=threshold,
        tuned_bacc=metrics(oof.y,oof.p,threshold)['bacc'],record_bacc_mean=float(rm.bacc.mean()),record_auc_mean=float(rm.auc.mean()),
        fold_bacc_std=float(np.std([m['bacc'] for m in folds])),fold_auc_std=float(np.std([m['auc'] for m in folds])),
        train_seconds=time.perf_counter()-start,gpu_peak_mb=max(peaks),epochs=c['epochs'],trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),checkpoint=str(out),device=device)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));update_tables();print(json.dumps(summary),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['cnn','tcn'],required=True);parser.add_argument('--lr',type=float,default=.001);parser.add_argument('--epochs',type=int,default=6);parser.add_argument('--seed',type=int,default=42);args=parser.parse_args()
    config=dict(model=args.model,seconds=6,width=16,dropout=.25,lr=args.lr,batch=128,epochs=args.epochs,seed=args.seed)
    if args.model=='tcn':config.update(downsample=5,cudnn_deterministic=False)
    run(config)
