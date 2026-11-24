import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,json,time,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.signal import resample_poly
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import joblib
from third_party.biot import BIOTEncoder
from src.evaluation import metrics,record_metrics,choose_threshold
from scripts.train_classical import source_digest,update_tables
from scripts.train_neural import seed_all

def load_backbone(pretrained=True):
    backbone=BIOTEncoder(n_channels=18)
    if pretrained:
        state=torch.load('third_party/biot_pretrained.ckpt',map_location='cpu',weights_only=True)
        backbone.load_state_dict(state,strict=True)
    return backbone

def prepare_x(x):
    # resample_poly sees ONLY this already arrived 6 s buffer; never a future or earlier record sample.
    z=resample_poly(np.asarray(x,dtype=np.float32),4,5,axis=-1)
    z-=z.mean(-1,keepdims=True)
    z/=np.maximum(np.std(z,axis=-1,keepdims=True),1)
    return np.clip(z,-10,10).astype(np.float32)

def extract(pretrained=True):
    seed_all(42);torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    name='pretrained' if pretrained else 'random';path=Path(f'cache/biot_{name}.npy')
    if path.exists():return
    backbone=load_backbone(pretrained).to(device).eval();x=np.load('cache/raw_6s.npy',mmap_mode='r');out=[];t=time.perf_counter()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for a in range(0,len(x),128):
            z=torch.from_numpy(prepare_x(x[a:a+128]))[:,None,:].to(device)
            out.append(backbone(z).cpu().numpy())
            if a%8192==0:print(name,a,len(x),flush=True)
    np.save(path,np.concatenate(out))
    info=dict(seconds=time.perf_counter()-t,device=device,parameters=sum(p.numel() for p in backbone.parameters()),gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0,
              weight_sha256=hashlib.sha256(Path('third_party/biot_pretrained.ckpt').read_bytes()).hexdigest())
    Path(f'cache/biot_{name}_info.json').write_text(json.dumps(info,indent=2));print(info,flush=True)

def probe(pretrained=True):
    extract(pretrained);name='pretrained' if pretrained else 'random';X=np.load(f'cache/biot_{name}.npy')
    meta=pd.read_csv('cache/windows_6s.csv',dtype={'record':str});split=json.loads(Path('configs/splits.json').read_text());info=json.loads(Path(f'cache/biot_{name}_info.json').read_text())
    for C in (.01,1):
        c=dict(model='biot_probe',seconds=6,pretrained=pretrained,C=C);eid=f'biot_{name}_probe_C{C}';out=Path('experiments')/eid;out.mkdir(exist_ok=True)
        if (out/'summary.json').exists():continue
        pred=[];folds=[];start=time.perf_counter()
        for fold,valid in enumerate(split['folds']):
            tr=meta.record.isin(set(split['development'])-set(valid));va=meta.record.isin(valid)
            model=make_pipeline(StandardScaler(),LogisticRegression(C=C,class_weight='balanced',max_iter=2000,random_state=42))
            with threadpool_limits(limits=4):model.fit(X[tr],meta.y[tr]);p=model.predict_proba(X[va])[:,1]
            joblib.dump(model,out/f'fold{fold}.joblib');part=meta[va].copy();part['p']=p;part['fold']=fold;pred.append(part);folds.append(dict(fold=fold,**metrics(part.y,p)))
        save_summary(out,eid,c,pred,folds,start,info)

def save_summary(out,eid,c,pred,folds,start,info):
    oof=pd.concat(pred);oof.to_csv(out/'oof.csv',index=False);pd.DataFrame(folds).to_csv(out/'fold_metrics.csv',index=False)
    rm=record_metrics(oof);rm.to_csv(out/'record_metrics.csv',index=False);threshold=choose_threshold(oof.y,oof.p)
    summary=dict(id=eid,config=c,seed=42,source_sha256=source_digest(),split='configs/splits.json',**metrics(oof.y,oof.p),threshold=threshold,
        tuned_bacc=metrics(oof.y,oof.p,threshold)['bacc'],record_bacc_mean=float(rm.bacc.mean()),record_auc_mean=float(rm.auc.mean()),
        fold_bacc_std=float(np.std([m['bacc'] for m in folds])),fold_auc_std=float(np.std([m['auc'] for m in folds])),
        train_seconds=time.perf_counter()-start,gpu_peak_mb=info['gpu_peak_mb'],epochs=c.get('epochs',0),
        trainable_parameters=info.get('trainable_parameters',257),feature_extraction_seconds=info.get('seconds',0),checkpoint=str(out))
    (out/'config.json').write_text(json.dumps(c,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2));update_tables();print(json.dumps(summary),flush=True)

def finetune(mode):
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';split=json.loads(Path('configs/splits.json').read_text())
    X=prepare_x(np.load('cache/raw_6s.npy',mmap_mode='r'));meta=pd.read_csv('cache/windows_6s.csv',dtype={'record':str});y=meta.y.to_numpy()
    c=dict(model='biot_'+mode,seconds=6,epochs=3,lr=.0001,batch=64);eid='biot_'+mode+'_3ep';out=Path('experiments')/eid;out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():return
    (out/'config.json').write_text(json.dumps(c,indent=2));pred=[];folds=[];start=time.perf_counter();peaks=[]
    for fold,valid in enumerate(split['folds']):
        seed_all(42+fold);backbone=load_backbone().to(device)
        if mode=='partial':
            for p in backbone.parameters():p.requires_grad=False
            # Verified source module: final transformer layer plus classification head.
            for p in backbone.transformer.layers.layers[-1].parameters():p.requires_grad=True
        model=nn.Sequential(backbone,nn.Linear(256,1)).to(device)
        params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=c['lr'],weight_decay=.01)
        tr=np.flatnonzero(meta.record.isin(set(split['development'])-set(valid)));va=np.flatnonzero(meta.record.isin(valid))
        loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y[tr]==0).sum()/(y[tr]==1).sum()),device=device))
        last=out/f'fold{fold}_resume.pt';epoch0=0
        if last.exists():
            ck=torch.load(last,map_location=device,weights_only=False);model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);epoch0=ck['epoch']+1
            torch.set_rng_state(ck['rng'].cpu());np.random.set_state(ck['numpy_rng'])
            if device=='cuda':torch.cuda.set_rng_state_all(ck['cuda_rng'])
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        for epoch in range(epoch0,c['epochs']):
            model.train()
            if mode=='partial':backbone.eval()  # Frozen dropout disabled; tuned layer still receives gradients.
            order=np.random.permutation(tr);losses=[]
            for a in range(0,len(order),c['batch']):
                ix=order[a:a+c['batch']];z=torch.from_numpy(X[ix])[:,None,:].to(device);t=torch.from_numpy(y[ix].astype('float32')).to(device)
                optimizer.zero_grad(set_to_none=True);loss=loss_fn(model(z).squeeze(-1),t);loss.backward();torch.nn.utils.clip_grad_norm_(params,1);optimizer.step();losses.append(loss.item())
            ck=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [])
            torch.save(ck,last)
            log=dict(fold=fold,epoch=epoch+1,loss=float(np.mean(losses)))
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
            print(eid,log,flush=True)
        model.eval();p=[]
        with torch.no_grad():
            for a in range(0,len(va),128):p.extend(torch.sigmoid(model(torch.from_numpy(X[va[a:a+128]])[:,None,:].to(device))).squeeze(-1).cpu().numpy())
        part=meta.iloc[va].copy();part['p']=p;part['fold']=fold;pred.append(part);folds.append(dict(fold=fold,**metrics(part.y,p)))
        torch.save(dict(model=model.cpu().state_dict(),config=c),out/f'fold{fold}.pt');peaks.append(torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0)
    save_summary(out,eid,c,pred,folds,start,dict(gpu_peak_mb=max(peaks),trainable_parameters=sum(p.numel() for p in params)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['probe','random','partial','full'],default='probe');args=parser.parse_args()
    if args.mode in ('probe','random'):probe(args.mode=='probe')
    else:finetune(args.mode)
