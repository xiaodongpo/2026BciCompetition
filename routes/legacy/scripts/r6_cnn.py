"""Nested record-only CNN budget/supervision ablations with complete validation.

Epoch selection uses one inner record holdout. Outer refits replay the chosen
inner learning-rate schedule and never inspect outer labels while fitting.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,time,random,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from src.models import SmallCNN
from src.transition_training import policy,sample_summary
from src.transition_evaluation import summary
from scripts.round3_protocol import ranked


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def predict(model,x,indices,latent=False):
    model.eval();probs=[];representations=[]
    from src.models import normalize_window
    with torch.no_grad():
        for ids in torch.split(indices,512):
            z=x[ids];probs.append(torch.sigmoid(model(z)).cpu().numpy())
            if latent:
                normalized,scale=normalize_window(z);h=model.encoder(normalized).squeeze(-1)
                representations.append(torch.cat((h,scale/10),1).cpu().numpy())
    return np.concatenate(probs),(np.concatenate(representations) if latent else None)


def fit_run(x,meta,train,valid,c,out,seed,refit_epochs=None,lr_replay=None):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return json.loads((out/'complete.json').read_text())
    device=x.device;seed_all(seed);torch.set_num_threads(4)
    tr=np.flatnonzero(meta.record.isin(train));va=np.flatnonzero(meta.record.isin(valid)) if valid else np.array([],dtype=int)
    train_meta=meta.iloc[tr];val_meta=meta.iloc[va]
    train_ids=torch.as_tensor(tr,device=device);val_ids=torch.as_tensor(va,device=device)
    model=SmallCNN(16,.25).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=8,min_lr=1e-5)
    max_epochs=refit_epochs if refit_epochs is not None else (6 if c['condition']=='legacy6_pure' else 150)
    last=out/'last.pt';best=out/'best.pt';start_epoch=1;best_score=-1.;best_epoch=0;no_improvement=0;history=[]
    replay=lr_replay if lr_replay is not None else []
    if last.exists():
        ck=torch.load(last,map_location=device,weights_only=False);assert ck['config']==c and ck['train']==sorted(train) and ck['valid']==sorted(valid)
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);start_epoch=ck['epoch']+1
        best_score=ck['best_score'];best_epoch=ck['best_epoch'];no_improvement=ck['no_improvement'];history=ck['history']
        torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([r.cpu() for r in ck['cuda_rng']]);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
    def checkpoint(epoch):
        return dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,config=c,train=sorted(train),valid=sorted(valid),best_epoch=best_epoch,best_score=best_score,no_improvement=no_improvement,history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),seed=seed,source_manifest='experiments/round6/cnn_manifest.json')
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats();stopped=False
    epoch_range=range(start_epoch,max_epochs+1) if not(valid and no_improvement>=25 and c['condition']!='legacy6_pure') else []
    for epoch in epoch_range:
        if refit_epochs is not None:
            optimizer.param_groups[0]['lr']=replay[epoch-1]
        lr=float(optimizer.param_groups[0]['lr']);model.train()
        progress=0. if epoch<=15 else min(1.,(epoch-15)/15)
        q,w=policy(train_meta,c['policy'],progress=progress);active=w>0
        ids=train_ids[torch.as_tensor(active,device=device)];targets=torch.as_tensor(q[active],device=device,dtype=torch.float32);weights=torch.as_tensor(w[active],device=device,dtype=torch.float32)
        order=torch.randperm(len(ids),device=device);loss_sum=0.;weight_sum=0.
        pos_weight=torch.tensor(float(np.sum(train_meta.y.to_numpy()[active]==0)/np.sum(train_meta.y.to_numpy()[active]==1)),device=device)
        epoch_start=time.perf_counter()
        for batch in torch.split(order,128):
            optimizer.zero_grad(set_to_none=True);logits=model(x[ids[batch]])
            if c['record_weights']:
                losses=F.binary_cross_entropy_with_logits(logits,targets[batch],reduction='none');bw=weights[batch];loss=(losses*bw).sum()/bw.sum();mass=float(bw.sum())
            else:
                loss=F.binary_cross_entropy_with_logits(logits,targets[batch],pos_weight=pos_weight);mass=len(batch)
            if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_sum+=float(loss.detach())*mass;weight_sum+=mass
        row=dict(epoch=epoch,training_objective=loss_sum/weight_sum,lr=lr,retained_rows=int(active.sum()),curriculum_progress=progress if c['policy']['kind']=='curriculum' else None,seconds=time.perf_counter()-epoch_start)
        if valid:
            pv,_=predict(model,x,val_ids);m,_=summary(val_meta,pv,.5);row.update(val_loss=m['pooled']['loss'],val_macro_bacc=m['macro_bacc'],val_macro_auc=m['macro_auc'],val_score=m['score'])
            # A fixed, whole-record training subset reports hard-label fit over time.
            train_probe=train_ids[::8];pp,_=predict(model,x,train_probe);tm,_=summary(train_meta.iloc[::8],pp,.5);row.update(train_hard_loss=tm['pooled']['loss'],train_macro_bacc=tm['macro_bacc'],train_macro_auc=tm['macro_auc'])
            eligible=c['policy']['kind']!='curriculum' or epoch>=31
            if c['condition']=='legacy6_pure':eligible=epoch==6
            improved=eligible and m['score']>best_score+1e-6
            if improved:best_score=m['score'];best_epoch=epoch;no_improvement=0
            elif eligible:no_improvement+=1
            if c['scheduler']:scheduler.step(m['pooled']['loss'])
        else:
            improved=epoch==max_epochs;best_epoch=epoch;best_score=-1
        history.append(row)
        with (out/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        ck=checkpoint(epoch);torch.save(ck,out/'last.tmp.pt');(out/'last.tmp.pt').replace(last)
        if improved:torch.save(ck,best)
        if epoch==6:torch.save(ck,out/'epoch6.pt')
        if epoch==1 or epoch%10==0 or epoch==max_epochs:print(out.as_posix(),epoch,'train',round(row['training_objective'],4),'val',round(row.get('val_score',-1),4),'best',best_epoch,flush=True)
        if valid and c['condition']!='legacy6_pure' and no_improvement>=25:stopped=True;break
    ck=torch.load(best,map_location=device,weights_only=False);model.load_state_dict(ck['model'])
    result=dict(config=c,seed=seed,train=sorted(train),validation=sorted(valid),max_epochs=max_epochs,actual_epochs=history[-1]['epoch'],best_epoch=best_epoch,best_inner_score=best_score,early_stopped=stopped,seconds_this_invocation=time.perf_counter()-started,gpu_peak_mb=float(torch.cuda.max_memory_allocated()/2**20),parameters=sum(v.numel() for v in model.parameters()),lr_replay=[r['lr'] for r in history[:best_epoch]],checkpoint_sha256=hashlib.sha256(best.read_bytes()).hexdigest(),training_policy_counts=sample_summary(train_meta,c['policy']),history='history.jsonl')
    if valid:
        pv,_=predict(model,x,val_ids);f=val_meta.copy();f['p']=pv;f['threshold']=.5;f['prediction']=(pv>=.5).astype(int);f.to_csv(out/'best_inner_predictions.csv',index=False)
    (out/'complete.json').write_text(json.dumps(result,indent=2));return result


def conditions(strategy):
    pure=dict(kind='pure',T_task=0,T_relax=0);hard=dict(kind='hard',T_task=0,T_relax=0)
    return [dict(condition='legacy6_pure',policy=pure,scheduler=False,record_weights=False),dict(condition='long150_pure',policy=pure,scheduler=False,record_weights=False),dict(condition='long150_full',policy=hard,scheduler=False,record_weights=False),dict(condition='scheduled150_full',policy=hard,scheduler=True,record_weights=False),dict(condition='scheduled150_transition',policy=strategy,scheduler=True,record_weights=True),dict(condition='scheduled150_curriculum',policy=dict(kind='curriculum',T_task=30,T_relax=30),scheduler=True,record_weights=True)]


def run():
    p=json.loads(Path('configs/round6/protocol.json').read_text());meta=pd.read_csv('cache/round6/windows.csv',dtype={'record':str})
    assert set(meta.record)==set(p['development']) and not set(meta.record)&set(p['excluded_previous_holdout'])
    assert torch.cuda.is_available(),'A GPU is required for this planned local training budget.'
    sources=[Path(__file__),Path('src/models.py'),Path('src/transition_training.py'),Path('src/transition_evaluation.py'),Path('scripts/round3_protocol.py'),Path('configs/round6/protocol.json')]
    hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in sources};manifest=Path('experiments/round6/cnn_manifest.json')
    if manifest.exists():assert json.loads(manifest.read_text())['sources']==hashes
    else:
        manifest.write_text(json.dumps(dict(sources=hashes,data=json.loads(Path('cache/round6/prepared.json').read_text()),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),versions=dict(numpy=np.__version__,pandas=pd.__version__)),indent=2))
        with zipfile.ZipFile('experiments/round6/cnn_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for q in sources:z.write(q)
    if Path('experiments/round6/cnn_complete.json').exists():print('CNN already complete; no refits.');return
    raw=np.load('cache/round6/raw.npy',mmap_mode='r');x=torch.empty((len(raw),1,1500),device='cuda')
    for a in range(0,len(raw),512):x[a:a+512,0]=torch.as_tensor(np.array(raw[a:a+512]),device='cuda')
    start=time.perf_counter();all_results=[]
    for k,valid in enumerate(p['outer_folds']):
        train=sorted(set(p['development'])-set(valid));order=ranked(train,'round6-cnn-inner-20261104-');iv=sorted(order[::3]);it=sorted(set(train)-set(iv))
        lr_path=Path(f'experiments/round6/lr/record/outer{k}/inner/selection.json')
        assert lr_path.exists(),'Run LR inner search first.'
        strategy=json.loads(lr_path.read_text())['config']
        for c in conditions(strategy):
            out=Path(f"experiments/round6/cnn/outer{k}/{c['condition']}");out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():all_results.append(json.loads((out/'complete.json').read_text()));continue
            inner=fit_run(x,meta,it,iv,c,out/'inner',20261104+k)
            refit=fit_run(x,meta,train,[],c,out/'refit',20261104+k,refit_epochs=inner['best_epoch'],lr_replay=inner['lr_replay'])
            model=SmallCNN(16,.25).cuda();ck=torch.load(out/'refit/best.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model'])
            va=np.flatnonzero(meta.record.isin(valid));proba,h=predict(model,x,torch.as_tensor(va,device='cuda'),latent=True)
            frame=meta.iloc[va].copy();frame['p']=proba;frame['threshold']=.5;frame['prediction']=(proba>=.5).astype(int);frame['outer_fold']=k;frame['condition']=c['condition'];frame.to_csv(out/'outer_predictions.csv',index=False)
            # All derived latent EEG data is stored under cache, with row order in outer_predictions.
            latent=Path('cache/round6/latent')/f"outer{k}_{c['condition']}.npy";latent.parent.mkdir(exist_ok=True);np.save(latent,h)
            metrics,records=summary(frame,proba,.5);records['outer_fold']=k;records.to_csv(out/'outer_records.csv',index=False);pure=frame.pure.to_numpy();pure_metrics,_=summary(frame[pure],proba[pure],.5)
            result=dict(outer_fold=k,condition=c['condition'],config=c,inner=inner,refit=refit,full=metrics,pure=pure_metrics,validation_records=valid,latent_cache=str(latent));(out/'complete.json').write_text(json.dumps(result,indent=2));all_results.append(result)
            print('CNN outer',k,c['condition'],'epoch',inner['best_epoch'],'finished',flush=True)
    Path('experiments/round6/cnn_complete.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,conditions=6,outer_folds=3,inner_training_runs=18,refit_training_runs=18,results=all_results),indent=2));print('CNN ALL COMPLETE',flush=True)


if __name__=='__main__':run()
