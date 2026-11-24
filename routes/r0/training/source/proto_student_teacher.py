"""仅在外层训练记录内部训练长历史教师，生成内层折外监督。

长期在线状态只存在于离线教师，绝不进入学生的部署路径。
不读取外层验证记录或全部57条记录的模型。
"""
from pathlib import Path
import json
import numpy as np
import torch
from pipeline_paths import DEVICE
from model_representation import EnergyFusionCNN
from proto_student_models import representation
from ps_oa_stream import load_rec,online_z,seq,center_cum,fit_tree


def logit(p):
    p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p))


def dump(path,obj):
    t=Path(str(path)+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False));t.replace(path)


def save(path,obj):
    t=Path(str(path)+'.tmp');torch.save(obj,t);t.replace(path)


def lag_targets(q):
    phase=np.r_[0,np.cumsum(np.diff(q['elapsed'])<0)]
    labs=[int(q['y'][phase==i][0]) for i in range(phase.max()+1)]
    prev=np.array([labs[p-1] if p>0 else -1 for p in phase])
    keep=((q['elapsed']<15)&(prev>=0))|(q['elapsed']>=45)
    target=np.where(q['elapsed']<15,prev,q['y']).astype(np.int64)
    stream=np.full(len(q['starts']),-1,np.int64);stream[q['eval'][keep]]=target[keep]
    return target,keep,stream


@torch.no_grad()
def encode(model,array):
    model.eval();emb=[];lp=[]
    for i in range(0,len(array),1024):
        x=torch.from_numpy(array[i:i+1024]).to(DEVICE);z=representation(model,x);a=model.classifier(z)
        emb.append(z.cpu().numpy());lp.append(a.softmax(-1)[:,1].cpu().numpy())
    return np.concatenate(emb).astype(np.float64),np.concatenate(lp)


def smooth(p,starts):
    out=np.empty(len(p))
    for i in range(len(p)):
        ix=[i]
        if i>=1 and starts[i-1]+250==starts[i]:
            ix.append(i-1)
            if i>=2 and starts[i-2]+250==starts[i-1]:ix.append(i-2)
        out[i]=p[ix].mean()
    return out


def proto_adjust(x,f,m0,m1,var,scale):
    n0=n1=100.;s0=100*m0.copy();s1=100*m1.copy();source=(m1-m0)/var;mid=(m0+m1)/2
    out=np.empty(len(f))
    for t in range(len(f)):
        a,b=s0/n0,s1/n1
        out[t]=((((b-a)/var)*(x[t]-.5*(a+b))).sum()-(source*(x[t]-mid)).sum())/scale
        if f[t]>.5:n1+=1;s1+=x[t]
        elif f[t]<-.5:n0+=1;s0+=x[t]
    return f+.25*out


def train_teacher(project,folder,train_ids,hold_ids,outer_val,seed=42):
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    assert set(train_ids).isdisjoint(hold_ids) and set(train_ids+hold_ids).isdisjoint(outer_val)
    metadata=dict(train_records=train_ids,held_records=hold_ids,outer_validation_excluded=outer_val,
                  cnn_epochs=50,cnn_seeds=[seed],tree_rounds=200,teacher_policy='LAG teacher only',
                  teacher_long_history_allowed_only_offline=True)
    dump(folder/'provenance.json',metadata)
    if (folder/'DONE.json').exists():return
    tr=[load_rec(r) for r in train_ids]
    xs=[];ys=[]
    cache=project/'runs/energy_stream_training_50epochs/stream_cache'
    for q in tr:
        target,keep,_=lag_targets(q)
        x=np.load(project/'prepared_1_50hz'/(q['rid']+'_X.npy'))
        assert np.array_equal(q['y'],np.load(project/'prepared_1_50hz'/(q['rid']+'_y.npy')))
        xs.append(x[keep]);ys.append(target[keep])
    x=torch.from_numpy(np.concatenate(xs)).to(DEVICE);y=torch.from_numpy(np.concatenate(ys)).to(DEVICE)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    model=EnergyFusionCNN().to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=1e-5)
    checkpoint=folder/'teacher_last.pth';start=0
    if checkpoint.exists():
        c=torch.load(checkpoint,map_location='cpu');model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);sch.load_state_dict(c['scheduler'])
        torch.set_rng_state(c['rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);start=c['epoch']
    for epoch in range(start+1,51):
        model.train();total=0.
        for ix in torch.randperm(len(x),device=DEVICE).split(256):
            xx=x[ix];packed=torch.cat([xx[:,0],xx.new_zeros((len(xx),6))],1)
            opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(packed),y[ix]);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss)*len(ix)
        sch.step();save(checkpoint,dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),epoch=epoch,
                                      rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),provenance=metadata))
        dump(folder/'progress.json',dict(status='TRAINING_TEACHER',epoch=epoch,train_windows=len(x)))
        print('TEACHER_EPOCH',epoch,total/len(x),flush=True)
    del x,y
    model.eval()
    stack=np.concatenate([q['F'][q['v6']] for q in tr]);mu=stack.mean(0);sd=stack.std(0)+1e-8
    xx=[];yy=[];rr=[];rng=np.random.default_rng(seed)
    for ri,q in enumerate(tr):
        _,_,targets=lag_targets(q)
        for order in ['ORIG','TASKFIRST','RANDOM','RANDOM']:
            idx=seq(q,order,rng);z=online_z(q['F'][idx],q['v6'][idx],mu,sd,60,[None,300.])
            keep=(targets[idx]>=0)&q['v6'][idx];xx.append(z[keep]);yy.append(targets[idx][keep]);rr.append(np.full(keep.sum(),ri))
    tree=fit_tree(np.concatenate(xx),np.concatenate(yy),np.concatenate(rr),200,4,seed)
    tree.booster_.save_model(str(folder/'teacher_tree.txt'));del xx,yy,rr
    zs=[];labels=[]
    for q in tr:
        emb,_=encode(model,np.load(cache/(q['rid']+'_X.npy')))
        keep=q['elapsed']>=45;zs.append(emb[q['eval']][keep]);labels.append(q['y'][keep])
    z=np.concatenate(zs);labels=np.concatenate(labels)
    m0=z[labels==0].mean(0);m1=z[labels==1].mean(0)
    var=.5*(z[labels==0].var(0)+z[labels==1].var(0))+1e-6
    scale=max(float((((m1-m0)/var)*(z-.5*(m0+m1))).sum(1).std()),1e-6)
    np.savez(folder/'teacher_prior.npz',mean=mu,sd=sd,m0=m0,m1=m1,var=var,scale=scale)
    for rid in hold_ids:
        q=load_rec(rid);emb,p=encode(model,np.load(cache/(rid+'_X.npy')))
        energy=center_cum(logit(smooth(p,q['starts'])),120)
        zz=online_z(q['F'],q['v6'],mu,sd,60,[None,300.])
        tl=tree.predict(zz,raw_score=True,num_threads=4)
        fused=np.where(q['v6'],.5*energy+.5*tl,energy)
        score=proto_adjust(emb,fused,m0,m1,var,scale)
        ix=q['eval'];agree=(energy[ix]*tl[ix]>0)&(score[ix]*energy[ix]>0)&q['v6'][ix]
        np.savez_compressed(folder/(rid+'_oof.npz'),logits=score[ix],agreement=agree,
                            original_y=q['y'],elapsed=q['elapsed'])
    dump(folder/'DONE.json',dict(status='PASS',**metadata));dump(folder/'progress.json',dict(status='COMPLETED',epoch=50))
