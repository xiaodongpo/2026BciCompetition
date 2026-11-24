"""训练最终R0：57条记录，LAG三种子各50轮，顺序增强树200轮，累计原型。"""
from pathlib import Path
import sys,json,time,hashlib,subprocess,traceback,argparse
SOURCE_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE_DIR));sys.path.insert(0,str(SOURCE_DIR/'online_adapt'))
from pipeline_paths import WORK_DIR as P, DEVICE, EPOCHS, TREE_ROUNDS
import numpy as np
import torch
from model_representation import EnergyFusionCNN
from proto_student_teacher import lag_targets,encode
from oa_stream import load_rec,seq,online_z,fit_tree
from oa_common import CACHE
ROOT=P/'runs/r0_all57_submission_20261107';SEEDS=[42,2024,3407]

def dump(p,v):
    t=Path(str(p)+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)

def save(p,v):
    t=Path(str(p)+'.tmp');torch.save(v,t);t.replace(p)

def ids():
    s=json.loads((P/'prepared_1_50hz/split.json').read_text())
    v=sorted(s['train_records']+s['validation_records']);assert len(set(v))==57
    return v

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cnn(seed):
    out=ROOT/('seed%d'%seed);out.mkdir(exist_ok=True,parents=True)
    if (out/'DONE.json').exists():return
    xs=[];ys=[]
    for rid in ids():
        q=load_rec(rid);target,keep,_=lag_targets(q)
        x=np.load(P/'prepared_1_50hz'/(rid+'_X.npy'));y=np.load(P/'prepared_1_50hz'/(rid+'_y.npy'))
        assert np.array_equal(y,q['y']);xs.append(x[keep]);ys.append(target[keep])
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    x=torch.from_numpy(np.concatenate(xs)).to(DEVICE);y=torch.from_numpy(np.concatenate(ys)).to(DEVICE)
    m=EnergyFusionCNN().to(DEVICE);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=1e-5);start=0;ck=out/'checkpoint_last.pth'
    if ck.exists():
        c=torch.load(ck,map_location='cpu');m.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);sch.load_state_dict(c['scheduler'])
        torch.set_rng_state(c['rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);start=c['epoch']
    history=[]
    for epoch in range(start+1,EPOCHS+1):
        m.train();total=0.
        for ix in torch.randperm(len(x),device=DEVICE).split(256):
            b=x[ix];opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(m(torch.cat([b[:,0],b.new_zeros((len(b),6))],1)),y[ix]);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();total+=float(loss)*len(ix)
        sch.step();history.append(dict(epoch=epoch,loss=total/len(x)))
        save(ck,dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),epoch=epoch,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),seed=seed,training_records=ids(),policy='LAG'))
        dump(out/'progress.json',dict(epoch=epoch,loss=total/len(x),training_windows=len(x)));dump(out/'history.json',history)
        print('CNN',seed,epoch,total/len(x),flush=True)
    save(out/'weights.pth',dict(model=m.state_dict(),epoch=EPOCHS,seed=seed,training_records=ids(),policy='LAG'))
    dump(out/'DONE.json',dict(status='COMPLETED',seed=seed,epochs=EPOCHS,records=len(ids())))

def tree_and_prior():
    out=ROOT/'model';out.mkdir(exist_ok=True)
    recs=[load_rec(r) for r in ids()];stack=np.concatenate([q['F'][q['v6']] for q in recs]);mu=stack.mean(0);sd=stack.std(0)+1e-8
    xx=[];yy=[];rr=[];rng=np.random.default_rng(45)
    for ri,q in enumerate(recs):
        _,_,lab=lag_targets(q)
        for order in ['ORIG','TASKFIRST','RANDOM','RANDOM']:
            ix=seq(q,order,rng);z=online_z(q['F'][ix],q['v6'][ix],mu,sd,60,[None,300.])
            keep=(lab[ix]>=0)&q['v6'][ix];xx.append(z[keep]);yy.append(lab[ix][keep]);rr.append(np.full(keep.sum(),ri))
    tree=fit_tree(np.concatenate(xx),np.concatenate(yy),np.concatenate(rr),TREE_ROUNDS,4,42)
    tree.booster_.save_model(str(out/'lightgbm.txt'));del xx,yy,rr,stack
    priors=[]
    for seed in SEEDS:
        model=EnergyFusionCNN().to(DEVICE).eval();c=torch.load(ROOT/('seed%d'%seed)/'weights.pth',map_location='cpu');model.load_state_dict(c['model'])
        xs=[];ys=[]
        for q in recs:
            emb,_=encode(model,np.load(CACHE/(q['rid']+'_X.npy')));keep=q['elapsed']>=45
            xs.append(emb[q['eval']][keep]);ys.append(q['y'][keep])
        x=np.concatenate(xs);y=np.concatenate(ys);a=x[y==0].mean(0);b=x[y==1].mean(0)
        var=.5*(x[y==0].var(0)+x[y==1].var(0))+1e-6;scale=np.sum(((b-a)/var)*(x-.5*(a+b)),axis=1).std()
        priors.append((a,b,var,scale));save(out/('energy_seed%d.pth'%seed),c)
        print('PRIOR',seed,flush=True)
    np.savez(out/'statistics.npz',feature_mean=mu,feature_sd=sd,m0=np.stack([p[0] for p in priors]),
        m1=np.stack([p[1] for p in priors]),variance=np.stack([p[2] for p in priors]),score_scale=np.array([p[3] for p in priors]))
    dump(out/'config.json',dict(model='R0_LAG_CUMULATIVE_PROTO',version='20261107',fs=250,channel='EEG',units='uV',
        seeds=SEEDS,epochs=EPOCHS,tree_rounds=TREE_ROUNDS,tree_weight=.5,alpha=.25,tau=.5,kappa=100,
        feature_prior=60,feature_halves=[None,300],center_prior=120,window_max_seconds=6,
        cnn_seconds=4,step_seconds=1,threshold=.5,training_records=ids(),training_records_count=len(ids()),
        official_training_target='LAG: center<15 previous phase, center15-45 omitted, >=45 official',
        prototype_training_selection='center>=45, official class',
        held_out_set_for_this_final_model=False,prior_results_are_previous_models_not_final57_test=True))

def main():
    ROOT.mkdir(exist_ok=True,parents=True);ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--pack',action='store_true');a=ap.parse_args()
    if a.seed:cnn(a.seed);return
    if a.pack:tree_and_prior();return
    dump(ROOT/'data_manifest.json',{rid:{'metadata_sha256':sha(P/'processing_current'/rid/'metadata.json'),
        'X_sha256':sha(P/'prepared_1_50hz'/(rid+'_X.npy')),'y_sha256':sha(P/'prepared_1_50hz'/(rid+'_y.npy'))} for rid in ids()})
    (ROOT/'train_r0_all57.py').write_bytes(Path(__file__).read_bytes())
    for seed in SEEDS:
        dump(ROOT/'progress.json',dict(status='TRAINING_CNN',seed=seed,epochs_each=EPOCHS))
        with (ROOT/('seed%d.log'%seed)).open('a') as f:subprocess.run([sys.executable,__file__,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,check=True)
    dump(ROOT/'progress.json',dict(status='TRAINING_TREE_AND_PRIORS'))
    tree_and_prior();dump(ROOT/'progress.json',dict(status='COMPLETED',records=len(ids()),cnn_epochs=len(SEEDS)*EPOCHS))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if '--seed' not in sys.argv:dump(ROOT/'progress.json',dict(status='FAILED',error=repr(e)))
        traceback.print_exc();raise
