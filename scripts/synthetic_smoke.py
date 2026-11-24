"""Small synthetic training/export/reload demo. Never a competition performance claim."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'routes/r0'))
from r0 import R0Decoder
from r0.model import EnergyFusionCNN
from r0.preprocess import transform
from r0.signal import extract_batch

def synthetic_signal(seconds=35,seed=20261123):
    rng=np.random.default_rng(seed);t=np.arange(seconds*250)/250
    return 40000+20*np.sin(2*np.pi*10*t)+9*np.sin(2*np.pi*6*t)+100*np.sin(2*np.pi*50*t)+rng.normal(0,3,len(t))

def train_bundle(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():raise ValueError('Use a fresh output directory')
    torch.set_num_threads(1);rng=np.random.default_rng(42)
    raw=40000+rng.normal(0,15,(24,1500)); y=np.arange(24)%2
    t=np.arange(1500)/250
    raw+=np.where(y[:,None]==0,15*np.sin(2*np.pi*10*t),15*np.sin(2*np.pi*6*t))
    _,wave,_,_,_,keep=transform(raw[:,-1000:,None].transpose(0,2,1).astype('float32'))
    assert keep.all()
    x=torch.from_numpy(wave.reshape(24,1,1000)); targets=torch.tensor(y)
    means0=[];means1=[];variances=[];losses=[]
    for seed in [42,2024,3407]:
        torch.manual_seed(seed);m=EnergyFusionCNN();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
        before=m.energy_filters.weight.detach().clone();m.train();opt.zero_grad()
        loss=torch.nn.functional.cross_entropy(m(x),targets);loss.backward();opt.step()
        assert torch.isfinite(loss) and not torch.equal(before,m.energy_filters.weight)
        m.eval()
        with torch.no_grad():emb=m.embedding(x).numpy()
        means0.append(emb[y==0].mean(0));means1.append(emb[y==1].mean(0));variances.append(np.maximum(.5*(emb[y==0].var(0)+emb[y==1].var(0)),1e-4))
        torch.save({'model':m.state_dict(),'synthetic':True,'seed':seed},out/f'energy_seed{seed}.pth')
        losses.append(float(loss.detach()))
    F,_=extract_batch(raw,np.ones(len(raw),bool));assert F.shape==(24,165)
    mu=F.mean(0).astype('float64');sd=np.maximum(F.std(0),1e-3).astype('float64')
    Z=np.tile((F-mu)/sd,(1,2))
    tree=lgb.LGBMClassifier(n_estimators=2,num_leaves=3,min_child_samples=2,verbosity=-1,n_jobs=1,random_state=42).fit(Z,y)
    tree.booster_.save_model(str(out/'lightgbm.txt'))
    np.savez(out/'statistics.npz',feature_mean=mu,feature_sd=sd,m0=np.array(means0),m1=np.array(means1),variance=np.array(variances),score_scale=np.ones(3))
    c=json.loads((ROOT/'routes/r0/training/recipe.json').read_text(encoding='utf8'))
    c.update(training_records=['synthetic'],training_records_count=1,epochs=1,tree_rounds=2,synthetic=True,acceptance_test_only=True)
    (out/'config.json').write_text(json.dumps(c,indent=2),encoding='utf8')
    decoder=R0Decoder(out);rows=decoder.push(synthetic_signal())
    assert len(rows)==32 and all(r['valid'] and 0<=r['score']<=1 for r in rows)
    return {'status':'PASS','synthetic':True,'cnn_seeds':3,'optimizer_steps_per_seed':1,'tree_rounds':2,'losses':losses,'reload_decisions':len(rows),'feature_dimensions':165,'competition_metrics_reproduced':False},rows

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=ROOT/'cache/synthetic_smoke');a=ap.parse_args()
    report,rows=train_bundle(a.out/'model')
    np.save(a.out/'raw_eeg_uv.npy',synthetic_signal())
    with (a.out/'predictions.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
if __name__=='__main__':main()
