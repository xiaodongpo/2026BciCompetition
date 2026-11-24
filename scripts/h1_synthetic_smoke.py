"""Train a small H1-shaped synthetic fixture and check the real finals inference interface."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'routes/h1'),str(ROOT/'routes/h1/finals/submission')]
from fin.net import HybridNet,wave_input,bandpass_notch
from fin.features import extract_multi
from fin.filtering import causal_lowpass
from src.models.hybrid import HybridNet as TrainingNet
import solution

def build(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():raise ValueError('Use a fresh output directory')
    torch.set_num_threads(1);rng=np.random.default_rng(23);y=np.arange(24)%2;t=np.arange(1500)/250
    x=40000+20*np.sin(2*np.pi*np.where(y[:,None]==0,10,6)*t)+rng.normal(0,3,(24,1500))
    F=extract_multi(x,[1500,500]);assert F.shape==(24,252)
    mu=F.mean(0);sd=F.std(0)+1e-6;Z=np.clip((F-mu)/sd,-8,8).astype('float32')
    members=[]
    for i,view in enumerate(['rank','abs']):
        ff=F if view=='rank' else extract_multi(bandpass_notch(x-x[:,:1]),[1500,500])
        amu=ff.mean(0);asd=ff.std(0)+1e-6
        inp=(np.sort(ff,axis=0)[:,None,:]<ff[None,:,:]).mean(0) if view=='rank' else .5+.5*np.tanh((ff-amu)/asd)
        tree=lgb.LGBMClassifier(n_estimators=2,num_leaves=3,min_child_samples=2,n_jobs=1,verbosity=-1,random_state=42).fit(inp,y)
        name=f'm{i}_lgb_0.txt';tree.booster_.save_model(str(out/name));m=dict(view=view,weight=[.2,.1][i],models=[name],lr=None)
        if view=='abs':
            scaler=StandardScaler().fit(inp);lr=LogisticRegression(C=.01,max_iter=100).fit(scaler.transform(inp),y)
            m.update(mu=amu.tolist(),sd=asd.tolist(),bpn=True,lr=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0])))
        members.append(m)
    losses=[]
    for i,wave in enumerate(['base','bpn'],start=2):
        torch.manual_seed(i);net=TrainingNet(252);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
        net.train();opt.zero_grad();w=torch.from_numpy(wave_input(x,wave));zz=torch.from_numpy(Z)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(net(w,zz),torch.tensor(y,dtype=torch.float32));loss.backward();opt.step()
        assert torch.isfinite(loss);losses.append(float(loss.detach()));net.eval();inference=HybridNet(252).eval();inference.load_state_dict(net.state_dict(),strict=True)
        with torch.no_grad():torch.testing.assert_close(net(w,zz),inference(w,zz))
        name=f'm{i}_seed0.pt';torch.save(net.state_dict(),out/name);members.append(dict(view='hybrid',weight=.35,wave=wave,nets=[name]))
    cfg=dict(windows=[1500,500],mu=mu.tolist(),sd=sd.tolist(),min_ref=20,min_len=750,fallback=.5,members=members,trained_on=['synthetic'],recipe='SYNTHETIC_H1_SCHEMA')
    (out/'config.json').write_text(json.dumps(cfg),encoding='utf8');return losses

def check(model):
    torch.set_num_threads(1);os.environ['FIN_MODEL_DIR']=str(model);solution._MODEL=None
    rng=np.random.default_rng(91);t=np.arange(7500)/250
    ref=np.array([40000+20*np.sin(2*np.pi*(7+j/4)*t)+rng.normal(0,3,len(t)) for j in range(10)])
    windows=np.array([ref[j,:1500]+rng.normal(0,1,1500) for j in range(4)])
    digest=lambda:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in model.iterdir() if p.is_file()}
    before=digest();original=ref.copy()
    scores=[solution.predict(w,ref) for w in windows]
    assert all(np.isfinite(p) and 0<=p<=1 for p in scores)
    assert scores==[solution.predict(w,ref) for w in windows[::-1]][::-1]
    solution.predict(windows[0],np.roll(ref,30,axis=1))
    assert solution.predict(windows[1],ref)==scores[1]
    assert abs(solution.predict(windows[1],ref[::-1].copy())-scores[1])<1e-6
    assert solution.predict(np.full(1500,.024),ref)==.5
    assert solution.predict(np.full(1500,403333.),ref)==.5
    badref=np.full_like(ref,.024);assert 0<=solution.predict(windows[0],badref)<=1
    gap=windows[0].copy();gap[:400]=.024;assert 0<=solution.predict(gap,ref)<=1
    x=np.r_[windows[0],windows[1]];xx=x.copy();xx[1800:]+=1e6
    assert np.array_equal(causal_lowpass(x)[:1800],causal_lowpass(xx)[:1800])
    np.testing.assert_allclose(causal_lowpass(3*x+1000),3*causal_lowpass(x)+1000,atol=1e-6)
    np.testing.assert_array_equal(ref,original);assert before==digest()
    return dict(status='PASS',synthetic_input=True,order_independence=True,reference_order_independence=True,cache_isolation=True,filter_future_invariance=True,filter_unit_dc_gain=True,neutral_bad_window=True,reference_fallback=True,model_files_unchanged=True,scores=scores,competition_metrics_reproduced=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--model',type=Path);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    model=a.model or a.out/'model';losses=None if a.model else build(model)
    report=check(model);report.update(training_losses=losses,existing_weights=bool(a.model),recipe=json.loads((model/'config.json').read_text())['recipe'])
    (a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
if __name__=='__main__':main()
