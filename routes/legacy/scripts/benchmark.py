import json,time,warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import joblib
from src.features import extract_features
from src.models import build_model
from scripts.train_biot import load_backbone,prepare_x
from scripts.train_neural import seed_all
from scripts.train_classical import update_tables

def run():
    torch.set_num_threads(1);warnings.filterwarnings('ignore',message='A window was not provided')
    x=np.load('cache/raw_6s.npy',mmap_mode='r')[0].copy()
    for p in Path('experiments').glob('*/summary.json'):
        s=json.loads(p.read_text());c=s['config'];name=c['model'];out=p.parent
        if name in ('lr','rbf','extra','hgb'):
            model=joblib.load(out/'fold0.joblib');z=x[-int(c['seconds']*250):]
            def predict():return model.predict_proba(extract_features(z,mode=c['features']))[0,1]
            if name in ('lr','rbf'):s['trainable_parameters']=int(model[-1].coef_.size+model[-1].intercept_.size)
        elif name in ('cnn','tcn'):
            ck=torch.load(out/'fold0.pt',map_location='cpu',weights_only=True);model=build_model(c);model.load_state_dict(ck['model']);model.eval()
            def predict():
                with torch.no_grad():return torch.sigmoid(model(torch.from_numpy(x)[None,None,:])).item()
        elif name=='biot_probe':
            seed_all(42);backbone=load_backbone(c['pretrained']).eval();model=joblib.load(out/'fold0.joblib')
            def predict():
                with torch.no_grad():f=backbone(torch.from_numpy(prepare_x(x[None,:]))[:,None,:]).numpy()
                return model.predict_proba(f)[0,1]
        elif name in ('biot_full','biot_partial'):
            model=nn.Sequential(load_backbone(),nn.Linear(256,1));ck=torch.load(out/'fold0.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model']);model.eval()
            def predict():
                with torch.no_grad():return torch.sigmoid(model(torch.from_numpy(prepare_x(x[None,:]))[:,None,:])).item()
        else:continue
        with threadpool_limits(limits=1):
            for _ in range(3):assert np.isfinite(predict())
            times=[]
            for _ in range(30):
                t=time.perf_counter();predict();times.append((time.perf_counter()-t)*1000)
        s['cpu_inference_ms_median']=float(np.median(times));s['cpu_inference_ms_p95']=float(np.quantile(times,.95));s['latency_protocol']='fold0, batch1, CPU 1 thread, 3 warmups + 30 runs, complete window preprocessing'
        p.write_text(json.dumps(s,indent=2));print(s['id'],s['cpu_inference_ms_median'],flush=True)
    update_tables()
if __name__=='__main__':run()
