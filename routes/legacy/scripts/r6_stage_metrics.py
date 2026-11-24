"""Stage-resolved ablations and offline within-trial latent distances."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.transition_evaluation import summary
from scripts.r6_diagnostics import interval,EDGES,LABELS


def run():
    rows=[];combined=[]
    paths=[*Path('results/round6').glob('lr_record_*_oof.csv'),*Path('results/round6').glob('cnn_*_oof.csv')]
    for path in paths:
        name=path.stem.removesuffix('_oof');f=pd.read_csv(path,dtype={'record':str});f=f[f.valid_switch].copy()
        f['correct']=(f.prediction==f.y).astype(float);f['period']=pd.cut(f.t,[0,6,20,60,np.inf],labels=['0-6','6-20','20-60','60+'],right=False).astype(str)
        for scope,mask in [('all',np.ones(len(f),bool)),('pure',f.pure.to_numpy())]:
            sub=f[mask];trial=sub.groupby(['record','trial_id','direction','period'])[['correct','p']].mean().reset_index();record=trial.groupby(['record','direction','period'])[['correct','p']].mean().reset_index()
            for (direction,period),g in record.groupby(['direction','period']):
                source=sub[(sub.direction==direction)&(sub.period==period)]
                rows.append(dict(model=name,scope=scope,direction=direction,period=period,metric='task_recall' if direction=='relax_to_task' else 'relax_specificity',n_windows=len(source),n_trials=source.trial_id.nunique(),mean_p=float(g.p.mean()),**interval(g.correct)))
        for period,g in f.groupby('period'):
            m,_=summary(g,g.p.to_numpy(),float(g.threshold.iloc[0]));combined.append(dict(model=name,period=period,**m))
    pd.DataFrame(rows).to_csv('results/round6/phase_metrics.csv',index=False);pd.DataFrame(combined).drop(columns='pooled').to_csv('results/round6/phase_combined_class_metrics.csv',index=False)
    s=json.loads(Path('results/round6/summary.json').read_text());latent_rows=[]
    for k,name in enumerate(s['cnn']['selected_conditions']):
        frame=pd.read_csv(f'experiments/round6/cnn/outer{k}/{name}/outer_predictions.csv',dtype={'record':str});h=np.load(f'cache/round6/latent/outer{k}_{name}.npy');assert len(frame)==len(h)
        for trial,g in frame[frame.valid_switch].groupby('trial_id'):
            late=g[g.t>=60].index.to_numpy()
            if len(late)<5:continue
            centroid=h[late].mean(0);distance=np.sqrt(np.mean((h[g.index]-centroid)**2,axis=1))
            part=g[['record','trial_id','direction','t']].copy();part['distance_to_late_centroid']=distance;latent_rows.append(part)
    latent=pd.concat(latent_rows);latent['bin']=pd.cut(latent.t,EDGES,labels=LABELS,right=False).astype(str)
    latent.to_csv('cache/round6/latent_diagnostics.csv',index=False)
    trial=latent.groupby(['record','trial_id','direction','bin']).distance_to_late_centroid.mean().reset_index();rec=trial.groupby(['record','direction','bin']).distance_to_late_centroid.mean().reset_index();out=[]
    for (direction,bin_name),g in rec.groupby(['direction','bin']):out.append(dict(direction=direction,bin=bin_name,**interval(g.distance_to_late_centroid)))
    pd.DataFrame(out).to_csv('results/round6/latent_distance_summary.csv',index=False)
    print('Wrote stage metrics for',len(paths),'procedures; latent distances use future stable centroids OFFLINE ONLY.',flush=True)


if __name__=='__main__':run()
