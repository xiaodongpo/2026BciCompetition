"""Offline transition curves from cross-fitted baseline predictions; no retuning."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
from src.round2_features import robust_signal

EDGES=[0,2,4,6,10,15,20,30,60,np.inf]
LABELS=['0-2','2-4','4-6','6-10','10-15','15-20','20-30','30-60','60+']


def interval(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if not len(a):return dict(n_records=0,mean=None,low=None,high=None)
    rng=np.random.default_rng(20261104);means=a[rng.integers(0,len(a),(5000,len(a)))].mean(1)
    lo,hi=np.quantile(means,[.025,.975]);return dict(n_records=len(a),mean=float(a.mean()),low=float(lo),high=float(hi))


def run():
    root=Path('results/round6');meta=pd.read_csv('cache/round6/windows.csv',dtype={'record':str});base=np.load('cache/round6/base.npy');ent=np.load('cache/round6/ordinal.npy')
    parts=[pd.read_csv(f'experiments/round6/lr/record/outer{k}/pure_reference.csv',dtype={'record':str}) for k in range(3)]
    oof=pd.concat(parts).sort_values(['record','end']).reset_index(drop=True);meta=meta.sort_values(['record','end']).reset_index(drop=True)
    assert oof[['record','end','y']].equals(meta[['record','end','y']]);f=meta.copy();f['p']=oof.p;f['prediction']=oof.prediction;f['error']=(f.prediction!=f.y).astype(float)
    f['bin']=pd.cut(f.t,EDGES,right=False,labels=LABELS).astype(str)
    names=dict(log_delta=base[:,0],log_theta=base[:,1],log_alpha=base[:,2],log_beta=np.log10(10.**base[:,3]+10.**base[:,4]),rel_delta=10.**base[:,8],rel_theta=10.**base[:,9],rel_alpha=10.**base[:,10],rel_beta=10.**base[:,11]+10.**base[:,12],log_theta_alpha=base[:,1]-base[:,2],log_beta_alpha=np.log10(10.**base[:,3]+10.**base[:,4])-base[:,2],log_variance=base[:,38],hjorth_mobility=base[:,40],hjorth_complexity=base[:,41],spectral_entropy=base[:,44],pe_m3_d4=ent[:,4],wpe_m3_d4=ent[:,5],pe_m4_d4=ent[:,14],wpe_m4_d4=ent[:,15],temporal_power_variability=base[:,59])
    for name,value in names.items():f[name]=value
    f.to_csv('cache/round6/baseline_transition_windows.csv',index=False)
    values=['p','error','current_fraction',*names]
    rows=[];record_rows=[]
    for scope,mask in [('all',np.ones(len(f),bool)),('pure',f.pure.to_numpy())]:
        subset=f[mask & f.valid_switch].copy()
        trial=subset.groupby(['record','trial_id','direction','bin'],observed=True)[values].mean().reset_index()
        record=trial.groupby(['record','direction','bin'],observed=True)[values].mean().reset_index();record['scope']=scope;record_rows.append(record)
        for (direction,bin_name),group in record.groupby(['direction','bin'],observed=True):
            windows=subset[(subset.direction==direction)&(subset.bin==bin_name)]
            for variable in values:rows.append(dict(scope=scope,direction=direction,bin=bin_name,variable=variable,n_windows=len(windows),n_trials=windows.trial_id.nunique(),**interval(group[variable])))
    pd.DataFrame(rows).to_csv(root/'directional_bins.csv',index=False);pd.concat(record_rows).to_csv(root/'directional_bins_by_record.csv',index=False)
    # Paired early pure windows versus late windows, each direction independently.
    pure=f[f.pure & f.valid_switch].copy();pure['period']=np.select([(pure.t>=6)&(pure.t<20),pure.t>=60],['early6_20','stable60plus'],default='omit');pure=pure[pure.period!='omit']
    trial=pure.groupby(['record','trial_id','direction','period'])[values].mean().reset_index();rec=trial.groupby(['record','direction','period'])[values].mean().reset_index()
    contrasts=[];pairs=[]
    for direction,g in rec.groupby('direction'):
        for variable in values:
            wide=g.pivot(index='record',columns='period',values=variable).dropna();d=wide.early6_20-wide.stable60plus
            contrasts.append(dict(direction=direction,variable=variable,contrast='early6_20 minus stable60plus',**interval(d)))
            for rid,v in d.items():pairs.append(dict(record=rid,direction=direction,variable=variable,difference=float(v)))
    pairframe=pd.DataFrame(pairs);pairframe.to_csv(root/'early_stable_paired_records.csv',index=False)
    for variable,g in pairframe.groupby('variable'):
        wide=g.pivot(index='record',columns='direction',values='difference').dropna();d=wide.relax_to_task-wide.task_to_relax
        contrasts.append(dict(direction='directional_asymmetry',variable=variable,contrast='(task early-stable) minus (relax early-stable)',**interval(d)))
    pd.DataFrame(contrasts).to_csv(root/'early_stable_contrasts.csv',index=False)
    # Re-anchor prior 30s and following 60s to each true transition, for offline plots.
    inventory=pd.read_csv('reports/round6/transition_inventory_all57.csv',dtype={'record':str});info=pd.read_csv('reports/round6/rechecked_records.csv',dtype={'record':str}).set_index('record')
    curves=[]
    for t in inventory[inventory.development & inventory.valid_switch].itertuples(index=False):
        g=f[f.record==t.record];start_ns=pd.Timestamp(t.event_start).value-pd.Timestamp(info.loc[t.record,'origin']).value
        relative=((g.end.to_numpy()-1)*4_000_000-start_ns)/1e9;keep=(relative>=-30)&(relative<60)
        part=g[keep][['record',*values]].copy();part['transition_id']=t.trial_id;part['direction']=t.direction;part['relative_bin']=(np.floor(relative[keep]/2)*2).astype(int);curves.append(part)
    curve=pd.concat(curves);trial=curve.groupby(['record','transition_id','direction','relative_bin'])[values].mean().reset_index();rec=trial.groupby(['record','direction','relative_bin'])[values].mean().reset_index();curve_rows=[]
    for (direction,b),g in rec.groupby(['direction','relative_bin']):
        for variable in ['p','error','log_theta','log_alpha','log_beta','pe_m3_d4']:curve_rows.append(dict(direction=direction,relative_bin=int(b),variable=variable,**interval(g[variable])))
    pd.DataFrame(curve_rows).to_csv(root/'aligned_transition_curves.csv',index=False)
    # PSD is an offline aggregate. The feature computation itself uses trailing windows only.
    raw=np.load('cache/round6/raw.npy',mmap_mode='r');spectra=[];eligible=np.flatnonzero(f.valid_switch & ((f.t<30)|(f.t>=60)))
    for a in range(0,len(eligible),128):
        z,_=robust_signal(raw[eligible[a:a+128]]);freq,power=welch(z,fs=250,nperseg=500,noverlap=250,axis=-1)
        spectra.append(np.log10(power[:,(freq>=1)&(freq<=45)]+1e-12))
    psd=np.concatenate(spectra);np.savez_compressed('cache/round6/diagnostic_psd.npz',row_indices=eligible,frequencies=freq[(freq>=1)&(freq<=45)],log_psd=psd)
    bins=f.iloc[eligible][['record','trial_id','direction','bin']].reset_index(drop=True);psd_df=pd.concat([bins,pd.DataFrame(psd)],axis=1);pc=list(range(psd.shape[1]));trial=psd_df.groupby(['record','trial_id','direction','bin'])[pc].mean().reset_index();rec=trial.groupby(['record','direction','bin'])[pc].mean().reset_index();rec.groupby(['direction','bin'])[pc].mean().to_csv(root/'directional_log_psd.csv')
    requested=[]
    for state in ('task','relax'):
        for T in (0,2,4,6,10,15,20,30):
            mask=f.valid_switch & (f.y==int(state=='task')) & (f.t<T)
            requested.append(dict(state=state,T=T,full_training_removed=int(mask.sum()),pure_training_removed=int((mask&f.pure).sum())))
    pd.DataFrame(requested).to_csv(root/'requested_exclusion_counts.csv',index=False)
    selected=[r for r in contrasts if r['variable']=='error'];Path('reports/round6/asymmetry_summary.json').write_text(json.dumps(selected,indent=2));print(json.dumps(selected,indent=2),flush=True)


if __name__=='__main__':run()
