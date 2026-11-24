import json, hashlib, platform, sys, subprocess, shutil
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import read_record

def run():
    for d in ('reports','results','cache/records','configs','experiments','checkpoints','tests','submission'):
        Path(d).mkdir(parents=True,exist_ok=True)
    records,trials_all,gaps_all,manifest=[],[],[],[]
    hashes={}; duplicates=[]
    for folder in sorted(Path(__import__('os').environ.get('BCI_DATA_ROOT', 'EEG_dataset')).iterdir()):
        if not folder.is_dir(): continue
        x,valid,labels,trials,gaps,events,info=read_record(folder)
        rid=folder.name
        info.update(valid_s=float(valid.sum()/250),gap_s=float((~valid).sum()/250),
                    relax_s=float((labels==0).sum()/250),task_s=float((labels==1).sum()/250),
                    abs_p99_uv=float(np.quantile(abs(x[valid]),.99)),median_uv=float(np.median(x[valid])),
                    max_abs_uv=float(np.max(abs(x[valid]))),over_500uv_s=float(((abs(x)>500)&valid).sum()/250),
                    exact_zero_s=float(((x==0)&valid).sum()/250),events=json.dumps(events.event.value_counts().to_dict()),
                    trials=len(trials))
        # Detect long constant runs, including quantized near-zero gaps that are not exactly zero.
        boundaries=np.r_[0,np.flatnonzero(np.diff(x)!=0)+1,len(x)]
        long=[(int(a),int(b)) for a,b in zip(boundaries[:-1],boundaries[1:]) if b-a>=250 and valid[a:b].all()]
        info['unannotated_flat_runs']=json.dumps(long)
        info['signal_sha256']=hashlib.sha256(x.tobytes()).hexdigest()
        if info['signal_sha256'] in hashes: duplicates.append([rid,hashes[info['signal_sha256']]])
        hashes[info['signal_sha256']]=rid
        np.savez(Path('cache/records')/(rid+'.npz'),x=x,valid=valid,labels=labels)
        records.append(info)
        trials_all.extend([dict(record=rid,**t) for t in trials])
        gaps_all.extend([dict(record=rid,**g) for g in gaps])
        for p in sorted(folder.iterdir()):
            if p.is_file(): manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        print(rid,info['duration_s'],info['relax_s'],info['task_s'],info['gap_s'],flush=True)
    df=pd.DataFrame(records); tf=pd.DataFrame(trials_all)
    df.to_csv('reports/records.csv',index=False); tf.to_csv('reports/trials.csv',index=False)
    pd.DataFrame(gaps_all).to_csv('reports/gaps.csv',index=False)
    pd.DataFrame(manifest).to_csv('reports/original_manifest.csv',index=False)
    # Exact overlap comparison for simultaneous records; do not assume IDs imply independent subjects.
    overlap=[]
    for i,a in df.iterrows():
        for _,b in df.iloc[i+1:].iterrows():
            ta,tb=pd.Timestamp(a.origin),pd.Timestamp(b.origin)
            lo=max(ta,tb); hi=min(ta+pd.Timedelta(seconds=a.duration_s),tb+pd.Timedelta(seconds=b.duration_s))
            if hi<=lo: continue
            aligned=(ta.value-tb.value)%4_000_000==0
            eq=None
            if aligned:
                xa=np.load('cache/records/'+a.record+'.npz'); xb=np.load('cache/records/'+b.record+'.npz')
                ia=int((lo-ta).total_seconds()*250+.01); ib=int((lo-tb).total_seconds()*250+.01)
                n=int((hi-lo).total_seconds()*250+.01)
                mask=xa['valid'][ia:ia+n]&xb['valid'][ib:ib+n]
                eq=float(np.mean(xa['x'][ia:ia+n][mask]==xb['x'][ib:ib+n][mask])) if mask.any() else None
            overlap.append(dict(a=a.record,b=b.record,seconds=(hi-lo).total_seconds(),aligned=aligned,exact_fraction=eq))
    pd.DataFrame(overlap).to_csv('reports/time_overlaps.csv',index=False)
    fig,axes=plt.subplots(3,1,figsize=(13,9))
    df.plot(x='record',y=['relax_s','task_s'],kind='bar',stacked=True,ax=axes[0]); axes[0].set_ylabel('Labeled seconds')
    df.plot(x='record',y='gap_s',kind='bar',ax=axes[1],legend=False);axes[1].set_ylabel('Missing seconds');axes[1].set_ylim(bottom=0)
    df.plot(x='record',y='abs_p99_uv',ax=axes[2],legend=False);axes[2].set_ylabel('99th |amplitude| (uV)');axes[2].set_yscale('log')
    fig.tight_layout();fig.savefig('reports/dataset_overview.png',dpi=150);plt.close(fig)
    text=f'''# 数据集检查\n\n实际读取 {len(df)} 份 BDF，均为单通道 EEG、250 Hz。原文件 SHA256 保存在 original_manifest.csv。\n\n有效 relax 秒数 {df.relax_s.sum():.3f}；task {df.task_s.sum():.3f}；标注缺失 {df.gap_s.sum():.3f} 秒。\n阶段总数 {len(tf)}；持续时间范围 {tf.duration_s.min():.6f} 至 {tf.duration_s.max():.6f} 秒。完整信号重复记录：{duplicates}。\n\n文件名起始时间使用 UTC+08:00 和毫秒；CSV 保留纳秒精度后用整数向上取整映射到 [start,end) 样本。MNE 把 BDF 本地时钟显示为 UTC，不能直接拿该值与带时区 CSV 相减。\n\nBDF subject_info 只有编号和通用信息，未取得可靠受试者身份映射，验证只保证记录隔离，不声称受试者隔离。时间重叠不自动说明同一受试者；精确重叠信号比较见 time_overlaps.csv。\n\n异常振幅仅报告，不用标签筛除；所有 BAD 标注与无标签样本排除。长平坦段另列供审核。状态转换窗口按全窗同标签规则排除；在线推理不读取阶段标签，另评估按末端标签的连续流。\n\n![数据分布](dataset_overview.png)\n\n细表：records.csv、trials.csv、gaps.csv、time_overlaps.csv。\n'''
    Path('reports/dataset_analysis.md').write_text(text,encoding='utf8')
    packages={}
    for p in ('numpy','scipy','pandas','scikit-learn','mne','torch','matplotlib','joblib','pytest'):
        try: packages[p]=version(p)
        except PackageNotFoundError: packages[p]='not installed'
    Path('reports/package_versions.json').write_text(json.dumps(packages,indent=2),encoding='utf8')
    Path('reports/audit_summary.json').write_text(json.dumps(dict(records=len(df),trials=len(tf),duplicates=duplicates,issues={r['record']:r['issues'] for r in records if r['issues']}),indent=2),encoding='utf8')
if __name__=='__main__': run()
