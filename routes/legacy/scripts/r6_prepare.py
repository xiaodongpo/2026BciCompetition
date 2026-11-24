"""Read originals, audit all event transitions, cache endpoint-labeled DEV windows only."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.data import read_record
from src.round2_features import filterbank_features
from src.round5_features import ordinal_features


def run():
    for name in ('configs','cache','reports','results','experiments','checkpoints'):(Path(name)/'round6').mkdir(exist_ok=True)
    previous=json.loads(Path('configs/round5/protocol.json').read_text());dev=set(previous['development'])
    root=Path('cache/round6');marker=root/'prepared.json'
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/data.py'),Path('src/features.py'),Path('src/round2_features.py'),Path('src/round5_features.py')]}
    if marker.exists():
        p=json.loads(marker.read_text());assert p['sources']==sources
        for name,digest in p['digests'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
        print('Preparation verified; unchanged.');return
    trial_rows=[];infos=[];rows=[];shards=[];start=time.perf_counter()
    for folder in sorted(Path(__import__('os').environ.get('BCI_DATA_ROOT', 'EEG_dataset')).iterdir()):
        if not folder.is_dir():continue
        x,valid,labels,trials,gaps,events,info=read_record(folder);rid=folder.name;infos.append(info)
        record_windows=[];prev=None
        for j,t in enumerate(trials):
            gap_s=(pd.Timestamp(t['event_start'])-pd.Timestamp(prev['event_end'])).total_seconds() if prev else None
            switch=bool(prev and prev['phase']!=t['phase'] and 0<=gap_s<=1)
            direction=(prev['phase']+'_to_'+t['phase']) if switch else ('initial_'+t['phase'] if prev is None else 'nonadjacent_'+t['phase'])
            tr=dict(record=rid,trial=j,trial_id=rid+'_'+str(j),**t,direction=direction,valid_switch=switch,event_gap_s=gap_s,development=rid in dev)
            trial_rows.append(tr)
            if rid in dev:
                origin=pd.Timestamp(info['origin']);start_ns=pd.Timestamp(t['event_start']).value-origin.value
                for end in range(1500,len(x)+1,500):
                    last=end-1
                    if not t['start']<=last<t['end']:continue
                    first=end-1500
                    if not valid[first:end].all() or labels[last]<0:continue
                    y=int(labels[last]);history=labels[first:end]
                    rows.append(dict(record=rid,end=end,y=y,trial=j,trial_id=tr['trial_id'],direction=direction,valid_switch=switch,t=(last*4_000_000-start_ns)/1e9,pure=bool((history==y).all()),mixed=bool((history==1-y).any()),unlabeled_history=bool((history<0).any()),current_fraction=float(np.mean(history==y))))
                    record_windows.append(x[first:end].copy())
            prev=t
        if rid in dev:
            shard=root/(rid+'_raw.npy');np.save(shard,np.asarray(record_windows,dtype=np.float32));shards.append(shard)
        print('audited',rid,'trials',len(trials),'dev_windows',len(record_windows),flush=True)
    meta=pd.DataFrame(rows);assert set(meta.record)==dev and not meta.duplicated(['record','end']).any()
    meta.to_csv(root/'windows.csv',index=False);pd.DataFrame(trial_rows).to_csv('reports/round6/transition_inventory_all57.csv',index=False)
    pd.DataFrame(infos).to_csv('reports/round6/rechecked_records.csv',index=False)
    raw=np.lib.format.open_memmap(root/'raw.npy',mode='w+',dtype='float32',shape=(len(meta),1500));a=0
    for shard in shards:
        part=np.load(shard,mmap_mode='r');raw[a:a+len(part)]=part;a+=len(part)
    raw.flush();assert a==len(meta)
    base=np.lib.format.open_memmap(root/'base.npy',mode='w+',dtype='float32',shape=(len(meta),125))
    entropy=np.lib.format.open_memmap(root/'ordinal.npy',mode='w+',dtype='float32',shape=(len(meta),20))
    for a in range(0,len(meta),128):
        base[a:a+128]=filterbank_features(raw[a:a+128]);entropy[a:a+128]=ordinal_features(raw[a:a+128])
    base.flush();entropy.flush();del raw,base,entropy
    # Existing pure windows and their exact raw values remain the same subset.
    old=pd.read_csv('cache/round2/windows.csv',dtype={'record':str});pure=meta[meta.pure]
    assert pure[['record','end','y']].reset_index(drop=True).equals(old[['record','end','y']].reset_index(drop=True))
    oldraw=np.load('cache/round2/raw.npy',mmap_mode='r');raw=np.load(root/'raw.npy',mmap_mode='r');ids=np.flatnonzero(meta.pure)
    for a in range(0,len(ids),512):np.testing.assert_array_equal(raw[ids[a:a+512]],oldraw[a:a+512])
    names=['raw.npy','base.npy','ordinal.npy','windows.csv'];digests={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    result=dict(sources=sources,digests=digests,records=sorted(dev),excluded_previous_holdout=previous['excluded_previous_holdout'],windows=len(meta),pure_windows=int(meta.pure.sum()),mixed_windows=int(meta.mixed.sum()),unlabeled_history_windows=int(meta.unlabeled_history.sum()),development_transitions=dict(pd.DataFrame(trial_rows).query('development and valid_switch').direction.value_counts().astype(int)),seconds=time.perf_counter()-start)
    result['development_transitions']={k:int(v) for k,v in result['development_transitions'].items()}
    marker.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
