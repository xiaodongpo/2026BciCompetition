# -*- coding: utf-8 -*-
"""构建不依赖阶段标签的连续4秒推理窗口；标签仅用于末端评估。"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset
from prepared_data import DATA,SPLIT,sha
from priority_data_oof import load_data as load_prepared
sys.path.insert(0,str(Path(__file__).resolve().parent/'data_processing_1_50hz'))
from quality_preprocess_1_50hz import transform

from pipeline_paths import WORK_DIR as ROOT
OUT=ROOT/'runs/energy_stream_training_50epochs'
CACHE=OUT/'stream_cache'


def prepare():
    import mne
    CACHE.mkdir()
    split=json.loads(SPLIT.read_text())
    meta={}
    for rid in split['train_records']+split['validation_records']:
        directory=ROOT/'processing_current'/rid
        info=json.loads((directory/'metadata.json').read_text())
        path=Path(info['bdf_path'])
        assert sha(path)==info['bdf_sha256']
        raw=mne.io.read_raw_bdf(path,preload=False,verbose='ERROR')
        signal=raw.get_data(units='uV');raw.close()
        assert signal.shape==(1,info['samples'])
        bits=np.load(directory/'quality_bits.npy',allow_pickle=False)
        # 不使用原离线平直标记bit8；仅在当前已到达窗口内重新判断1秒平直。
        bad=(bits&7)!=0
        prefix=np.r_[0,np.cumsum(bad)]
        starts=[]
        for start in range(0,signal.shape[1]-999,250):
            if prefix[start+1000]!=prefix[start]:continue
            x=signal[:,start:start+1000]
            flat=np.abs(np.diff(x[0]))<=info['quantization_step_uv']/4
            transitions=np.diff(np.r_[False,flat,False].astype(np.int8))
            spans=np.flatnonzero(transitions==-1)-np.flatnonzero(transitions==1)
            if len(spans) and spans.max()+1>=250:continue
            starts.append(start)
        starts=np.array(starts,dtype=np.int64)
        windows=np.stack([signal[:,s:s+1000].astype(np.float32) for s in starts])
        _,model,_,_,_,keep=transform(windows)
        starts,model=starts[keep],model[keep,0]
        old=np.load(directory/'paired_windows.npz',allow_pickle=False)
        lookup={int(s):i for i,s in enumerate(starts)}
        evaluation=np.array([lookup[int(s)] for s in old['start_sample']],dtype=np.int64)
        expected=np.load(DATA/(rid+'_X.npy'),allow_pickle=False)
        assert np.array_equal(model[evaluation],expected)
        np.save(CACHE/(rid+'_X.npy'),model,allow_pickle=False)
        np.save(CACHE/(rid+'_starts.npy'),starts,allow_pickle=False)
        np.save(CACHE/(rid+'_eval.npy'),evaluation,allow_pickle=False)
        meta[rid]=dict(stream_windows=len(starts),evaluation_windows=len(evaluation),source_bdf_sha256=sha(path),
                       source_quality_sha256=sha(directory/'quality_bits.npy'),
                       hashes={k:sha(CACHE/(rid+'_'+k+'.npy')) for k in ['X','starts','eval']})
        print('STREAM_CACHE',rid,len(starts),len(evaluation),flush=True)
    (CACHE/'metadata.json').write_text(json.dumps(dict(status='PASS',records=meta,uses_phase_labels_for_window_availability=False),indent=2))


def load(split_path):
    groups,origins,split,audit=load_prepared(DATA,split_path)
    training=groups['train']
    xs,indices,previous=[],[],[]
    offset=0
    meta=json.loads((CACHE/'metadata.json').read_text())
    for rid in split['validation_records']:
        for k,h in meta['records'][rid]['hashes'].items():assert sha(CACHE/(rid+'_'+k+'.npy'))==h
        x=np.load(CACHE/(rid+'_X.npy'),allow_pickle=False)
        starts=np.load(CACHE/(rid+'_starts.npy'),allow_pickle=False)
        evaluate=np.load(CACHE/(rid+'_eval.npy'),allow_pickle=False)
        prev=np.full((len(starts),3),-1,dtype=np.int64)
        for i,s in enumerate(starts):
            valid=[i]
            if i>=1 and starts[i-1]+250==s:
                valid.append(i-1)
                if i>=2 and starts[i-2]+250==starts[i-1]:valid.append(i-2)
            prev[i,:len(valid)]=np.array(valid)+offset
            assert int(s+1000-starts[valid[-1]])<=1500
        xs.append(x);indices.append(evaluate+offset);previous.append(prev);offset+=len(x)
    return training,TensorDataset(torch.from_numpy(np.concatenate(xs))),groups['validation'].tensors[1].numpy(),np.concatenate(indices),np.concatenate(previous),origins['validation'],split


def average(p,previous,width):
    ix=previous[:,:width]
    return np.where(ix>=0,p[np.maximum(ix,0)],0).sum(1)/(ix>=0).sum(1)


if __name__=='__main__':prepare()
