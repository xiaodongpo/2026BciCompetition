"""命令行入口。输入原始单通道微伏EEG；不接受events.csv或状态标签。"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from r0 import R0Decoder


def load_signal(path,channel=None,fs=250):
    if fs!=250:raise ValueError('Only 250 Hz is supported; do not silently reinterpret sampling rate')
    path=Path(path)
    if path.suffix.lower()=='.bdf':
        import mne
        raw=mne.io.read_raw_bdf(str(path),preload=True,verbose='ERROR')
        if abs(raw.info['sfreq']-250)>1e-6:raise ValueError('BDF sampling rate must be 250 Hz')
        if channel is None:
            if len(raw.ch_names)!=1:raise ValueError('Multiple channels: specify --channel, available: '+str(raw.ch_names))
            channel=raw.ch_names[0]
        x=raw.get_data(picks=[channel],units='uV')[0];mask=np.ones(len(x),bool)
        for ann in raw.annotations:
            if str(ann['description']).upper().startswith('BAD_ACQ_SKIP'):
                start=int(np.floor((ann['onset']-raw.first_time)*250))
                end=int(np.ceil((ann['onset']-raw.first_time+ann['duration'])*250))
                mask[max(0,start):min(len(x),end)]=False
        raw.close();return x,mask
    if path.suffix.lower()!='.npy':raise ValueError('Input must be raw .npy or .bdf')
    x=np.load(path,mmap_mode='r',allow_pickle=False)
    if x.ndim==2 and x.shape[0]==1:x=x[0]
    if x.ndim!=1:raise ValueError('Expected [N] or [1,N], one EEG channel')
    return x,None


def main():
    base=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description='R0 raw EEG streaming inference, 250 Hz, microvolt')
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',default=str(base/'model'));p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--chunk-size',type=int,default=137);p.add_argument('--channel');p.add_argument('--fs',type=int,default=250)
    args=p.parse_args()
    if args.chunk_size<1:raise ValueError('chunk-size must be positive')
    decoder=R0Decoder(args.model,args.device)
    print('MODEL_LOADED R0 LAG cumulative prototype; training records:',decoder.config['training_records_count'])
    x,mask=load_signal(args.input,args.channel,args.fs);rows=[]
    for start in range(0,len(x),args.chunk_size):
        mm=None if mask is None else mask[start:start+args.chunk_size]
        rows.extend(decoder.push(x[start:start+args.chunk_size],mm))
    out=Path(args.output);out.parent.mkdir(exist_ok=True,parents=True)
    keys=['end_sample','time_sec','score','state','valid','reason','tree_valid','energy_logit','tree_logit','prototype_delta','fused_logit','final_logit']
    if out.suffix.lower()=='.jsonl':
        out.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows),encoding='utf8')
    else:
        with out.open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    print('INFERENCE_SUCCESS',json.dumps(dict(samples=len(x),decisions=len(rows),valid=sum(r['valid'] for r in rows),device=args.device)))
    if rows:print('FIRST_RESULT',json.dumps(rows[0]));print('LAST_RESULT',json.dumps(rows[-1]))


if __name__=='__main__':main()
