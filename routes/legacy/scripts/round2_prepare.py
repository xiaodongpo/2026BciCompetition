import json,time,hashlib,argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.round2_features import get_features,singlem_input

def backbone(pretrained=True):
    import torch
    from third_party.singlem_model import Config,EEGEncoder
    torch.manual_seed(20261103);config=Config();config.mask_prob=0.
    model=EEGEncoder(config)
    if pretrained:
        state=torch.load('third_party/singlem_downstream_excluded.pt',map_location='cpu',weights_only=True)
        if 'encoder_state_dict' in state:
            assert all(getattr(config,k)==v for k,v in state['model_config'].items())
            state=state['encoder_state_dict']
        elif 'model_state_dict' in state:state=state['model_state_dict']
        elif 'state_dict' in state:state=state['state_dict']
        if any(k.startswith('encoder.') for k in state):state={k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')}
        model.load_state_dict(state,strict=True)
    return model.eval()

def run(stage):
    protocol=json.loads(Path('configs/round2/protocol.json').read_text());directory=Path('cache/round2')
    meta=pd.read_csv('cache/windows_6s.csv',dtype={'record':str});selected=np.flatnonzero(meta.record.isin(protocol['development']))
    meta=meta.iloc[selected].reset_index(drop=True);assert not set(meta.record)&set(protocol['excluded_previous_holdout'])
    meta.to_csv(directory/'windows.csv',index=False)
    raw=np.load('cache/raw_6s.npy',mmap_mode='r')
    if not (directory/'raw.npy').exists():
        out=np.lib.format.open_memmap(directory/'raw.npy',mode='w+',dtype='float32',shape=(len(selected),1500))
        for a in range(0,len(selected),256):out[a:a+256]=raw[selected[a:a+256]]
        del out
    x=np.load(directory/'raw.npy',mmap_mode='r');timings={}
    for kind in ('legacy','robust','filterbank'):
        if stage!='classical':break
        path=directory/(kind+'.npy')
        marker=directory/(kind+'_complete.json')
        if path.exists() and marker.exists():continue
        temporary=directory/(kind+'.partial')
        start=time.perf_counter();first=get_features(x[:1],kind);out=np.lib.format.open_memmap(temporary,mode='w+',dtype='float32',shape=(len(x),first.shape[1]))
        for a in range(0,len(x),128):out[a:a+128]=get_features(x[a:a+128],kind)
        assert np.isfinite(out).all();timings[kind]=time.perf_counter()-start;print(kind,out.shape,timings[kind],flush=True);out.flush();del out
        temporary.replace(path);marker.write_text(json.dumps(dict(seconds=timings[kind],complete=True)))
    if stage=='classical':return
    import torch
    weight=Path('third_party/singlem_downstream_excluded.pt')
    expected=json.loads(Path('third_party/singlem_checksums.json').read_text())[weight.name]['sha256']
    assert hashlib.sha256(weight.read_bytes()).hexdigest()==expected,'Official weight checksum mismatch'
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    for pretrained in (True,False):
        kind='singlem' if pretrained else 'singlem_random';path=directory/(kind+'.npy')
        marker=directory/(kind+'_complete.json');temporary=directory/(kind+'.partial')
        if path.exists() and marker.exists():continue
        model=backbone(pretrained).to(device);start=time.perf_counter();torch.cuda.reset_peak_memory_stats() if device=='cuda' else None
        out=np.lib.format.open_memmap(temporary,mode='w+',dtype='float32',shape=(len(x),48))
        with torch.no_grad():
            for a in range(0,len(x),64):
                z=torch.from_numpy(singlem_input(x[a:a+64])).to(device);rep,_,_=model(z)
                # Pool fixed 5 one-second tokens; all context is contained inside this decision's buffer.
                out[a:a+64]=torch.cat([rep.mean(1),rep.std(1),rep[:,-1,:]],dim=1).cpu().numpy()
                if a%8192==0:print(kind,a,len(x),flush=True)
        assert np.isfinite(out).all();timings[kind]=dict(seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0)
        print(kind,timings[kind],flush=True);out.flush();del out,model
        temporary.replace(path);marker.write_text(json.dumps(timings[kind]))
    np.save(directory/'fusion.npy',np.concatenate([np.load(directory/'robust.npy'),np.load(directory/'singlem.npy')],axis=1))
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('src/round2_features.py'),Path('src/features.py'),Path('third_party/singlem_model.py'),Path('third_party/singlem_downstream_excluded.pt'),Path('configs/round2/protocol.json')]}
    timings={p.name.removesuffix('_complete.json'):json.loads(p.read_text()) for p in directory.glob('*_complete.json')}
    feature_digests={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.npy') if p.name!='raw.npy'}
    Path('reports/round2/feature_provenance.json').write_text(json.dumps(dict(records=sorted(set(meta.record)),windows=len(meta),timings=timings,sources=sources,feature_digests=feature_digests,device=device),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['classical','singlem'],required=True)
    run(parser.parse_args().stage)
