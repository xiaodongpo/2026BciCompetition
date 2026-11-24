import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.round2_features import filterbank_features

def run():
    root=Path('cache/round4');protocol=json.loads(Path('configs/round4/protocol.json').read_text())
    meta=pd.read_csv('cache/round2/windows.csv',dtype={'record':str});assert set(meta.record)==set(protocol['development']);assert not set(meta.record)&set(protocol['excluded_previous_holdout'])
    meta.to_csv(root/'windows.csv',index=False);raw=np.load('cache/round2/raw.npy',mmap_mode='r');base=np.load('cache/round2/filterbank.npy')
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('src/features.py'),Path('src/round2_features.py'),Path(__file__),Path('configs/round4/protocol.json')]}
    for key,w in [('w0',0),('w015',.15),('w035',.35)]:
        path=root/(key+'.npy');marker=root/(key+'_complete.json')
        if marker.exists():assert json.loads(marker.read_text())['sources']==sources;continue
        start=time.perf_counter();temp=root/(key+'.partial.npy');dims=125 if w==0 else 375
        out=np.lib.format.open_memmap(temp,mode='w+',dtype='float32',shape=(len(raw),dims));out[:,:125]=base
        if w:
            for a in range(0,len(raw),128):
                x=np.asarray(raw[a:a+128],dtype=np.float64)
                out[a:a+128,125:250]=filterbank_features(x*np.exp(-w));out[a:a+128,250:375]=filterbank_features(x*np.exp(w))
        assert np.isfinite(out).all();out.flush();del out;temp.replace(path)
        item=dict(seconds=time.perf_counter()-start,shape=[len(raw),dims],sources=sources,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        marker.write_text(json.dumps(item,indent=2));print(key,item['seconds'],flush=True)
    Path('reports/round4/feature_provenance.json').write_text(json.dumps(dict(records=sorted(set(meta.record)),windows=len(meta),raw_sha256=hashlib.sha256(Path('cache/round2/raw.npy').read_bytes()).hexdigest(),features={k:json.loads((root/(k+'_complete.json')).read_text()) for k in ('w0','w015','w035')}),indent=2))

if __name__=='__main__':run()
