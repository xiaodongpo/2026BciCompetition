import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.round5_features import ordinal_features,feature_names


def run():
    root=Path('cache/round5');p=json.loads(Path('configs/round5/protocol.json').read_text())
    meta=pd.read_csv('cache/round2/windows.csv',dtype={'record':str})
    assert set(meta.record)==set(p['development']) and not set(meta.record)&set(p['excluded_previous_holdout'])
    meta.to_csv(root/'windows.csv',index=False)
    raw=np.load('cache/round2/raw.npy',mmap_mode='r');base=np.load('cache/round2/filterbank.npy')
    assert len(raw)==len(base)==len(meta)
    sources={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path('src/features.py'),Path('src/round2_features.py'),Path('src/round5_features.py'),Path(__file__),Path('configs/round5/protocol.json')]}
    for key,dims in [('base',125),('ordinal',20),('fusion',145)]:
        path=root/(key+'.npy');marker=root/(key+'_complete.json')
        if marker.exists():
            item=json.loads(marker.read_text());assert item['sources']==sources
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];continue
        start=time.perf_counter();temp=root/(key+'.partial.npy')
        out=np.lib.format.open_memmap(temp,mode='w+',dtype='float32',shape=(len(raw),dims))
        if key=='base':out[:]=base
        elif key=='ordinal':
            for a in range(0,len(raw),128):out[a:a+128]=ordinal_features(raw[a:a+128])
        else:
            out[:,:125]=base;out[:,125:]=np.load(root/'ordinal.npy')
        assert np.isfinite(out).all();out.flush();del out;temp.replace(path)
        item=dict(seconds=time.perf_counter()-start,shape=[len(raw),dims],sources=sources,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        marker.write_text(json.dumps(item,indent=2));print(key,item['seconds'],flush=True)
    Path('reports/round5/feature_provenance.json').write_text(json.dumps(dict(records=sorted(set(meta.record)),windows=len(meta),ordinal_names=feature_names(),raw_sha256=hashlib.sha256(Path('cache/round2/raw.npy').read_bytes()).hexdigest(),metadata_sha256=hashlib.sha256((root/'windows.csv').read_bytes()).hexdigest(),features={k:json.loads((root/(k+'_complete.json')).read_text()) for k in ('base','ordinal','fusion')}),indent=2))


if __name__=='__main__':run()
