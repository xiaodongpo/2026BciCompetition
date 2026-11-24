import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.round3_features import get_features


def run():
    root=Path('cache/round3');protocol=json.loads(Path('configs/round3/protocol.json').read_text())
    meta=pd.read_csv('cache/round2/windows.csv',dtype={'record':str})
    assert set(meta.record)==set(protocol['development']) and not set(meta.record)&set(protocol['excluded_previous_holdout'])
    meta.to_csv(root/'windows.csv',index=False)
    raw=np.load('cache/round2/raw.npy',mmap_mode='r')
    source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('src/round3_features.py'),Path('src/round2_features.py'),Path('src/features.py'),Path('scripts/round3_prepare.py'),Path('configs/round3/protocol.json')]}
    for kind in ('base','normalized','multiscale'):
        path=root/(kind+'.npy');marker=root/(kind+'_complete.json')
        if path.exists() and marker.exists():
            assert json.loads(marker.read_text())['sources']==source
            continue
        start=time.perf_counter()
        if kind=='base':
            values=np.load('cache/round2/filterbank.npy')
            # Verify reuse against raw samples, rather than merely trusting a filename.
            np.testing.assert_allclose(values[[0,len(values)//2,-1]],get_features(raw[[0,len(raw)//2,-1]],kind),rtol=1e-5,atol=1e-6)
            np.save(root/'base.partial.npy',values);(root/'base.partial.npy').replace(path)
            dims=values.shape[1]
        else:
            first=get_features(raw[:1],kind);dims=first.shape[1]
            temp=root/(kind+'.partial.npy');out=np.lib.format.open_memmap(temp,mode='w+',dtype='float32',shape=(len(raw),dims))
            for a in range(0,len(raw),128):out[a:a+128]=get_features(raw[a:a+128],kind)
            assert np.isfinite(out).all();out.flush();del out;temp.replace(path)
        item=dict(shape=[len(raw),dims],seconds=time.perf_counter()-start,sources=source,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        marker.write_text(json.dumps(item,indent=2));print(kind,item['shape'],item['seconds'],flush=True)
    provenance=dict(records=sorted(set(meta.record)),windows=len(meta),raw_cache='cache/round2/raw.npy',raw_sha256=hashlib.sha256(Path('cache/round2/raw.npy').read_bytes()).hexdigest(),features={k:json.loads((root/(k+'_complete.json')).read_text()) for k in ('base','normalized','multiscale')})
    Path('reports/round3/feature_provenance.json').write_text(json.dumps(provenance,indent=2))


if __name__=='__main__':run()
