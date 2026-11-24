"""Independent implementation of the published reference/window sampling contract.

Only for local replay. Random draws match the competition example's documented
order so historical evaluation can be reproduced without bundling that example.
"""
import numpy as np

def segment_and_shuffle(eeg,seed=14):
    x=np.asarray(eeg)
    if x.ndim!=1 or not np.issubdtype(x.dtype,np.floating) or not np.isfinite(x).all():
        raise ValueError('Expected a finite floating point vector')
    stride,span,count=1500,5,10
    available=min(len(x),150000)//stride
    total=len(x)//stride
    if available<span*count or total<=span*count:raise ValueError('Insufficient recording duration')
    random=np.random.default_rng(seed)
    compressed=random.choice(available-count*(span-1),size=count,replace=False)
    starts=random.permutation(np.sort(compressed)+np.arange(count)*(span-1))*stride
    reference=np.stack([x[int(t):int(t)+stride*span] for t in starts])
    omitted={int(t)//stride+j for t in starts for j in range(span)}
    kept=np.array([j for j in range(total) if j not in omitted],dtype=np.int64)
    selected=kept[random.permutation(len(kept))]
    windows=x[:total*stride].reshape(total,stride)[selected]
    return reference,windows
