# -*- coding: utf-8 -*-
"""为全部 57 条记录的完整流窗口（与 ENERGY 流缓存同一时刻）计算 165 维 6 秒内特征。不读标签决定特征。"""
import json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_paths import WORK_DIR as PROJECT, WORKERS
from strict6s_pipeline import current_quality, extract_batch
CACHE = PROJECT / 'runs/energy_stream_training_50epochs/stream_cache'
OUT = PROJECT / 'runs/online_adaptive_energy_20261106/stream_feat'
ST = PROJECT / 'runs/strict6s_energy_lgbm_cv3_50_20261105/cache'

def work(rid):
    import mne, csv
    d = OUT / rid
    if (d / 'DONE').exists(): return rid, 'skip'
    d.mkdir(parents=True, exist_ok=True)
    original = PROJECT / 'processing_current' / rid
    meta = json.loads((original / 'metadata.json').read_text())
    raw = mne.io.read_raw_bdf(meta['bdf_path'], preload=True, verbose='ERROR'); signal = raw.get_data(units='uV')[0]; raw.close()
    starts = np.load(CACHE / (rid + '_starts.npy')); ev = np.load(CACHE / (rid + '_eval.npy'))
    ends = starts + 1000; n = len(ends)
    windows = np.zeros((n, 1500)); valid = np.zeros(n, bool)
    for i, e in enumerate(ends):
        x = signal[max(0, int(e) - 1500):int(e)]
        good, _ = current_quality(x); valid[i] = good
        if len(x) == 1500: windows[i] = x
    F, _ = extract_batch(windows, valid)
    with np.load(original / 'paired_windows.npz', allow_pickle=False) as p:
        ps, pe, y, trial = p['start_sample'].copy(), p['end_sample_exclusive'].copy(), p['y'].copy(), p['trial_id'].copy()
    assert np.array_equal(starts[ev], ps)
    with open(original / 'trials.csv') as f: trials = {int(r['trial_id']): r for r in csv.DictReader(f)}
    elapsed = np.array([(int(s) + int(e)) / 500 - float(trials[int(t)]['start_seconds']) for s, e, t in zip(ps, pe, trial)])
    if (ST / rid / 'F.npy').exists():
        old = np.load(ST / rid / 'F.npy'); assert np.allclose(old, F[ev], atol=1e-3), rid
    np.save(d / 'F.npy', F); np.save(d / 'valid6.npy', valid); np.save(d / 'y.npy', y); np.save(d / 'elapsed.npy', elapsed); np.save(d / 'trial.npy', trial)
    (d / 'DONE').write_text('ok'); return rid, n

if __name__ == '__main__':
    split = json.loads((PROJECT / 'prepared_1_50hz/split.json').read_text()) if (PROJECT / 'prepared_1_50hz/split.json').exists() else None
    rids = sorted(p.name[:3] for p in CACHE.glob('*_X.npy'))
    print(len(rids), flush=True)
    with Pool(WORKERS) as pool:
        for r in pool.imap_unordered(work, rids): print(r, flush=True)
    print('ALLDONE', flush=True)
