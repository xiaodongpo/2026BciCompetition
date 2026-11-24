# -*- coding: utf-8 -*-
"""读取当前prepared缓存：依据manifest动态核验记录数和窗口数，不重复预处理。"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset

from pipeline_paths import WORK_DIR as ROOT
DATA = ROOT / 'prepared_1_50hz'
SPLIT = DATA / 'split.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load_data(source=DATA, split_path=SPLIT):
    source = Path(source).resolve()
    assert json.loads((source / 'verification.json').read_text())['status'] == 'PASS'
    assert json.loads((source / 'preprocessing.json').read_text())['bandpass_hz'] == [
        1.0,
        50.0,
    ]
    manifest = json.loads((source / 'manifest.json').read_text())
    entries = {r['record_id']: r for r in manifest['records']}
    split = json.loads(Path(split_path).read_text())
    train, val = set(split['train_records']), set(split['validation_records'])
    assert not train & val and len(train) > 0 and len(val) > 0
    assert len(entries) == manifest['record_count']
    assert train | val <= set(entries)
    entries = {rid: entry for rid, entry in entries.items() if rid in train | val}
    origins_by_record = {rid: [] for rid in entries}
    with (source / 'source_windows_paired.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rid = row['record_id']
            if rid not in entries:
                continue
            origins_by_record[rid].append(
                {
                    k: row[k] if k == 'record_id' else int(row[k])
                    for k in [
                        'record_id',
                        'array_index',
                        'start_sample',
                        'end_sample_exclusive',
                        'trial_id',
                    ]
                }
            )
    groups, origins, summaries = {}, {}, []
    for group, ids in [
        ('train', split['train_records']),
        ('validation', split['validation_records']),
    ]:
        xs, ys, metadata = [], [], []
        for rid in ids:
            xp = source / (rid + '_X.npy')
            yp = source / (rid + '_y.npy')
            xh, yh = sha(xp), sha(yp)
            assert xh == entries[rid]['x_sha256'] and yh == entries[rid]['y_sha256']
            x = np.load(xp, allow_pickle=False)
            y = np.load(yp, allow_pickle=False)
            assert x.dtype == np.float32 and y.dtype == np.int64
            assert x.shape == (len(y), 1, 1000) and np.isfinite(x).all()
            assert np.isin(y, [0, 1]).all()
            assert len(origins_by_record[rid]) == len(y)
            assert [r['array_index'] for r in origins_by_record[rid]] == list(
                range(len(y))
            )
            xs.append(x)
            ys.append(y)
            metadata.extend(origins_by_record[rid])
            summaries.append(
                dict(
                    record_id=rid,
                    split=group,
                    windows=len(y),
                    source_x_sha256=xh,
                    source_y_sha256=yh,
                    source_x=str(xp),
                    source_y=str(yp),
                )
            )
            print('LOADED_PREPARED', rid, group, len(y), flush=True)
        groups[group] = TensorDataset(
            torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys))
        )
        origins[group] = metadata
    assert len(groups['train']) > 0
    assert len(groups['validation']) > 0
    return groups, origins, split, summaries
