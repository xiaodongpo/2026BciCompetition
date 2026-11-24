# -*- coding: utf-8 -*-
"""在线自适应实验公共部分：读取已保存的三折基线完整流评分，并按记录切分。只读，不改动原结果。"""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_paths import WORK_DIR as PROJECT
FROZEN = PROJECT / 'runs/energy_frozen_substate_cv_20261105'
CACHE = PROJECT / 'runs/energy_stream_training_50epochs/stream_cache'
OUT = PROJECT / 'runs/online_adaptive_energy_20261106'
SEEDS = [42, 2024, 3407]
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def metric(y, score):
    """score 为 logit 域连续评分，0 为阈值。"""
    pred = score >= 0
    t = pred[y == 1].mean(); r = (~pred[y == 0]).mean()
    return dict(bacc=float((t + r) / 2), auc=float(roc_auc_score(y, score)), task=float(t), relax=float(r))


def load_cv3():
    """返回 records 列表；每条含 p3(3种子×流窗)、eval 索引、y、elapsed、starts、previous(相对本记录)。"""
    records = []
    for fold in [1, 2, 3]:
        split = json.loads((FROZEN / 'cv3' / ('fold%d_split.json' % fold)).read_text())
        jobs = [FROZEN / 'cv3' / ('fold%d_seed%d' % (fold, s)) for s in SEEDS]
        stream = np.stack([np.load(j / 'baseline_stream_probabilities.npy') for j in jobs])
        ctx = np.load(jobs[0] / 'evaluation_context.npz', allow_pickle=False)
        index, previous = ctx['index'], ctx['previous']
        y = np.load(jobs[0] / 'evaluation_y.npy'); e = np.load(jobs[0] / 'evaluation_elapsed.npy')
        origins = json.loads((jobs[0] / 'evaluation_origins.json').read_text())
        rids = np.array([o['record_id'] for o in origins])
        offset = 0
        for rid in split['validation_records']:
            starts = np.load(CACHE / (rid + '_starts.npy'))
            n = len(starts)
            mask = rids == rid
            ev = index[mask] - offset
            assert ev.min() >= 0 and ev.max() < n
            prev = previous[offset:offset + n]
            prev = np.where(prev >= 0, prev - offset, -1)
            records.append(dict(fold=fold, rid=rid, p3=stream[:, offset:offset + n], eval=ev,
                                y=y[mask], elapsed=e[mask], starts=starts, previous=prev))
            offset += n
        assert offset == stream.shape[1]
    return records


def smooth(p, previous):
    ix = previous[:, :3]
    return np.where(ix >= 0, p[np.maximum(ix, 0)], 0).sum(1) / (ix >= 0).sum(1)


def pool(records, scores):
    """scores: 每条记录在评价窗口上的评分。"""
    y = np.concatenate([r['y'] for r in records]); e = np.concatenate([r['elapsed'] for r in records])
    s = np.concatenate(scores)
    out = metric(y, s)
    out['early_bacc'] = metric(y[e < 30], s[e < 30])['bacc']; out['early_auc'] = metric(y[e < 30], s[e < 30])['auc']
    out['later_bacc'] = metric(y[e >= 30], s[e >= 30])['bacc']; out['later_auc'] = metric(y[e >= 30], s[e >= 30])['auc']
    per = [metric(r['y'], x) for r, x in zip(records, scores)]
    out['rec_auc'] = float(np.mean([m['auc'] for m in per])); out['rec_bacc'] = float(np.mean([m['bacc'] for m in per]))
    out['rec_auc_lt_05'] = int(sum(m['auc'] < .5 for m in per))
    out['combo'] = (out['bacc'] + out['auc']) / 2
    return out


def bootstrap(records, a, b, n=1000, seed=0):
    """按记录有放回重采样，比较 b 相对 a 的 combo 差值。"""
    rng = np.random.default_rng(seed); deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(records), len(records))
        y = np.concatenate([records[i]['y'] for i in idx])
        sa = np.concatenate([a[i] for i in idx]); sb = np.concatenate([b[i] for i in idx])
        ma = metric(y, sa); mb = metric(y, sb)
        deltas.append((mb['bacc'] + mb['auc'] - ma['bacc'] - ma['auc']) / 2)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
