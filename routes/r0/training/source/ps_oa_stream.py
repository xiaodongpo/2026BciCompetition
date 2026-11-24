# -*- coding: utf-8 -*-
"""完整流上的顺序重排与在线树特征。流 = 前缀(首个评价窗之前) + 各阶段(含其后不计分窗口)。"""
import numpy as np
from ps_oa_common import *
SF = OUT / 'stream_feat'


def load_rec(rid):
    d = SF / rid
    F = np.nan_to_num(np.load(d / 'F.npy').astype(np.float64), posinf=0, neginf=0)
    ev = np.load(CACHE / (rid + '_eval.npy'))
    return dict(rid=rid, F=F, v6=np.load(d / 'valid6.npy'), y=np.load(d / 'y.npy'), elapsed=np.load(d / 'elapsed.npy'), eval=ev,
                starts=np.load(CACHE / (rid + '_starts.npy')))


def blocks(rec):
    """返回 (前缀下标, [每阶段下标...], [每阶段标签...])，覆盖全部流窗口。"""
    ev, e, y = rec['eval'], rec['elapsed'], rec['y']
    cuts = np.flatnonzero(np.diff(e) < 0) + 1
    first = np.r_[0, cuts]
    head = ev[first]
    n = len(rec['starts'])
    bounds = np.r_[head, n]
    return np.arange(0, head[0]), [np.arange(bounds[i], bounds[i + 1]) for i in range(len(head))], [int(y[f]) for f in first]


def order(labs, rng, kind):
    ids = list(range(len(labs)))
    if kind == 'ORIG':
        return ids
    if kind == 'TASKFIRST':
        rel = [i for i in ids if labs[i] == 0]
        tsk = [i for i in ids if labs[i] == 1]
        o = []
        for a, b in zip(tsk, rel):
            o += [a, b]
        return o + tsk[len(rel):] + rel[len(tsk):]
    rng.shuffle(ids)
    return ids


def seq(rec, kind, rng):
    pre, ph, labs = blocks(rec)
    o = order(labs, rng, kind)
    return np.concatenate([pre] + [ph[i] for i in o])


def online_z(F, valid, mu0, sd0, prior, halves):
    out = []
    for h in halves:
        decay = 1.0 if h is None else 0.5 ** (1 / h)
        mean = mu0.copy()
        w = float(prior)
        m2 = (sd0 ** 2 + 1e-12) * w
        z = np.zeros_like(F)
        for t in range(len(F)):
            z[t] = (F[t] - mean) / np.sqrt(m2 / w)
            if valid[t]:
                w = w * decay + 1
                m2 = m2 * decay
                d = F[t] - mean
                mean = mean + d / w
                m2 = m2 + d * (F[t] - mean)
        out.append(z)
    return np.concatenate(out, 1)


def center_cum(l, k):
    """累计 logit 中心化：先判断后更新。"""
    n = len(l)
    c = np.r_[0.0, np.cumsum(l)[:-1]]
    return l - c / (k + np.arange(n))


def build_train(recs, mu0, sd0, copies, rng, halves=(None, 300.0), prior=60, shuffle_rng=None):
    Xs, Ys, Rs = [], [], []
    for ri, rec in enumerate(recs):
        ylab = np.full(len(rec['starts']), -1)
        ylab[rec['eval']] = rec['y']
        if shuffle_rng is not None:
            # 泄漏自检：按阶段打乱标签
            pre, ph, labs = blocks(rec)
            l2 = list(labs)
            shuffle_rng.shuffle(l2)
            for p, nl in zip(ph, l2):
                m = ylab[p] >= 0
                ylab[p[m]] = nl
        for c in copies:
            idx = seq(rec, c, rng)
            x = online_z(rec['F'][idx], rec['v6'][idx], mu0, sd0, prior, list(halves))
            k = (ylab[idx] >= 0) & rec['v6'][idx]
            Xs.append(x[k])
            Ys.append(ylab[idx][k])
            Rs.append(np.full(k.sum(), ri))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Rs)


def fit_tree(X, Y, RR, n, copies, seed=42):
    import lightgbm as lgb
    w = np.zeros(len(Y))
    for r in np.unique(RR):
        for c in [0, 1]:
            m = (RR == r) & (Y == c)
            if m.sum():
                w[m] = 1.0 / m.sum()
    w = w * len(Y) / w.sum()
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=.05, num_leaves=15, max_depth=5, min_child_samples=200 * copies, reg_lambda=5,
                              subsample=.7, subsample_freq=1, colsample_bytree=.7, random_state=seed, n_jobs=8, verbosity=-1,
                              force_col_wise=True).fit(X, Y, sample_weight=w)


def fuse(tree_logit, energy_c, v6, wt=.5):
    return np.where(v6, wt * tree_logit + (1 - wt) * energy_c, energy_c)
