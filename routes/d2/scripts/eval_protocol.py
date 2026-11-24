"""Does the adaptive decoder rely on the fixed relax-first order? Stream reordered recordings; report score vs elapsed time."""
import json, os, shutil, sys, tempfile
sys.path.insert(0, os.getcwd())
import numpy as np, lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from src.data.io import load_all, FS
from src.evaluation.splits import load_splits
from src.inference.decoder import StreamingDecoder
from scripts.train_final import build_rank
z = np.load("cache/feat2_w6_s1.npz")
F, y, rec, clean, pure = np.nan_to_num(z["F"]), z["y"], z["rec"], z["clean"], z["pure"]
sp = load_splits(); recs = {r.rec_id: r for r in load_all("EEG_dataset") if r.rec_id in sum(sp["folds"], [])}
P = {**json.load(open("configs/lgb_best.json")), "subsample_freq": 1}
ORDERS = {"original rtrtrt": [0,1,2,3,4,5], "swapped trtrtr": [1,0,3,2,5,4], "blocks rrrttt": [0,2,4,1,3,5], "blocks tttrrr": [1,3,5,0,2,4]}
def run(rid, d, order, keep_pre):
    r = recs[rid]; segs, labs = [], []
    if keep_pre and r.trials[0].start > 0:
        segs.append(r.x[:r.trials[0].start]); labs.append(np.full(r.trials[0].start, -1))
    for k in order:
        t = r.trials[k]; s = r.x[t.start:t.end].copy(); s[r.bad[t.start:t.end]] = np.nan
        segs.append(s); labs.append(np.full(len(s), t.label))
    x, lab = np.concatenate(segs), np.concatenate(labs)
    dec = StreamingDecoder(d); out = dec.push(x)
    idx = np.array([o.sample_index for o in out]); p = np.array([o.p_task for o in out])
    q = (lab[idx] >= 0) & np.isfinite(x[idx])
    return lab[idx][q], p[q], idx[q] / FS
tmp = tempfile.mkdtemp(); M = {}
for k, fold in enumerate(sp["folds"]):
    pool = sum([f for j, f in enumerate(sp["folds"]) if j != k], [])
    d = np.isin(rec, pool) & clean; mu, sd = F[d].mean(0), F[d].std(0) + 1e-6
    R = build_rank(F, rec, clean, mu, sd); tr = np.isin(rec, pool) & clean & pure & (y >= 0)
    dd = f"{tmp}/f{k}"; os.makedirs(dd); fn = "m.txt"
    lgb.LGBMClassifier(**P, n_jobs=8, verbose=-1, random_state=0).fit(R[tr], y[tr]).booster_.save_model(f"{dd}/{fn}")
    json.dump(dict(models=[fn], mu=mu.tolist(), sd=sd.tolist(), columns=None, threshold=0.5, adapt=True), open(f"{dd}/config.json", "w"))
    M[k] = (dd, fold)
ALL = {}
for name, order in ORDERS.items():
    res = []
    for k, (dd, fold) in M.items():
        res += Parallel(n_jobs=6)(delayed(run)(r, dd, order, name.startswith("original")) for r in fold)
    yy = np.concatenate([a[0] for a in res]); pp = np.concatenate([a[1] for a in res]); tt = np.concatenate([a[2] for a in res])
    ra = [roc_auc_score(a[0], a[1]) for a in res]
    print(f"{name:18s} AUC {roc_auc_score(yy, pp):.4f} bACC {balanced_accuracy_score(yy, pp >= .5):.4f} recAUC {np.mean(ra):.4f}+-{np.std(ra):.3f}", flush=True)
    ALL[name] = (yy, pp, tt)
yy, pp, tt = ALL["original rtrtrt"]
for a, b in [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500), (1500, 1900)]:
    q = (tt >= a) & (tt < b)
    cls = sorted(set(yy[q].tolist()))
    print(f"elapsed {a:4d}-{b:4d}s n={q.sum():6d} classes {cls} mean_p {pp[q].mean():.3f} bACC {balanced_accuracy_score(yy[q], pp[q] >= .5):.3f} AUC {roc_auc_score(yy[q], pp[q]) if len(cls) == 2 else float('nan'):.3f}", flush=True)
shutil.rmtree(tmp)
