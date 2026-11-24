"""Train the hybrid network with 5-fold CV and score all 20 trial orders (GPU, on the training host).

Matches eegcomp's recipe: recording-equal weighted BCE, AdamW (wd 1e-3), OneCycle
(peak 2e-3), batch 256, 30 epochs, amplitude jitter 0.9-1.1, optional 30-s label lag
with a 15-s guard. Order augmentation follows this project's protocol: every epoch
uses a random trial order per training recording (no preference for the original).

Predictions use the same row order as scripts/order_bench.py so they can be compared
and blended with the LightGBM recipes row by row.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import torch.nn.functional as F_

from src.data.io import FS, load_all
from src.evaluation.splits import load_splits
from src.models.hybrid import WAVE_MODES, HybridNet, net_kwargs, online_z, wave_input
from src.preprocessing.rank_norm import rank_recording

PATTERNS = sorted({"".join(p) for p in itertools.permutations("rrrttt")})


def load(features, ids):
    z = np.load(features[0])
    keep = np.isin(z["rec"], ids)
    F = [np.nan_to_num(z["F"][keep])]
    rec, end = z["rec"][keep], z["end"][keep]
    for p in features[1:]:
        e = np.load(p)
        key = {(r, t): i for i, (r, t) in enumerate(zip(e["rec"], e["end"]))}
        F.append(np.nan_to_num(e["F"][np.array([key[(r, t)] for r, t in zip(rec, end)])]))
    return np.concatenate(F, 1).astype(np.float32), z["y"][keep], rec, end, z["clean"][keep], z["pure"][keep]


def lagged_targets(y_rows, end, rec, recs, lag_s, guard_s):
    """Training labels: official label lag_s earlier; keep rows whose (window + guard) span one lagged phase.

    Computed in each recording's original time, so a row carries a label about its own
    physical past; rows then move with their trial when trial order is shuffled.
    """
    yl = np.full(len(y_rows), -1, np.int8)
    keep = np.zeros(len(y_rows), bool)
    span = int(round((6 + guard_s)))  # rows are 1 s apart
    for rid, r in recs.items():
        i = np.where(rec == rid)[0]
        lab = r.label_array()
        t = end[i] - int(round(lag_s * FS))
        ok = t >= 0
        yl[i[ok]] = lab[t[ok]]
        run = 0
        for k, ii in enumerate(i):
            same = k > 0 and yl[ii] == yl[i[k - 1]] and yl[ii] >= 0 and end[ii] - end[i[k - 1]] == FS
            run = run + 1 if same else 0
            keep[ii] = yl[ii] >= 0 and run >= span
    return yl, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--features", default="cache/feat2_w6_s1.npz,cache/feat2_w2_s1.npz")
    ap.add_argument("--adapt", default="z", choices=["z", "rank"])
    ap.add_argument("--lag", type=float, default=30.0)
    ap.add_argument("--guard", type=float, default=15.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n_aug", type=int, default=8)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max_lr", type=float, default=2e-3)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--wave", default="base", choices=list(WAVE_MODES), help="waveform preprocessing variant (src/models/hybrid.py)")
    ap.add_argument("--export", default=None, choices=[None, "dev", "all"], help="train on all dev (46) or all (57) recordings, save nets to fold<export>/, no evaluation")
    a = ap.parse_args()
    dev = torch.device("cuda")
    assert torch.cuda.is_available(), "GPU required (run on the training host)"
    torch.backends.cudnn.benchmark = True
    out = os.path.join("experiments", a.name)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "args.json"), "w"), indent=1)
    sp = load_splits()
    ids = sum(sp["folds"], []) + (sp["holdout"] if a.export == "all" else [])
    feats = a.features.split(",")
    F, y, rec, end, clean, pure = load(feats, ids)
    recs = {r.rec_id: r for r in load_all("EEG_dataset") if r.rec_id in ids}
    t0 = time.time()
    raw = None
    for rid, r in recs.items():
        i = np.where(rec == rid)[0]
        x = np.nan_to_num(r.x)
        for s in range(0, len(i), 512):
            b = i[s : s + 512]
            chunk = wave_input(np.stack([x[e - 1499 : e + 1] for e in end[b]]), a.wave)
            if raw is None:
                raw = np.zeros((len(y),) + chunk.shape[1:], np.float32)
            raw[b] = chunk
    # "scale": divide the waveform by an online amplitude scale (cumulative mean of the
    # window log-RMS over the recording's earlier clean windows, training mean as prior).
    i_lrms = [str(n) for n in np.load(feats[0])["names"]].index("log_rms")
    if a.lag > 0:
        y_tr, keep_tr = lagged_targets(y, end, rec, recs, a.lag, a.guard)
    else:
        y_tr, keep_tr = y, pure & (y >= 0)
    keep_tr = keep_tr & clean
    seg = {}
    for rid, r in recs.items():
        i = np.where(rec == rid)[0]
        e = end[i]
        seg[rid] = (i[e < r.trials[0].start], [i[(e >= t.start) & (e < t.end)] for t in r.trials], [t.label for t in r.trials])
    print(f"prepared {len(y)} rows, train-eligible {keep_tr.sum()} in {time.time() - t0:.0f}s", flush=True)

    def replay(rid, order, mu, sd):
        pre, tr, _ = seg[rid]
        idx = np.concatenate([pre] + [tr[k] for k in order])
        Z = online_z(F[idx], clean[idx], mu, sd) if a.adapt == "z" else rank_recording(F[idx], clean[idx], mu, sd)
        S = None
        if a.wave == "scale":
            v = F[idx, i_lrms].astype(np.float64)
            c = clean[idx].astype(np.float64)
            m = (60.0 * mu[i_lrms] + np.cumsum(v * c) - v * c) / (60.0 + np.cumsum(c) - c)
            S = (10.0 ** m).astype(np.float32)
        return idx, Z, S

    def wave_batch(rows, S_rows):
        w = raw[rows]
        if S_rows is not None:
            w = np.clip(w / S_rows[:, None], -10, 10)
        return torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(dev)

    def order_for(rid, p):
        labs = seg[rid][2]
        rr = [k for k, l in enumerate(labs) if l == 0]
        tt = [k for k, l in enumerate(labs) if l == 1]
        return [rr.pop(0) if c == "r" else tt.pop(0) for c in p]

    plan = [(a.export, [])] if a.export else [(int(f), sp["folds"][int(f)]) for f in a.folds.split(",")]
    for k, test in plan:
        fd = os.path.join(out, f"fold{k}")
        if os.path.exists(os.path.join(fd, "pred.npz" if not a.export else f"seed{a.seeds - 1}.pt")):
            print(f"fold {k} already done", flush=True)
            continue
        os.makedirs(fd, exist_ok=True)
        tf = time.time()
        train = [r for r in ids if r not in test]
        m = np.isin(rec, train) & clean
        mu, sd = F[m].mean(0), F[m].std(0) + 1e-6
        kk = k if isinstance(k, int) else 7
        rng = np.random.default_rng(100 + kk)
        reps = {r: [replay(r, rng.permutation(6).tolist(), mu, sd) for _ in range(a.n_aug)] for r in train}
        tests = [(p, r, replay(r, order_for(r, p), mu, sd)) for p in PATTERNS for r in test]
        logits = {}
        for s in range(a.seeds):
            torch.manual_seed(1000 * kk + s)
            g = np.random.default_rng(1000 * kk + s)
            net = HybridNet(F.shape[1], **net_kwargs(a.wave)).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
            n_rows = sum(int(keep_tr[reps[r][0][0]].sum()) for r in train)
            steps = n_rows // a.bs
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.max_lr, total_steps=a.epochs * steps)
            net.train()
            for ep in range(a.epochs):
                # One random trial order per recording per epoch; every eligible row once.
                Zs, Is, Ws, Ss = [], [], [], []
                for r in train:
                    idx, Z, S = reps[r][g.integers(a.n_aug)]
                    q = keep_tr[idx]
                    Zs.append(Z[q]); Is.append(idx[q]); Ws.append(np.full(q.sum(), 1.0 / max(q.sum(), 1)))
                    Ss.append(S[q] if S is not None else np.ones(q.sum(), np.float32))
                Z_ep, I_ep = np.concatenate(Zs), np.concatenate(Is)
                S_ep = np.concatenate(Ss) if a.wave == "scale" else None
                W_ep = np.concatenate(Ws)
                W_ep = (W_ep / W_ep.mean()).astype(np.float32)
                perm = g.permutation(len(I_ep))
                tot = 0.0
                for b in range(steps):
                    j = perm[b * a.bs : (b + 1) * a.bs]
                    rw = wave_batch(I_ep[j], None if S_ep is None else S_ep[j])
                    jit = 1 + 0.2 * (torch.rand(len(j), 1, device=dev) - 0.5)
                    rw = rw * (jit if rw.dim() == 2 else jit[:, :, None])
                    zz = torch.from_numpy(Z_ep[j]).to(dev)
                    tgt = torch.from_numpy(y_tr[I_ep[j]].astype(np.float32)).to(dev)
                    wt = torch.from_numpy(W_ep[j]).to(dev)
                    loss = F_.binary_cross_entropy_with_logits(net(rw, zz), tgt, weight=wt)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    sched.step()
                    tot += loss.item()
                if ep % 10 == 9 or ep == a.epochs - 1:
                    print(f"{a.name} fold{k} seed{s} ep{ep} loss {tot / steps:.4f} {time.time() - tf:.0f}s", flush=True)
            torch.save(net.state_dict(), os.path.join(fd, f"seed{s}.pt"))
            net.eval()
            if a.export:
                continue
            with torch.no_grad():
                emb = {}
                if a.wave != "scale":  # waveform embedding does not depend on trial order: compute once
                    for r in test:
                        i = np.where(rec == r)[0]
                        emb[r] = (i, torch.cat([net.wave(wave_batch(i[s0 : s0 + 2048], None)) for s0 in range(0, len(i), 2048)]))
                for p, r, (idx, Z, S) in tests:
                    q = clean[idx] & (y[idx] >= 0)
                    if a.wave == "scale":
                        rows, Sq = idx[q], S[q]
                        E_q = torch.cat([net.wave(wave_batch(rows[s0 : s0 + 2048], Sq[s0 : s0 + 2048])) for s0 in range(0, len(rows), 2048)])
                    else:
                        i_all, E = emb[r]
                        E_q = E[torch.from_numpy(np.searchsorted(i_all, idx)[q]).to(dev)]
                    lg = net.head_from(E_q, torch.from_numpy(Z[q]).to(dev)).float().cpu().numpy()
                    logits.setdefault((p, r), []).append(lg)
                    logits.setdefault(("y", p, r), y[idx][q])
        n_params = sum(p.numel() for p in net.parameters())
        if a.export:
            json.dump(dict(args=vars(a), export=a.export, train_ids=train, mu=mu.tolist(), sd=sd.tolist(), train_s=time.time() - tf, n_params=n_params), open(os.path.join(fd, "info.json"), "w"))
            print(f"{a.name} export {a.export} DONE {time.time() - tf:.0f}s on {len(train)} recordings", flush=True)
            continue
        np.savez(os.path.join(fd, "pred.npz"), **{f"{p}|{r}|y": logits[("y", p, r)] for p, r, _ in tests}, **{f"{p}|{r}|p": 1 / (1 + np.exp(-np.mean(logits[(p, r)], axis=0))) for p, r, _ in tests},
                 **{f"{p}|{r}|logit": np.mean(logits[(p, r)], axis=0) for p, r, _ in tests})
        n_params = sum(p.numel() for p in net.parameters())
        info = dict(args=vars(a), fold=k, train_s=time.time() - tf, n_params=n_params, gpu_peak_mb=torch.cuda.max_memory_allocated() / 2**20, best_val=None, best_ep=a.epochs - 1, test=test)
        json.dump(info, open(os.path.join(fd, "info.json"), "w"), indent=1)
        print(f"{a.name} fold{k} DONE {time.time() - tf:.0f}s params {n_params}", flush=True)


if __name__ == "__main__":
    main()
