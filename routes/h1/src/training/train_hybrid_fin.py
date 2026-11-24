"""Finals version of the hybrid network: features are z-scored against the recording's reference
windows (ten 30-s segments from the first 10 minutes) instead of an online running estimate.

Same network and optimisation recipe as src/training/train_hybrid.py (30-s label lag, 15-s guard,
recording-equal weights, AdamW + OneCycle, amplitude jitter). Every epoch draws a new reference
(and, with --train_orders aug, a new trial order) per training recording.

Evaluation uses the organiser-style simulation in scripts/finals/sim.py with the same random
streams as scripts/finals/bench.py, so predictions can be blended row by row.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import torch.nn.functional as F_

from scripts.finals.sim import PATTERNS, instance, load_recs, rank_vs_ref, z_vs_ref
from src.data.io import load_all
from src.evaluation.splits import load_splits
from src.models.hybrid import HybridNet, net_kwargs, wave_input

FS, WIN = 250, 1500


def lagged(rec, raw_rec, lag_s, guard_s):
    """Label lag_s before the window end; eligible when (window + guard) lies in one lagged phase."""
    lab = raw_rec.label_array()
    end = np.arange(len(rec.F)) * FS + WIN - 1
    t = end - int(round(lag_s * FS))
    yl = np.where(t >= 0, lab[np.maximum(t, 0)], -1).astype(np.int8)
    span = int(round(6 + guard_s))
    keep = np.zeros(len(yl), bool)
    run = 0
    for i in range(len(yl)):
        run = run + 1 if (i > 0 and yl[i] == yl[i - 1] and yl[i] >= 0) else 0
        keep[i] = yl[i] >= 0 and run >= span
    return yl, keep & rec.clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--wave", default="base", choices=["base", "bpn"])
    ap.add_argument("--train_orders", default="aug", choices=["real", "aug"])
    ap.add_argument("--eval_orders", default="all20", choices=["real", "all20"])
    ap.add_argument("--eval_seeds", type=int, default=2)
    ap.add_argument("--lag", type=float, default=30.0)
    ap.add_argument("--guard", type=float, default=15.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max_lr", type=float, default=2e-3)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--export", default=None, choices=[None, "dev", "all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_steps", type=int, default=0, help="smoke test: cap steps per epoch")
    ap.add_argument("--extra", default=None, help="second feature cache appended to the features (e.g. cache/fin/featbpn.npz)")
    ap.add_argument("--adapt", default="z", choices=["z", "zr"], help="z: z-score vs reference; zr: z-score and percentile rank")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if a.device == "cuda":
        assert torch.cuda.is_available(), "GPU required (run on the training host)"
        torch.backends.cudnn.benchmark = True
    out = os.path.join("experiments", a.name)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "args.json"), "w"), indent=1)
    sp = load_splits()
    ids = sum(sp["folds"], []) + (sp["holdout"] if a.export == "all" else [])
    R = load_recs(ids=set(ids), extra=a.extra)

    def adapt(F, Fref):
        Z = z_vs_ref(F, Fref)
        return Z if a.adapt == "z" else np.concatenate([Z, 2.0 * rank_vs_ref(F, Fref) - 1.0], axis=1)

    raw = {r.rec_id: r for r in load_all("EEG_dataset") if r.rec_id in R}
    t0 = time.time()
    for rid, rec in R.items():
        x = np.load(f"cache/fin/x/{rid}.npy").astype(np.float64)
        starts = np.arange(len(rec.F)) * FS
        rec.W = np.concatenate([wave_input(x[starts[s : s + 512, None] + np.arange(WIN)], a.wave) for s in range(0, len(starts), 512)])
        rec.yl, rec.keep = lagged(rec, raw[rid], a.lag, a.guard) if a.lag > 0 else (rec.y, rec.clean & rec.pure)
    print(f"prepared {len(R)} recordings in {time.time() - t0:.0f}s", flush=True)
    zdim = next(iter(R.values())).F.shape[1] * (2 if a.adapt == "zr" else 1)
    plan = [(a.export, [])] if a.export else [(int(f), sp["folds"][int(f)]) for f in a.folds.split(",")]
    for k, test in plan:
        fd = os.path.join(out, f"fold{k}")
        if os.path.exists(os.path.join(fd, "rows.pkl" if not a.export else f"seed{a.seeds - 1}.pt")):
            print(f"fold {k} already done", flush=True)
            continue
        os.makedirs(fd, exist_ok=True)
        tf = time.time()
        train = [r for r in ids if r not in test]
        Fall = np.concatenate([R[r].F[R[r].clean] for r in train])
        mu, sd = Fall.mean(0), Fall.std(0) + 1e-6
        kk = k if isinstance(k, int) else 7
        logit_sum = {}
        for s in range(a.seeds):
            torch.manual_seed(1000 * kk + s)
            g = np.random.default_rng(1000 * kk + s)
            net = HybridNet(zdim, **net_kwargs(a.wave)).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
            n_rows = sum(int(R[r].keep.sum()) for r in train)
            steps = int(n_rows * 0.8) // a.bs  # about a sixth of the rows sit under the reference each epoch
            if a.max_steps:
                steps = min(steps, a.max_steps)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.max_lr, total_steps=a.epochs * steps)
            net.train()
            for ep in range(a.epochs):
                Zs, Ws, Ys, Wt = [], [], [], []
                for r in train:
                    rec = R[r]
                    ins = instance(rec, g, g.permutation(6).tolist() if a.train_orders == "aug" else None, train_ok=rec.keep)
                    rows, ref = ins["train_rows"], ins["ref_rows"]
                    if len(ref) < 20 or len(rows) == 0:
                        continue
                    Zs.append(adapt(rec.F[rows], rec.F[ref])); Ws.append(rec.W[rows]); Ys.append(rec.yl[rows].astype(np.float32))
                    Wt.append(np.full(len(rows), 1.0 / len(rows), np.float32))
                Z_ep, W_ep, Y_ep, T_ep = map(np.concatenate, (Zs, Ws, Ys, Wt))
                T_ep = T_ep / T_ep.mean()
                perm = g.permutation(len(Y_ep))
                tot, nb = 0.0, min(steps, len(perm) // a.bs)
                for b in range(nb):
                    j = np.sort(perm[b * a.bs : (b + 1) * a.bs])
                    rw = torch.from_numpy(W_ep[j]).to(dev)
                    rw = rw * (1 + 0.2 * (torch.rand(len(j), 1, device=dev) - 0.5))
                    loss = F_.binary_cross_entropy_with_logits(net(rw, torch.from_numpy(Z_ep[j]).to(dev)), torch.from_numpy(Y_ep[j]).to(dev), weight=torch.from_numpy(T_ep[j]).to(dev))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    if sched.last_epoch < a.epochs * steps - 1:
                        sched.step()
                    tot += loss.item()
                if ep % 10 == 9 or ep == a.epochs - 1:
                    print(f"{a.name} fold{k} seed{s} ep{ep} loss {tot / max(nb, 1):.4f} {time.time() - tf:.0f}s", flush=True)
            torch.save(net.state_dict(), os.path.join(fd, f"seed{s}.pt"))
            if a.export:
                continue
            net.eval()
            with torch.no_grad():
                for rid in test:
                    rec = R[rid]
                    E = torch.cat([net.wave(torch.from_numpy(rec.W[s0 : s0 + 2048]).to(dev)) for s0 in range(0, len(rec.W), 2048)])
                    for seed in range(a.eval_seeds):
                        for pat in (PATTERNS if a.eval_orders == "all20" else ["real"]):
                            erng = np.random.default_rng([seed, int(rid), PATTERNS.index(pat) if pat != "real" else 99])
                            ins = instance(rec, erng, None if pat == "real" else rec.order_for(pat))
                            t = ins["test_rows"]
                            t = t[(t >= 0) & rec.pure[np.maximum(t, 0)]]
                            ok = rec.clean_sig[t] & (len(ins["ref_rows"]) >= 20)
                            lg = np.zeros(len(t), np.float32)
                            if ok.any():
                                Z = torch.from_numpy(adapt(rec.F[t[ok]], rec.F[ins["ref_rows"]])).to(dev)
                                lg[ok] = net.head_from(E[torch.from_numpy(t[ok]).to(dev)], Z).float().cpu().numpy()
                            key = (rid, seed, pat)
                            if key not in logit_sum:
                                logit_sum[key] = dict(rec=rid, seed=seed, pat=pat, y=rec.y[t].astype(np.int8), ok=ok, logit=np.zeros(len(t), np.float32))
                            logit_sum[key]["logit"] += lg / a.seeds
        n_params = sum(p.numel() for p in net.parameters())
        if a.export:
            json.dump(dict(args=vars(a), export=a.export, train_ids=train, mu=mu.tolist(), sd=sd.tolist(), train_s=time.time() - tf, n_params=n_params), open(os.path.join(fd, "info.json"), "w"))
            print(f"{a.name} export {a.export} DONE {time.time() - tf:.0f}s on {len(train)} recordings", flush=True)
            continue
        rows = list(logit_sum.values())
        for r in rows:
            r["hyb"] = np.where(r["ok"], 1 / (1 + np.exp(-r["logit"])), 0.5)
        pickle.dump(rows, open(os.path.join(fd, "rows.pkl"), "wb"))
        json.dump(dict(args=vars(a), fold=k, train_s=time.time() - tf, n_params=n_params, test=test), open(os.path.join(fd, "info.json"), "w"), indent=1)
        print(f"{a.name} fold{k} DONE {time.time() - tf:.0f}s params {n_params}", flush=True)


if __name__ == "__main__":
    main()
