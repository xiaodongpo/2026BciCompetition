"""Train RankDeep with recording-grouped 5-fold CV and score all 20 trial orders (GPU, on TianXuan).

Training batches: for each sampled recording, a random trial order is drawn; targets
(clean, pure, labeled windows) and history samples (clean windows) get their position
in that order, and each target is compared only with samples that precede it.

Outputs, per fold: model weights, training log, and predictions for the 20 orders in
the same row order as scripts/order_bench.py, so the two can be blended exactly.
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.data.io import load_all
from src.evaluation.splits import load_splits
from src.models.rankdeep import RankDeep, exact_sequence_rank
from src.preprocessing.features import linear_detrend

PATTERNS = sorted({"".join(p) for p in itertools.permutations("rrrttt")})
VAL_PATTERNS = ["rtrtrt", "trtrtr", "rrrttt", "tttrrr"]
WIN, SCALE, CLIP = 1500, 15.0, 20.0


class Data:
    """Signals and the 1-s window grid shared with the hand-crafted feature cache."""

    def __init__(self, feat_path: str, ids: list[str]):
        z = np.load(feat_path)
        keep = np.isin(z["rec"], ids)
        self.F = np.nan_to_num(z["F"][keep]).astype(np.float32)
        self.y, self.rec, self.end = z["y"][keep], z["rec"][keep], z["end"][keep]
        self.clean, self.pure = z["clean"][keep], z["pure"][keep]
        recs = {r.rec_id: r for r in load_all("EEG_dataset") if r.rec_id in ids}
        self.x = {rid: np.nan_to_num(r.x).astype(np.float32) for rid, r in recs.items()}
        self.rows, self.pre, self.trials, self.labs = {}, {}, {}, {}
        for rid, r in recs.items():
            i = np.where(self.rec == rid)[0]
            e = self.end[i]
            self.rows[rid] = i
            self.pre[rid] = i[e < r.trials[0].start]
            self.trials[rid] = [i[(e >= t.start) & (e < t.end)] for t in r.trials]
            self.labs[rid] = [t.label for t in r.trials]

    def windows(self, idx: np.ndarray) -> np.ndarray:
        out = np.empty((len(idx), WIN), np.float32)
        for k, i in enumerate(idx):
            e = self.end[i]
            out[k] = self.x[self.rec[i]][e - WIN + 1 : e + 1]
        w = linear_detrend(out.astype(np.float64)) / SCALE
        return np.clip(w, -CLIP, CLIP).astype(np.float32)

    def sequence(self, rid: str, order: list[int]) -> np.ndarray:
        return np.concatenate([self.pre[rid]] + [self.trials[rid][k] for k in order])

    def order_for(self, rid: str, pattern: str) -> list[int]:
        labs = self.labs[rid]
        rr = [k for k, l in enumerate(labs) if l == 0]
        tt = [k for k, l in enumerate(labs) if l == 1]
        return [rr.pop(0) if c == "r" else tt.pop(0) for c in pattern]


def to_dev(a, dev):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev)


def embed_rows(model, data, idx, hand_mu, hand_sd, dev, bs=512):
    zs = []
    for s in range(0, len(idx), bs):
        b = idx[s : s + bs]
        x = to_dev(data.windows(b)[:, None, :], dev) if model.backbone is not None else None
        h = to_dev(np.clip((data.F[b] - hand_mu) / hand_sd, -5, 5), dev) if model.n_hand else None
        zs.append(model.embed(x, h))
    return torch.cat(zs)


@torch.no_grad()
def score_patterns(model, data, ids, hand_mu, hand_sd, dev, patterns):
    """Per pattern and recording: labels and fused probabilities for clean labeled rows, in replay order."""
    model.eval()
    out = {p: [] for p in patterns}
    for rid in ids:
        rows = data.rows[rid]
        z_all = embed_rows(model, data, rows, hand_mu, hand_sd, dev)
        pos = {int(r): k for k, r in enumerate(rows)}
        for p in patterns:
            seq = data.sequence(rid, data.order_for(rid, p))
            zi = z_all[torch.as_tensor([pos[int(r)] for r in seq], device=dev)]
            r = exact_sequence_rank(model, zi, to_dev(data.clean[seq], dev))
            l_rank, l_abs = model.heads(r, zi)
            prob = model.fuse(l_rank, l_abs).float().cpu().numpy()
            q = data.clean[seq] & (data.y[seq] >= 0)
            out[p].append((rid, data.y[seq][q], prob[q]))
    return out


def batch(data, ids, rng, n_rec, n_tgt, n_ref):
    """Indices and masks for one step: a random trial order per recording."""
    tgt_all, ref_all, blocks = [], [], []
    for rid in rng.choice(ids, n_rec, replace=False):
        seq = data.sequence(rid, rng.permutation(len(data.trials[rid])).tolist())
        pos = np.arange(len(seq))
        cl = data.clean[seq]
        n_hist = np.cumsum(cl) - cl
        ok = cl & data.pure[seq] & (data.y[seq] >= 0)
        t = rng.choice(np.where(ok)[0], n_tgt, replace=False)
        r = rng.choice(np.where(cl)[0], n_ref, replace=False)
        tgt_all.append(seq[t])
        ref_all.append(seq[r])
        blocks.append((pos[t], pos[r], n_hist[t]))
    return tgt_all, ref_all, blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--backbone", default="logpower", choices=["none", "logpower", "tcn", "eegnet", "msnet"])
    ap.add_argument("--hand", type=int, default=0, help="1: concatenate the 126 hand-crafted features")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n_rec", type=int, default=8)
    ap.add_argument("--n_tgt", type=int, default=64)
    ap.add_argument("--n_ref", type=int, default=128)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--feat", default="cache/feat2_w6_s1.npz")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert dev.type == "cuda", "GPU required (run on TianXuan)"
    torch.backends.cudnn.benchmark = True
    out_root = os.path.join("experiments", a.name)
    os.makedirs(out_root, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out_root, "args.json"), "w"), indent=1)
    sp = load_splits()
    dev_ids = sum(sp["folds"], [])
    data = Data(a.feat, dev_ids)
    for k in [int(f) for f in a.folds.split(",")]:
        od = os.path.join(out_root, f"fold{k}")
        if os.path.exists(os.path.join(od, "pred.npz")):
            print(f"fold {k} already done", flush=True)
            continue
        os.makedirs(od, exist_ok=True)
        torch.manual_seed(a.seed * 100 + k)
        rng = np.random.default_rng(a.seed * 100 + k)
        test = sp["folds"][k]
        pool = sorted(r for r in dev_ids if r not in test)
        val = sorted(rng.choice(pool, 6, replace=False).tolist())
        train = [r for r in pool if r not in val]
        m = np.isin(data.rec, train) & data.clean
        hand_mu, hand_sd = data.F[m].mean(0), data.F[m].std(0) + 1e-6
        model = RankDeep(a.backbone, 126 if a.hand else 0, a.tau, a.hidden, a.dropout).to(dev)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps, pct_start=0.1)
        best, best_ep, t0 = -1.0, -1, time.time()
        log = open(os.path.join(od, "train.log"), "a")
        for ep in range(a.epochs):
            model.train()
            tot = 0.0
            for _ in range(a.steps):
                tg, rf, blocks = batch(data, train, rng, a.n_rec, a.n_tgt, a.n_ref)
                idx = np.concatenate(tg + rf)
                x = to_dev(data.windows(idx)[:, None, :], dev) if model.backbone is not None else None
                h = to_dev(np.clip((data.F[idx] - hand_mu) / hand_sd, -5, 5), dev) if model.n_hand else None
                z = model.embed(x, h)
                nt = a.n_rec * a.n_tgt
                zt, zr = z[:nt], z[nt:]
                rs = []
                for b, (pt, pr, nh) in enumerate(blocks):
                    mask = to_dev(pr[None, :] < pt[:, None], dev)
                    rs.append(model.soft_rank(zt[b * a.n_tgt : (b + 1) * a.n_tgt], zr[b * a.n_ref : (b + 1) * a.n_ref], mask, to_dev(nh, dev)))
                l_rank, l_abs = model.heads(torch.cat(rs), zt)
                yt = to_dev(data.y[np.concatenate(tg)].astype(np.float32), dev)
                loss = F.binary_cross_entropy_with_logits(l_rank, yt) + 0.5 * F.binary_cross_entropy_with_logits(l_abs, yt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                sched.step()
                tot += loss.item()
            vs = score_patterns(model, data, val, hand_mu, hand_sd, dev, VAL_PATTERNS)
            v = float(np.mean([roc_auc_score(np.concatenate([y for _, y, _ in vs[p]]), np.concatenate([q for _, _, q in vs[p]])) for p in VAL_PATTERNS]))
            if v > best:
                best, best_ep = v, ep
                torch.save(model.state_dict(), os.path.join(od, "best.pt"))
            msg = f"{a.name} fold{k} ep{ep} loss {tot / a.steps:.4f} valAUC(4 orders) {v:.4f} best {best:.4f}@{best_ep} {time.time() - t0:.0f}s gpuMB {torch.cuda.max_memory_allocated() / 2**20:.0f}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            if ep - best_ep >= a.patience:
                break
        model.load_state_dict(torch.load(os.path.join(od, "best.pt"), map_location=dev))
        res = score_patterns(model, data, test, hand_mu, hand_sd, dev, PATTERNS)
        np.savez(os.path.join(od, "pred.npz"), **{f"{p}|{rid}|y": y for p in PATTERNS for rid, y, _ in res[p]}, **{f"{p}|{rid}|p": q for p in PATTERNS for rid, _, q in res[p]})
        aucs = [roc_auc_score(np.concatenate([y for _, y, _ in res[p]]), np.concatenate([q for _, _, q in res[p]])) for p in PATTERNS]
        info = dict(args=vars(a), fold=k, best_val=best, best_ep=best_ep, n_params=n_params, train_s=time.time() - t0, gpu_peak_mb=torch.cuda.max_memory_allocated() / 2**20, test=test, val=val, test_auc_mean20=float(np.mean(aucs)))
        json.dump(info, open(os.path.join(od, "info.json"), "w"), indent=1)
        print(f"{a.name} fold{k} DONE test mean AUC over 20 orders {np.mean(aucs):.4f} params {n_params}", flush=True)
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
