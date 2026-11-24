"""Recording-grouped CV for raw-window nets. One process per fold; resumes from checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from sklearn.metrics import roc_auc_score

from src.data.io import FS, load_all
from src.data.windows import index_windows
from src.evaluation.splits import load_splits
from src.models.nets import build_model
from src.preprocessing.features import linear_detrend
from src.preprocessing.online import DEFAULT_SCALE, scale_trace
from src.preprocessing.quality import make_valid_mask


def prep(w: np.ndarray, scale: np.ndarray, clip: float) -> np.ndarray:
    """Stateless transform: detrend inside the window, scale, soft clip."""
    z = linear_detrend(w.astype(np.float64)) / scale[:, None]
    return np.clip(z, -clip, clip).astype(np.float32)


class Store:
    """Holds whole signals; windows are cut on demand to keep memory low."""

    def __init__(self, recs, win, norm):
        self.win, self.norm = win, norm
        self.x, self.scale, self.pool_ends, self.pool_y, self.eval_ends, self.eval_y = {}, {}, {}, {}, {}, {}
        for r in recs:
            valid = make_valid_mask(r.x, r.bad)
            self.x[r.rec_id] = np.nan_to_num(r.x).astype(np.float32)
            self.scale[r.rec_id] = scale_trace(r.x, valid) if norm == "online" else None
            ix = index_windows(r, win, 25)  # 0.1 s candidates for training
            m = ix.clean & ix.pure & (ix.y >= 0)
            self.pool_ends[r.rec_id], self.pool_y[r.rec_id] = ix.ends[m], ix.y[m].astype(np.float32)
            ix = index_windows(r, win, FS)
            m = ix.clean & (ix.y >= 0)
            self.eval_ends[r.rec_id], self.eval_y[r.rec_id] = ix.ends[m], ix.y[m].astype(np.float32)

    def batch(self, rid, ends, clip):
        idx = ends[:, None] - self.win + 1 + np.arange(self.win)
        w = self.x[rid][idx]
        s = self.scale[rid][ends] if self.scale[rid] is not None else np.full(len(ends), DEFAULT_SCALE, np.float32)
        return prep(w, s, clip)


@torch.no_grad()
def predict(model, store, ids, clip, bs=512):
    model.eval()
    out = {}
    for rid in ids:
        ends = store.eval_ends[rid]
        p = []
        for s in range(0, len(ends), bs):
            X = torch.from_numpy(store.batch(rid, ends[s : s + bs], clip))[:, None]
            p.append(torch.sigmoid(model(X)).numpy())
        out[rid] = np.concatenate(p) if p else np.zeros(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--norm", default="online", choices=["online", "fixed"])
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per_epoch", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--clip", type=float, default=20.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--model_kw", default="{}")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed + a.fold)
    rng = np.random.default_rng(a.seed + a.fold)
    outdir = f"experiments/{a.name}/fold{a.fold}"
    os.makedirs(outdir, exist_ok=True)
    if os.path.exists(f"{outdir}/oof.npz"):
        print("already done")
        return
    sp = load_splits()
    te_ids = sp["folds"][a.fold]
    pool = sorted(sum([f for j, f in enumerate(sp["folds"]) if j != a.fold], []))
    # The inner validation recordings drive early stopping; the outer fold stays untouched.
    val_ids = sorted(rng.choice(pool, 6, replace=False).tolist())
    tr_ids = [r for r in pool if r not in val_ids]
    recs = [r for r in load_all("EEG_dataset") if r.rec_id in set(te_ids + pool)]
    store = Store(recs, int(a.win * FS), a.norm)
    model = build_model(a.model, **json.loads(a.model_kw))
    n_params = sum(p.numel() if hasattr(p, "numel") else p.nelement() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * (a.per_epoch // a.bs), pct_start=0.1)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    start, best, best_ep, t0 = 0, -1.0, -1, time.time()
    ck = f"{outdir}/last.pt"
    if os.path.exists(ck):
        c = torch.load(ck, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        start, best, best_ep = c["epoch"] + 1, c["best"], c["best_ep"]
    for ep in range(start, a.epochs):
        model.train()
        per_rec = a.bs // 8
        tot = 0.0
        n_steps = a.per_epoch // a.bs
        for _ in range(n_steps):
            # Stratified over recordings and classes so noisy long recordings do not dominate.
            rids = rng.choice(tr_ids, 8)
            Xs, ys = [], []
            for rid in rids:
                yy = store.pool_y[rid]
                i1, i0 = np.where(yy == 1)[0], np.where(yy == 0)[0]
                k = rng.choice(i1, per_rec // 2), rng.choice(i0, per_rec - per_rec // 2)
                sel = np.concatenate(k)
                Xs.append(store.batch(rid, store.pool_ends[rid][sel], a.clip))
                ys.append(yy[sel])
            X = np.concatenate(Xs)
            X *= rng.uniform(0.8, 1.25, (len(X), 1)).astype(np.float32)
            logit = model(torch.from_numpy(X)[:, None])
            loss = loss_fn(logit, torch.from_numpy(np.concatenate(ys)))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item()
        pv = predict(model, store, val_ids, a.clip)
        y = np.concatenate([store.eval_y[r] for r in val_ids]); p = np.concatenate([pv[r] for r in val_ids])
        val_auc = roc_auc_score(y, p)
        if val_auc > best:
            best, best_ep = val_auc, ep
            torch.save(model.state_dict(), f"{outdir}/best.pt")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, best=best, best_ep=best_ep), ck)
        print(f"{a.name} f{a.fold} ep{ep} loss {tot / n_steps:.4f} valAUC {val_auc:.4f} best {best:.4f}@{best_ep} {time.time() - t0:.0f}s", flush=True)
        if ep - best_ep >= a.patience:
            break
    model.load_state_dict(torch.load(f"{outdir}/best.pt"))
    pt = predict(model, store, te_ids, a.clip)
    np.savez(f"{outdir}/oof.npz", **{f"p_{r}": pt[r] for r in te_ids}, **{f"y_{r}": store.eval_y[r] for r in te_ids}, **{f"e_{r}": store.eval_ends[r] for r in te_ids})
    json.dump(dict(args=vars(a), best=best, best_ep=best_ep, n_params=n_params, train_s=time.time() - t0, val_ids=val_ids), open(f"{outdir}/info.json", "w"), indent=1)
    y = np.concatenate([store.eval_y[r] for r in te_ids]); p = np.concatenate([pt[r] for r in te_ids])
    print(f"{a.name} f{a.fold} DONE testAUC {roc_auc_score(y, p):.4f} params {n_params}", flush=True)


if __name__ == "__main__":
    main()
