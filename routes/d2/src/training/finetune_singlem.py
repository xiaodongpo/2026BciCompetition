"""Fine-tune SingLEM on raw 6-s windows. Strategies: partial (last K transformer layers) or lora."""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.data.io import FS, load_all
from src.data.windows import index_windows
from src.evaluation.splits import load_splits
from src.models.singlem_official import load_official
from src.preprocessing.online import DEFAULT_SCALE, scale_trace
from src.preprocessing.quality import make_valid_mask
from src.preprocessing.singlem_prep import prep_tokens


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 8.0, dropout: float = 0.1):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.drop = nn.Dropout(dropout)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.B(self.A(self.drop(x))) * self.scale


def inject_lora(enc, r, alpha, dropout) -> None:
    """In the official EEGEncoder the global layers are TransformerEncoder.Transformer_layers[i]
    with self_att_heads.concat_attn / concat_proj and mlp.W_1 / W_2."""
    for layer in enc.TransformerEncoder.Transformer_layers:
        a = layer.self_att_heads
        a.concat_attn = LoRALinear(a.concat_attn, r, alpha, dropout)
        a.concat_proj = LoRALinear(a.concat_proj, r, alpha, dropout)
        layer.mlp.W_2 = LoRALinear(layer.mlp.W_2, r, alpha, dropout)


class Net(nn.Module):
    def __init__(self, enc, tokens=6, rep=16, dropout=0.3):
        super().__init__()
        self.enc = enc
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(tokens * rep, 1))

    def forward(self, x):
        return self.head(self.enc(x).flatten(1)).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--strategy", choices=["partial", "lora", "scratch"], required=True)
    ap.add_argument("--unfreeze", type=int, default=2)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--per_epoch", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default="pretrained/singlem/SingLEM/checkpoints/singlem_downstream_included.pt")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.fold)
    rng = np.random.default_rng(a.fold)
    out = f"experiments/{a.name}/fold{a.fold}"
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/oof.npz"):
        return
    sp = load_splits()
    te = sp["folds"][a.fold]
    pool = sorted(sum([f for j, f in enumerate(sp["folds"]) if j != a.fold], []))
    val = sorted(rng.choice(pool, 6, replace=False).tolist())
    tr = [r for r in pool if r not in val]
    WIN = 6 * FS
    D = {}
    for r in load_all("EEG_dataset"):
        if r.rec_id not in set(te + pool):
            continue
        valid = make_valid_mask(r.x, r.bad)
        ix = index_windows(r, WIN, 25)
        m = ix.clean & ix.pure & (ix.y >= 0)
        ie = index_windows(r, WIN, 2 * FS)
        me = ie.clean & (ie.y >= 0)
        D[r.rec_id] = dict(x=np.nan_to_num(r.x).astype(np.float32), s=scale_trace(r.x, valid), e=ix.ends[m], y=ix.y[m].astype(np.float32), ee=ie.ends[me], ey=ie.y[me].astype(np.float32))

    def tokens(rid, ends):
        d = D[rid]
        idx = ends[:, None] - WIN + 1 + np.arange(WIN)
        w = d["x"][idx] * (DEFAULT_SCALE / d["s"][ends])[:, None]  # online amplitude normalization
        return torch.from_numpy(prep_tokens(w))

    wrapper = load_official(a.ckpt)  # official repository code
    enc = wrapper.encoder
    for p in enc.parameters():
        p.requires_grad = False
    if a.strategy == "partial":
        for layer in enc.TransformerEncoder.Transformer_layers[-a.unfreeze :]:
            for p in layer.parameters():
                p.requires_grad = True
        for mod in (enc.TransformerEncoder.ln_final, enc.DimRedLayer):
            for p in mod.parameters():
                p.requires_grad = True
    elif a.strategy == "lora":
        inject_lora(enc, a.rank, a.alpha, 0.1)
        for p in enc.DimRedLayer.parameters():
            p.requires_grad = True
    else:
        for p in enc.parameters():
            p.requires_grad = True
        enc.apply(lambda m: m.reset_parameters() if isinstance(m, (nn.Linear, nn.Conv1d, nn.LayerNorm)) else None)
    model = Net(wrapper)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train, n_all = sum(p.nelement() for p in params), sum(p.nelement() for p in model.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    steps = a.per_epoch // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps, pct_start=0.1)
    best, t0 = -1.0, time.time()

    @torch.no_grad()
    def pred(ids):
        model.eval()
        res = {}
        for rid in ids:
            e = D[rid]["ee"]
            res[rid] = np.concatenate([torch.sigmoid(model(tokens(rid, e[i : i + 128]))).numpy() for i in range(0, len(e), 128)])
        return res

    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for _ in range(steps):
            xs, ys = [], []
            for rid in rng.choice(tr, 8):
                d = D[rid]
                i1, i0 = np.where(d["y"] == 1)[0], np.where(d["y"] == 0)[0]
                k = a.bs // 16
                sel = np.concatenate([rng.choice(i1, k), rng.choice(i0, k)])
                xs.append(tokens(rid, d["e"][sel]))
                ys.append(d["y"][sel])
            loss = nn.functional.binary_cross_entropy_with_logits(model(torch.cat(xs)), torch.from_numpy(np.concatenate(ys)))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)
            opt.step()
            sched.step()
            tot += loss.item()
        pv = pred(val)
        v = roc_auc_score(np.concatenate([D[r]["ey"] for r in val]), np.concatenate([pv[r] for r in val]))
        if v > best:
            best = v
            torch.save({k: t for k, t in model.state_dict().items()}, f"{out}/best.pt")
        print(f"{a.name} f{a.fold} ep{ep} loss {tot / steps:.4f} val {v:.4f} best {best:.4f} {time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(torch.load(f"{out}/best.pt"))
    pt = pred(te)
    np.savez(f"{out}/oof.npz", **{f"p_{r}": pt[r] for r in te}, **{f"y_{r}": D[r]["ey"] for r in te}, **{f"e_{r}": D[r]["ee"] for r in te})
    os.remove(f"{out}/best.pt") if n_all > 3_000_000 and a.strategy == "scratch" else None
    info = dict(args={**vars(a), "model": f"singlem_{a.strategy}", "norm": "online", "seed": a.fold}, best=best, n_params=n_train, n_all=n_all, train_s=time.time() - t0, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    json.dump(info, open(f"{out}/info.json", "w"), indent=1)
    y = np.concatenate([D[r]["ey"] for r in te])
    p = np.concatenate([pt[r] for r in te])
    print(f"{a.name} f{a.fold} DONE test {roc_auc_score(y, p):.4f} trainable {n_train}/{n_all} peak {info['peak_rss_gb']:.2f}GB {info['train_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
