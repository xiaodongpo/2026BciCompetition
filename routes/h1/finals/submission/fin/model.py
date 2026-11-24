"""Prediction model for the finals: one 6-s window + ten 30-s reference segments -> P(task).

All parameters are fixed files loaded once. The reference EEG is used only to normalise the
window's features (percentile rank or z-score against the reference windows). Nothing is kept
between calls except a cache of the reference statistics, which is a pure function of the
reference array, so a prediction never depends on call order or on other windows.
"""
from __future__ import annotations

import hashlib
import json
import os

import lightgbm as lgb
import numpy as np

from .features import extract_multi
from .quality import invalid_mask, longest_clean_run

FS, WIN, REF_STEP = 250, 1500, 250


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class FinalsModel:
    def __init__(self, model_dir: str):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.cfg = cfg
        self.views = tuple(cfg["windows"])  # e.g. (1500, 500): whole window and its last 2 s
        self.mu, self.sd = np.array(cfg["mu"]), np.array(cfg["sd"])
        self.min_ref = int(cfg.get("min_ref", 20))
        self.min_len = int(cfg.get("min_len", 750))
        self.fallback = float(cfg.get("fallback", 0.5))
        self.members = []
        for m in cfg["members"]:
            mem = dict(view=m["view"], weight=float(m["weight"]))
            if m["view"] in ("rank", "abs"):
                mem["boosters"] = [lgb.Booster(model_file=os.path.join(model_dir, f)) for f in m["models"]]
                mem["lr"] = None if m.get("lr") is None else {k: np.array(v) if isinstance(v, list) else v for k, v in m["lr"].items()}
                if m["view"] == "abs":
                    mem["mu"], mem["sd"], mem["bpn"] = np.array(m["mu"]), np.array(m["sd"]), bool(m.get("bpn"))
            elif m["view"] == "hybrid":
                import torch

                from .net import HybridNet

                torch.set_num_threads(1)
                mem["wave"] = m["wave"]
                mem["nets"] = []
                for f in m["nets"]:
                    net = HybridNet(len(self.mu))
                    net.load_state_dict(torch.load(os.path.join(model_dir, f), map_location="cpu", weights_only=True))
                    mem["nets"].append(net.eval())
            self.members.append(mem)
        self._ref_key, self._ref = None, None

    # ---- reference: features of its clean 6-s windows (1-s step inside each 30-s segment) ----
    def _reference(self, reference: np.ndarray) -> dict:
        ref = np.ascontiguousarray(reference, dtype=np.float64)
        key = hashlib.blake2b(ref.tobytes(), digest_size=16).digest()
        if key != self._ref_key:
            wins = []
            for seg in ref:
                bad = invalid_mask(seg)
                c = np.concatenate([[0], np.cumsum(bad)])
                for s in range(0, len(seg) - WIN + 1, REF_STEP):
                    if c[s + WIN] - c[s] == 0:
                        wins.append(seg[s : s + WIN])
            if len(wins) >= self.min_ref:
                F = np.nan_to_num(extract_multi(np.stack(wins), self.views)).astype(np.float64)
                stats = dict(n=len(F), sorted=np.sort(F, axis=0), mean=F.mean(0), sd=F.std(0) + 1e-6)
            else:
                stats = dict(n=0)
            self._ref_key, self._ref = key, stats
        return self._ref

    def _lgb(self, mem, x):
        p = float(np.mean([b.predict(x[None, :], num_threads=1)[0] for b in mem["boosters"]]))
        if mem["lr"] is not None:
            z = (x - mem["lr"]["mean"]) / mem["lr"]["scale"]
            p = 0.5 * p + 0.5 * float(_sigmoid(z @ mem["lr"]["coef"] + mem["lr"]["intercept"]))
        return p

    def predict(self, window: np.ndarray, reference: np.ndarray) -> float:
        w = np.asarray(window, dtype=np.float64).ravel()
        bad = invalid_mask(w)
        a, b = (0, len(w)) if not bad.any() else longest_clean_run(bad)
        if b - a < self.min_len:
            return self.fallback  # no usable stretch of at least 3 s in this window
        seg = w[a:b]
        full = (b - a) == WIN
        ref = self._reference(reference)
        f = np.nan_to_num(extract_multi(seg[None, :], self.views)[0]).astype(np.float64)
        if ref["n"]:
            rank = (ref["sorted"] < f).sum(axis=0) / ref["n"]
            z = np.clip((f - ref["mean"]) / ref["sd"], -8, 8)
        else:  # reference unusable: fall back to the training prior
            rank = 0.5 + 0.5 * np.tanh((f - self.mu) / self.sd)
            z = np.clip((f - self.mu) / self.sd, -8, 8)
        num = den = 0.0
        for mem in self.members:
            if mem["view"] == "rank":
                p = self._lgb(mem, rank)
            elif mem["view"] == "abs":
                g = f
                if mem["bpn"]:
                    from .net import bandpass_notch

                    g = np.nan_to_num(extract_multi(bandpass_notch(seg[None, :] - seg[0]), self.views)[0]).astype(np.float64)
                p = self._lgb(mem, 0.5 + 0.5 * np.tanh((g - mem["mu"]) / mem["sd"]))
            else:
                if not full:
                    continue  # the network needs the complete 6 s
                import torch

                from .net import wave_input

                with torch.no_grad():
                    raw = torch.from_numpy(wave_input(seg[None, :], mem["wave"]))
                    zz = torch.from_numpy(z[None, :].astype(np.float32))
                    p = float(_sigmoid(np.mean([net(raw, zz).item() for net in mem["nets"]])))
            num += mem["weight"] * p
            den += mem["weight"]
        return float(min(1.0, max(0.0, num / den)))
