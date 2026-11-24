"""Strictly causal streaming decoder.

Contract
--------
* ``push(chunk)`` takes consecutive samples (microvolts, 250 Hz, NaN for missing)
  of any chunk size and returns one Decision per completed decision hop.
* A decision at sample t depends on
    (a) at most the last 1500 samples up to and including t (6 s), and
    (b) for members with the "rank" view, percentile histories of features of
        *earlier* windows of the same recording (the online adaptation the rules allow).
  The model is a weighted average of members; "abs" members and ``adapt=False``
  are pure functions of the last 6 s.
* Decisions land on a fixed grid, and histories update only on that grid, so
  outputs do not depend on how the stream is cut into chunks.

Score: ``p_task`` in [0, 1]; higher means more likely task. relax=0, task=1.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import lightgbm as lgb
import numpy as np

from src.preprocessing.features import extract_multi
from src.preprocessing.quality import invalid_mask
from src.preprocessing.rank_norm import OnlineRank

FS = 250
WIN = 1500
LABELS = {0: "relax", 1: "task"}


@dataclass
class Decision:
    sample_index: int  # index of the newest sample used
    time_s: float
    p_task: float
    state: int
    label: str
    evaluated: bool  # False: the score was held because the sample/window was unusable
    window_samples: int  # samples the model saw; 0 when the score was held
    reason: str


class _Member:
    """One scorer: LightGBM models (+ optional logistic regression) on one feature view.

    view "rank": percentile of each feature within the recording's history (online adaptation).
    view "abs":  features squashed with the training prior only (a pure function of the window).
    """

    def __init__(self, model_dir: str, spec: dict):
        self.view = spec["view"]
        self.weight = float(spec.get("weight", 1.0))
        self.boosters = [lgb.Booster(model_file=os.path.join(model_dir, m)) for m in spec["models"]]
        self.columns = spec.get("columns")
        lr = spec.get("lr")
        self.lr = None if lr is None else {k: np.array(v) if isinstance(v, list) else v for k, v in lr.items()}

    def score(self, x: np.ndarray) -> float:
        if self.columns is not None:
            x = x[self.columns]
        p = float(np.mean([b.predict(x[None, :], num_threads=1)[0] for b in self.boosters]))
        if self.lr is not None:
            z = (x - self.lr["mean"]) / self.lr["scale"]
            p_lr = float(1.0 / (1.0 + np.exp(-(z @ self.lr["coef"] + self.lr["intercept"]))))
            p = (1 - self.lr["weight"]) * p + self.lr["weight"] * p_lr
        return p


class StreamingDecoder:
    def __init__(self, model_dir: str, hop: int = FS, min_window: int = 3 * FS, adapt: bool | None = None, threshold: float | None = None):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.cfg = cfg
        self.hop, self.min_window = int(hop), int(min_window)
        if "members" not in cfg:  # single-member format of earlier exports
            view = "rank" if cfg.get("adapt", True) else "abs"
            cfg = {**cfg, "members": [dict(view=view, models=cfg["models"], columns=cfg.get("columns"), lr=cfg.get("lr"), weight=1.0)]}
        self.members = [_Member(model_dir, m) for m in cfg["members"]]
        total = sum(m.weight for m in self.members)
        for m in self.members:
            m.weight /= total
        uses_history = any(m.view == "rank" for m in self.members)
        # adapt=False turns online adaptation off: rank views fall back to the prior view.
        self.adapt = uses_history if adapt is None else (adapt and uses_history)
        self.threshold = cfg.get("threshold", 0.5) if threshold is None else threshold
        self.mu = np.array(cfg["mu"], float)
        self.sd = np.array(cfg["sd"], float)
        # Feature views, newest-sample aligned, e.g. [1500, 500] = 6-s and last-2-s windows.
        self.windows = tuple(int(n) for n in cfg.get("windows", [WIN]))
        assert max(self.windows) <= WIN
        self.reset()

    def reset(self) -> None:
        """Forget everything; call at the start of each new recording."""
        self.buf = np.empty(0, dtype=np.float64)
        self.total = 0
        self.last_p = 0.5
        self.rank = OnlineRank(self.mu, self.sd)

    def _predict(self, f: np.ndarray) -> float:
        prior = 0.5 + 0.5 * np.tanh((np.nan_to_num(f) - self.mu) / self.sd)
        rank = self.rank.transform(f) if self.adapt else prior
        return float(sum(m.weight * m.score(rank if m.view == "rank" else prior) for m in self.members))

    def _decide(self, update: bool = True) -> Decision:
        t = self.total - 1
        w = self.buf
        bad = invalid_mask(w)
        evaluated, used, reason = False, 0, "held"
        if not bad[-1]:
            if bad.any():
                last_bad = int(np.flatnonzero(bad)[-1])
                seg = w[last_bad + 1 :]
                reason = "gap_in_window"
            elif len(w) < WIN:
                seg = w
                reason = "warmup"
            else:
                seg = w
                reason = "ok"
            if len(seg) >= self.min_window:
                f = extract_multi(seg[None, :], self.windows)[0]
                p = self._predict(f)
                if update:
                    self.last_p = p
                evaluated, used = True, len(seg)
                if update and len(seg) == WIN and self.adapt:
                    self.rank.update(f)
                state = int(p >= self.threshold)
                return Decision(t, t / FS, p, state, LABELS[state], True, used, reason)
            else:
                reason += "_too_short"
        else:
            reason = "sample_invalid"
        state = int(self.last_p >= self.threshold)
        return Decision(t, t / FS, self.last_p, state, LABELS[state], evaluated, used, reason)

    def push(self, chunk) -> list[Decision]:
        """Feed consecutive samples; returns the decisions completed by them, in time order."""
        chunk = np.asarray(chunk, dtype=np.float64).ravel()
        out: list[Decision] = []
        i = 0
        while i < len(chunk):
            # Next decision falls on sample index k*hop + hop - 1.
            need = self.hop - (self.total % self.hop)
            take = min(need, len(chunk) - i)
            self.buf = np.concatenate([self.buf, chunk[i : i + take]])[-WIN:]
            self.total += take
            i += take
            if self.total % self.hop == 0:
                out.append(self._decide())
        return out

    def process_chunk(self, chunk) -> Decision:
        """One result per call, for callers that want a decision at the end of every chunk.

        Grid decisions inside the chunk still update the adaptation history, so the
        history never depends on the chunking. An end-of-chunk decision off the grid
        reads the state without changing it.
        """
        decisions = self.push(chunk)
        if decisions and decisions[-1].sample_index == self.total - 1:
            return decisions[-1]
        return self._decide(update=False)


def decisions_to_rows(decisions: list[Decision]) -> list[dict]:
    return [asdict(d) for d in decisions]
