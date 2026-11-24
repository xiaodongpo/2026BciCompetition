"""BDF / events.csv loading and sample-accurate label alignment.

Time base (official README): sample n of a file is at
``filename_time + n / 250`` seconds, Beijing time.  Everything downstream works
in integer sample indices on that grid.

Label convention: ``relax = 0``, ``task = 1``, unlabeled = -1.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import mne
import numpy as np
import pandas as pd

FS = 250
UNLABELED = -1
RELAX = 0
TASK = 1
PHASE_TO_LABEL = {"relax": RELAX, "task": TASK}
TZ_BJ = timezone(timedelta(hours=8))
_NAME_RE = re.compile(r"(\d{3})_(\d{8})_(\d{6})\.(\d{3})_UTC\+0800\.bdf$")


@dataclass
class Trial:
    phase: str
    label: int
    start: int  # first sample inside the trial (clipped to the file)
    end: int  # one past the last sample inside the trial (clipped to the file)
    start_raw: int  # unclipped, may be negative or beyond the file
    end_raw: int


@dataclass
class Recording:
    rec_id: str
    bdf_path: str
    csv_path: str
    t0: datetime
    x: np.ndarray  # float32, microvolts, shape (n,)
    bad: np.ndarray  # bool, True inside BAD_ACQ_SKIP
    gaps: list  # [(start_sample, end_sample)] from annotations
    trials: list
    events: pd.DataFrame = field(repr=False, default=None)

    @property
    def n(self) -> int:
        return len(self.x)

    def label_array(self) -> np.ndarray:
        y = np.full(self.n, UNLABELED, dtype=np.int8)
        for t in self.trials:
            y[t.start : t.end] = t.label
        return y


def parse_t0(bdf_path: str) -> tuple[str, datetime]:
    m = _NAME_RE.search(os.path.basename(bdf_path))
    if m is None:
        raise ValueError(f"unexpected BDF file name: {bdf_path}")
    rid, d, t, ms = m.groups()
    dt = datetime.strptime(d + t, "%Y%m%d%H%M%S").replace(tzinfo=TZ_BJ)
    return rid, dt + timedelta(milliseconds=int(ms))


def list_recordings(root: str) -> list[tuple[str, str]]:
    root = os.environ.get("BCI_DATA_ROOT", root)
    bdfs = sorted(glob.glob(os.path.join(root, "*", "*.bdf")))
    out = []
    for b in bdfs:
        csvs = glob.glob(os.path.join(os.path.dirname(b), "*.csv"))
        if len(csvs) != 1:
            raise ValueError(f"expected one csv next to {b}, got {csvs}")
        out.append((b, csvs[0]))
    return out


def read_events(csv_path: str, t0: datetime) -> pd.DataFrame:
    # utf-8-sig: the official CSVs start with a BOM.
    df = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False)
    if list(df.columns) != ["local_time", "event", "phase", "key"]:
        raise ValueError(f"{csv_path}: unexpected columns {list(df.columns)}")
    ts = pd.to_datetime(df["local_time"], format="ISO8601", utc=True)
    t0_utc = pd.Timestamp(t0).tz_convert("UTC")
    df["t_sec"] = (ts - t0_utc).dt.total_seconds()
    df["sample"] = np.round(df["t_sec"].to_numpy() * FS).astype(np.int64)
    return df


def pair_trials(events: pd.DataFrame, n_samples: int) -> tuple[list[Trial], list[str]]:
    """Pair each trial_start with the following trial_end of the same phase.

    Returns the trials plus a list of human readable anomalies; a start without
    a matching end is dropped instead of being guessed.
    """
    trials, issues = [], []
    open_start = None
    for _, r in events.iterrows():
        if r["event"] == "trial_start":
            if open_start is not None:
                issues.append(f"trial_start at {open_start['local_time']} has no trial_end")
            open_start = r
        elif r["event"] == "trial_end":
            if open_start is None:
                issues.append(f"trial_end at {r['local_time']} has no trial_start")
                continue
            if open_start["phase"] != r["phase"]:
                issues.append(f"phase mismatch at {r['local_time']}")
                open_start = None
                continue
            s, e = int(open_start["sample"]), int(r["sample"])
            if e <= s:
                issues.append(f"non-positive trial at {r['local_time']}")
            elif r["phase"] not in PHASE_TO_LABEL:
                issues.append(f"unknown phase {r['phase']}")
            else:
                cs, ce = max(0, s), min(n_samples, e)
                if s < 0 or e > n_samples:
                    issues.append(
                        f"trial {r['phase']} exceeds file bounds by "
                        f"{max(0, -s) / FS:.3f}s/{max(0, e - n_samples) / FS:.3f}s"
                    )
                if ce > cs:
                    trials.append(Trial(r["phase"], PHASE_TO_LABEL[r["phase"]], cs, ce, s, e))
            open_start = None
    if open_start is not None:
        issues.append(f"trial_start at {open_start['local_time']} has no trial_end (end of file)")
    return trials, issues


def read_bdf(bdf_path: str) -> tuple[np.ndarray, list, dict]:
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")
    if raw.ch_names != ["EEG"] or abs(raw.info["sfreq"] - FS) > 1e-6:
        raise ValueError(f"{bdf_path}: unexpected {raw.ch_names} @ {raw.info['sfreq']}")
    # mne scales the file's uV to V; go back to uV.
    x = (raw.get_data()[0] * 1e6).astype(np.float32)
    gaps = []
    for desc, onset, dur in zip(raw.annotations.description, raw.annotations.onset, raw.annotations.duration):
        if desc == "BAD_ACQ_SKIP":
            s = int(round(onset * FS))
            gaps.append((s, min(len(x), s + int(round(dur * FS)))))
    ex = raw._raw_extras[0]
    meta = {
        "physical_min": float(np.ravel(ex["physical_min"])[0]),
        "physical_max": float(np.ravel(ex["physical_max"])[0]),
        "digital_min": float(np.ravel(ex["digital_min"])[0]),
        "digital_max": float(np.ravel(ex["digital_max"])[0]),
        "units": raw._orig_units.get("EEG"),
        "meas_date": str(raw.info["meas_date"]),
        "annotations": list(raw.annotations.description),
    }
    return x, gaps, meta


def load_recording(bdf_path: str, csv_path: str) -> Recording:
    rid, t0 = parse_t0(bdf_path)
    x, gaps, _ = read_bdf(bdf_path)
    bad = np.zeros(len(x), dtype=bool)
    for s, e in gaps:
        bad[s:e] = True
    ev = read_events(csv_path, t0)
    trials, _ = pair_trials(ev, len(x))
    return Recording(rid, bdf_path, csv_path, t0, x, bad, gaps, trials, ev)


def load_all(root: str, cache: str | None = "cache/recordings") -> list[Recording]:
    """Load every recording; caches the decoded signal since BDF parsing is slow."""
    recs = []
    for b, c in list_recordings(root):
        rid, t0 = parse_t0(b)
        if cache:
            os.makedirs(cache, exist_ok=True)
            p = os.path.join(cache, f"{rid}.npz")
            if os.path.exists(p):
                z = np.load(p)
                x, gaps = z["x"], [tuple(g) for g in z["gaps"].tolist()]
            else:
                x, gaps, _ = read_bdf(b)
                np.savez(p, x=x, gaps=np.array(gaps, dtype=np.int64).reshape(-1, 2))
        else:
            x, gaps, _ = read_bdf(b)
        bad = np.zeros(len(x), dtype=bool)
        for s, e in gaps:
            bad[s:e] = True
        ev = read_events(c, t0)
        trials, _ = pair_trials(ev, len(x))
        recs.append(Recording(rid, b, c, t0, x, bad, gaps, trials, ev))
    return recs
