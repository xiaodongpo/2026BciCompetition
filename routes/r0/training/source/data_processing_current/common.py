# -*- coding: utf-8 -*-
"""新数据的时间、BDF量程和输出工具；时间计算用100纳秒整数，不丢CSV第7位小数。"""
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_paths import WORK_DIR as ROOT, RAW_DATA as SOURCE
AUDIT = ROOT / 'processing_current'
STAGING = ROOT / 'prepared_1_50hz_new_staging'
FINAL = ROOT / 'prepared_1_50hz'
LOCAL = timezone(timedelta(hours=8))
TICKS = 10_000_000
SAMPLE_TICKS = 40_000


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    Path(p).write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=lambda v: v.item() if isinstance(v, np.generic) else str(v),
        ),
        encoding='utf-8',
    )


def rows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def table(p, data, fields=None):
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(
            f, fieldnames=fields or (list(data[0]) if data else ['record_id'])
        )
        w.writeheader()
        w.writerows(data)


def time_ticks(value):
    match = re.fullmatch(
        r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,7}))?([+-]\d\d:\d\d|Z)?',
        value.strip(),
    )
    assert match, value
    base, fraction, zone = match.groups()
    assert zone in [None, '+08:00'], ('CSV timezone', value)
    dt = datetime.fromisoformat(base + (zone or '+08:00'))
    return int(dt.timestamp()) * TICKS + int((fraction or '').ljust(7, '0'))


def ticks_iso(tick):
    seconds, part = divmod(int(tick), TICKS)
    return (
        datetime.fromtimestamp(seconds, LOCAL).strftime('%Y-%m-%dT%H:%M:%S')
        + f'.{part:07d}+08:00'
    )


def filename_origin(p):
    match = re.fullmatch(r'(\d{3})_(\d{8})_(\d{6})\.(\d{3})_UTC\+0800\.bdf', p.name)
    assert match, p.name
    rid, date, clock, ms = match.groups()
    assert rid == p.parent.name
    dt = datetime.strptime(date + clock, '%Y%m%d%H%M%S').replace(tzinfo=LOCAL)
    return int(dt.timestamp()) * TICKS + int(ms) * 10000


def channel_headers(path):
    with open(path, 'rb') as f:
        fixed = f.read(256)
        n = int(fixed[252:256])
        content = f.read(n * 256)
    values = {}
    offset = 0
    for key, width in [
        ('name', 16),
        ('transducer', 80),
        ('unit', 8),
        ('pmin', 8),
        ('pmax', 8),
        ('dmin', 8),
        ('dmax', 8),
        ('prefilter', 80),
        ('samples', 8),
        ('reserved', 32),
    ]:
        values[key] = [
            content[offset + i * width : offset + (i + 1) * width]
            .decode('latin1')
            .strip()
            for i in range(n)
        ]
        offset += n * width
    return values


def intervals(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
