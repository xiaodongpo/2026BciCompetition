#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""独立从新BDF/CSV重建时间、质量掩码和全部滤波输入，核验窗口完整性及X/y一致性。"""
import re
from datetime import datetime, timezone, timedelta
import mne
import numpy as np
from scipy import signal as sp
from common import (
    ROOT,
    SOURCE,
    AUDIT,
    STAGING,
    FINAL,
    TICKS,
    SAMPLE_TICKS,
    rows,
    sha,
    dump,
    channel_headers,
)


def event_ticks(text):
    # 独立解析原CSV：整数秒与7位小数分开，避开浮点/微秒截断。
    assert text.endswith('+08:00')
    seconds = int(datetime.fromisoformat(text[:19] + '+08:00').timestamp())
    fraction = text[19:-6]
    sub = int(fraction[1:].ljust(7, '0')) if fraction else 0
    return seconds * 10_000_000 + sub


def independent_origin(path):
    value = path.stem.split('_')
    dt = datetime.strptime(value[1] + value[2], '%Y%m%d%H%M%S.%f').replace(
        tzinfo=timezone(timedelta(hours=8))
    )
    return int(dt.replace(microsecond=0).timestamp()) * 10_000_000 + dt.microsecond * 10


def raw_trials(path, origin, n):
    result = []
    pending = None
    for event in rows(path):
        if event['event'] == 'trial_start':
            assert pending is None
            pending = event
        elif event['event'] == 'trial_end':
            assert pending is not None and pending['phase'] == event['phase']
            a = event_ticks(pending['local_time']) - origin
            b = event_ticks(event['local_time']) - origin
            assert a < b and b > 0 and a < n * 40_000
            result.append((a, b, {'relax': 0, 'task': 1}[event['phase']]))
            pending = None
    assert pending is None
    assert all(t[1] <= u[0] for t, u in zip(result, result[1:]))
    return result


def main():
    target = STAGING if STAGING.exists() else FINAL
    manifest = __import__('json').loads((target / 'manifest.json').read_text())
    assert manifest['record_count'] == 57 and manifest['raw_source'] == str(SOURCE)
    split = __import__('json').loads((target / 'split.json').read_text())
    assert len(split['train_records']) == 45 and len(split['validation_records']) == 12
    assert not set(split['train_records']) & set(split['validation_records'])
    assert set(split['train_records']) | set(split['validation_records']) == set(
        r['record_id'] for r in manifest['records']
    )
    totals = np.zeros(2, dtype=np.int64)
    counts = {'train': 0, 'validation': 0}
    partial = []
    checked = 0
    for record in manifest['records']:
        rid = record['record_id']
        folder = AUDIT / rid
        bdf = next((SOURCE / rid).glob('*.bdf'))
        csvfile = SOURCE / rid / (rid + '_events.csv')
        assert (
            sha(bdf) == record['source_bdf_sha256']
            and sha(csvfile) == record['source_events_sha256']
        )
        assert sha(folder / 'paired_windows.npz') == record['source_sha256']
        assert (
            sha(target / record['x_file']) == record['x_sha256']
            and sha(target / record['y_file']) == record['y_sha256']
        )
        raw = mne.io.read_raw_bdf(bdf, preload=False, verbose='ERROR')
        source = raw.get_data(units='uV')
        n = source.shape[-1]
        origin = independent_origin(bdf)
        meta = __import__('json').loads((folder / 'metadata.json').read_text())
        assert meta['origin_ticks_100ns'] == origin
        trials = raw_trials(csvfile, origin, n)
        hdr = channel_headers(bdf)
        index = hdr['name'].index('EEG')
        lo, hi = float(hdr['pmin'][index]), float(hdr['pmax'][index])
        quantum = (hi - lo) / (int(hdr['dmax'][index]) - int(hdr['dmin'][index]))
        quality = np.zeros(n, dtype=np.uint8)
        for onset, duration, description in zip(
            raw.annotations.onset, raw.annotations.duration, raw.annotations.description
        ):
            if str(description).upper().startswith('BAD'):
                first = max(0, int(np.floor(onset * 250 + 1e-6)))
                last = min(n, int(np.ceil((onset + duration) * 250 - 1e-6)))
                quality[first:last] |= 1
        quality[~np.isfinite(source).all(axis=0)] |= 2
        quality[
            (
                (np.abs(source - lo) <= quantum / 4)
                | (np.abs(source - hi) <= quantum / 4)
            ).any(axis=0)
        ] |= 4
        equal = (
            np.isfinite(source[0, :-1])
            & np.isfinite(source[0, 1:])
            & (np.abs(source[0, 1:] - source[0, :-1]) <= quantum / 4)
        )
        breaks = np.r_[0, np.flatnonzero(~equal) + 1, n]
        for first, last in zip(breaks, breaks[1:]):
            if last - first >= 250:
                quality[first:last] |= 8
        assert np.array_equal(quality, np.load(folder / 'quality_bits.npy'))
        prefix = np.r_[0, np.cumsum(quality != 0)]
        starts = []
        labels = []
        trial_ids = []
        for tid, (begin, end, label) in enumerate(trials):
            if begin < 0 or end > n * 40_000:
                partial.append(
                    dict(record_id=rid, trial_id=tid, start_ticks=begin, end_ticks=end)
                )
            first = max(0, -(-begin // 10_000_000))
            last = min((n - 1000) // 250, (end - 40_000_000) // 10_000_000)
            for second in range(first, last + 1):
                a = second * 250
                b = a + 1000
                if prefix[b] == prefix[a]:
                    starts.append(a)
                    labels.append(label)
                    trial_ids.append(tid)
        starts = np.array(starts, dtype=np.int64)
        labels = np.array(labels, dtype=np.int64)
        trial_ids = np.array(trial_ids, dtype=np.int16)
        blocks = np.stack([source[:, a : a + 1000].astype(np.float32) for a in starts])
        z = blocks.astype(np.float64)
        sos = sp.butter(4, [1, 50], btype='bandpass', fs=250, output='sos')
        z, _ = sp.sosfilt(
            sos,
            z,
            axis=-1,
            zi=sp.sosfilt_zi(sos)[:, None, None, :] * z[None, ..., 0, None],
        )
        b, a = sp.iirnotch(50, 30, fs=250)
        z, _ = sp.lfilter(
            b, a, z, axis=-1, zi=sp.lfilter_zi(b, a)[None, None, :] * z[..., 0, None]
        )
        z -= z.mean(axis=-1, keepdims=True)
        center = np.median(z, axis=-1, keepdims=True)
        scale = np.maximum(
            1.4826 * np.median(np.abs(z - center), axis=-1, keepdims=True), 1.0
        )
        unclip = (z - center) / scale
        fraction = (np.abs(unclip) > 8).mean(axis=(1, 2))
        model = np.clip(unclip, -8, 8)
        model -= model.mean(axis=-1, keepdims=True)
        keep = np.isfinite(model).all(axis=(1, 2)) & (
            np.ptp(z, axis=-1).max(axis=1) > 1e-5
        )
        starts, labels, trial_ids, blocks, z, model, center, scale, fraction = [
            v[keep]
            for v in [
                starts,
                labels,
                trial_ids,
                blocks,
                z,
                model,
                center,
                scale,
                fraction,
            ]
        ]
        with np.load(folder / 'paired_windows.npz') as d:
            assert np.array_equal(d['start_sample'], starts)
            assert np.array_equal(d['end_sample_exclusive'], starts + 1000)
            assert np.array_equal(d['y'], labels) and np.array_equal(
                d['trial_id'], trial_ids
            )
            assert np.array_equal(d['X_uv'], blocks)
            assert np.array_equal(d['X_filtered_uv'], z.astype(np.float32))
            assert np.array_equal(d['X_model'][:, 0], model.astype(np.float32))
            assert np.array_equal(d['norm_scale_uv'], scale.ravel())
            assert np.array_equal(d['clipped_fraction'], fraction)
        actual_x = np.load(target / record['x_file'])
        actual_y = np.load(target / record['y_file'])
        assert actual_x.dtype == np.float32 and actual_y.dtype == np.int64
        assert np.array_equal(actual_x, model.astype(np.float32)) and np.array_equal(
            actual_y, labels
        )
        assert np.isfinite(actual_x).all()
        origins = rows(folder / 'windows_paired.csv')
        assert [int(r['array_index']) for r in origins] == list(range(len(labels)))
        assert [int(r['start_sample']) for r in origins] == starts.tolist()
        totals += np.bincount(labels, minlength=2)
        counts[record['split']] += len(labels)
        checked += len(labels)
        raw.close()
        print('INDEPENDENT_VERIFY', rid, len(labels), flush=True)
    global_windows = rows(target / 'source_windows_paired.csv')
    assert len(global_windows) == checked == manifest['windows']
    assert totals.tolist() == [
        manifest['class_counts']['relax'],
        manifest['class_counts']['task'],
    ]
    assert counts == manifest['split_windows']
    report = dict(
        status='PASS',
        records=57,
        windows=checked,
        full_raw_and_filtered_and_model_values_exact=True,
        full_npy_array_equality=True,
        window_completeness_independently_verified=True,
        labels_and_trial_boundaries_independently_verified=True,
        all_source_bdf_and_events_sha256_verified=57,
        quality_masks_independently_verified=True,
        partial_trial_recording_boundaries=partial,
        time_basis='filename exact milliseconds and CSV Beijing time; 100ns integers',
        new_split_records=dict(train=45, validation=12),
        split_windows=counts,
        class_counts=manifest['class_counts'],
        verification_script_sha256=sha(__file__),
    )
    dump(target / 'verification.json', report)
    dump(AUDIT / 'verification.json', report)
    print(
        'VERIFIED_ALL',
        checked,
        totals.tolist(),
        counts,
        'partial_trials',
        partial,
        flush=True,
    )


if __name__ == '__main__':
    main()
