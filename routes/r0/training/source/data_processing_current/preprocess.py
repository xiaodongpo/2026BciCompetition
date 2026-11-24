#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从更新BDF/CSV重新对齐、质量筛选和1–50Hz预处理，输出待核验NPY与完整追溯资料。"""
import math
import sys
from collections import Counter
from datetime import datetime
import mne
import numpy as np
from scipy.signal import welch
from common import *

sys.path.insert(0, str(ROOT / 'data_processing_1_50hz'))
from quality_preprocess_1_50hz import transform, CONFIG


def parse_trials(events, origin, n, rid):
    trials = []
    issues = []
    pending = None
    for event in events:
        if event['event'] == 'trial_start':
            if pending is not None:
                issues.append(
                    dict(
                        record_id=rid,
                        reason='unmatched_start',
                        csv_row=pending['csv_row'],
                    )
                )
            pending = event
        elif event['event'] == 'trial_end':
            if pending is None:
                issues.append(
                    dict(
                        record_id=rid, reason='unmatched_end', csv_row=event['csv_row']
                    )
                )
                continue
            start = pending
            pending = None
            a, b = start['relative_ticks'], event['relative_ticks']
            reasons = []
            if start['phase'] not in ['relax', 'task']:
                reasons.append('unknown_phase')
            if start['phase'] != event['phase']:
                reasons.append('phase_mismatch')
            if not a < b:
                reasons.append('nonpositive_interval')
            elif b <= 0 or a >= n * SAMPLE_TICKS:
                reasons.append('no_recorded_samples_in_trial')
            trials.append(
                dict(
                    record_id=rid,
                    trial_id=len(trials),
                    phase=start['phase'],
                    label={'relax': 0, 'task': 1}.get(start['phase'], -1),
                    start_ticks=a,
                    end_ticks=b,
                    start_seconds=a / TICKS,
                    end_seconds=b / TICKS,
                    duration_seconds=(b - a) / TICKS,
                    start_sample=max(0, (a + SAMPLE_TICKS - 1) // SAMPLE_TICKS),
                    end_sample_exclusive=min(n, (b + SAMPLE_TICKS - 1) // SAMPLE_TICKS),
                    partial_recording_coverage=(a < 0 or b > n * SAMPLE_TICKS),
                    start_csv_row=start['csv_row'],
                    end_csv_row=event['csv_row'],
                    valid=not reasons,
                    invalid_reason=';'.join(reasons),
                )
            )
    if pending is not None:
        issues.append(
            dict(record_id=rid, reason='unmatched_start', csv_row=pending['csv_row'])
        )
    for i, t in enumerate(trials):
        for u in trials[i + 1 :]:
            if max(t['start_ticks'], u['start_ticks']) < min(
                t['end_ticks'], u['end_ticks']
            ):
                for item in [t, u]:
                    item['valid'] = False
                    item['invalid_reason'] += ';overlap'
    for t in trials:
        if not t['valid']:
            issues.append(
                dict(
                    record_id=rid,
                    reason=t['invalid_reason'],
                    csv_row=t['start_csv_row'],
                )
            )
    return trials, issues


def features(windows):
    z = np.asarray(windows[:, 0], dtype=np.float64)
    frequencies, power = welch(z, fs=250, nperseg=1000, detrend='constant', axis=-1)
    energy = (
        lambda lo, hi: power[:, (frequencies >= lo) & (frequencies <= hi)].sum(axis=-1)
        * 0.25
    )
    brain = np.maximum(energy(1, 40), 1e-20)
    tt = np.arange(1000) / 250
    tt -= tt.mean()
    return dict(
        dc_uv=z.mean(axis=-1),
        std_uv=z.std(axis=-1),
        slope_uv_per_s=z @ tt / (tt @ tt),
        line_ratio=energy(49, 51) / brain,
        slow_ratio=energy(0.25, 0.75) / brain,
    )


def process(record):
    rid = record.name
    destination = AUDIT / rid
    destination.mkdir()
    bdfs = list(record.glob('*.bdf'))
    assert len(bdfs) == 1
    bdf = bdfs[0]
    events_path = record / (rid + '_events.csv')
    origin = filename_origin(bdf)
    raw = mne.io.read_raw_bdf(bdf, preload=False, verbose='ERROR')
    assert raw.ch_names == ['EEG'] and np.isclose(
        raw.info['sfreq'], 250, rtol=0, atol=1e-8
    )
    assert raw.first_samp == 0
    n = int(raw.n_times)
    # 库把BDF头的墙上时间标为UTC，但新文件说明规定它实际上是北京时间。
    # 原点采用文件名毫秒时间，并与T0注释和BDF头的日历字段交叉检查。
    t0 = [str(d) for d in raw.annotations.description if str(d).startswith('T0_')]
    filename_t0 = (
        'T0_' + bdf.name.split('_')[1] + 'T' + bdf.name.split('_')[2] + '+0800'
    )
    assert t0 == [filename_t0], (rid, t0, filename_t0)
    expected_header = datetime.fromtimestamp(origin // TICKS, LOCAL).replace(
        tzinfo=None, microsecond=0
    )
    assert raw.info['meas_date'].replace(tzinfo=None, microsecond=0) == expected_header
    source_events = rows(events_path)
    events = []
    for line, event in enumerate(source_events, 2):
        tick = time_ticks(event['local_time'])
        assert event['event'] in [
            'trial_start',
            'trial_end',
            'key_press',
            'task_failure',
        ]
        events.append(
            dict(
                record_id=rid,
                csv_row=line,
                local_time=event['local_time'],
                relative_ticks=tick - origin,
                relative_seconds=(tick - origin) / TICKS,
                event=event['event'],
                phase=event['phase'],
                key=event['key'],
            )
        )
    assert all(
        a['relative_ticks'] <= b['relative_ticks'] for a, b in zip(events, events[1:])
    ), rid
    trials, issues = parse_trials(events, origin, n, rid)
    table(destination / 'events_aligned.csv', events)
    table(destination / 'trials.csv', trials)
    y = np.full(n, -1, dtype=np.int8)
    tids = np.full(n, -1, dtype=np.int16)
    for t in trials:
        if t['valid']:
            a, b = t['start_sample'], t['end_sample_exclusive']
            y[a:b] = t['label']
            tids[a:b] = t['trial_id']
    signal = raw.get_data(units='uV')
    header = channel_headers(bdf)
    c = header['name'].index('EEG')
    assert header['unit'][c] in ['uV', '\xb5V']
    pmin, pmax = float(header['pmin'][c]), float(header['pmax'][c])
    quantum = (pmax - pmin) / (int(header['dmax'][c]) - int(header['dmin'][c]))
    bits = np.zeros(n, dtype=np.uint8)
    bad = []
    for onset, duration, desc in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        if not str(desc).upper().startswith('BAD'):
            continue
        assert duration > 0, (rid, str(desc), duration)
        # 向外取整覆盖与缺失标记相交的采样点；微小浮点误差按采样网格容差处理。
        a = max(0, math.floor(float(onset) * 250 + 1e-6))
        b = min(n, math.ceil((float(onset) + float(duration)) * 250 - 1e-6))
        if a < b:
            bits[a:b] |= 1
        bad.append(
            dict(
                record_id=rid,
                start_sample=a,
                end_sample_exclusive=b,
                onset_seconds=float(onset),
                duration_seconds=float(duration),
                description=str(desc),
            )
        )
    bits[~np.isfinite(signal).all(axis=0)] |= 2
    rail = (
        (np.abs(signal - pmin) <= quantum * 0.25)
        | (np.abs(signal - pmax) <= quantum * 0.25)
    ).any(axis=0)
    bits[rail] |= 4
    same = (
        np.isfinite(signal[0, :-1])
        & np.isfinite(signal[0, 1:])
        & (np.abs(np.diff(signal[0])) <= quantum * 0.25)
    )
    starts = np.r_[0, np.flatnonzero(~same) + 1]
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts, ends):
        if b - a >= 250:
            bits[a:b] |= 8
    clean_y = y.copy()
    clean_y[(bits & 14) != 0] = -3
    clean_y[(bits & 1) != 0] = -2
    np.save(destination / 'sample_labels.npy', y, allow_pickle=False)
    np.save(destination / 'sample_labels_clean.npy', clean_y, allow_pickle=False)
    np.save(destination / 'sample_trial_ids.npy', tids, allow_pickle=False)
    np.save(destination / 'quality_bits.npy', bits, allow_pickle=False)
    table(
        destination / 'bad_intervals.csv',
        bad,
        [
            'record_id',
            'start_sample',
            'end_sample_exclusive',
            'onset_seconds',
            'duration_seconds',
            'description',
        ],
    )
    quality_intervals = []
    for flag, name in [
        (1, 'BAD_ACQ_SKIP'),
        (2, 'nonfinite'),
        (4, 'ADC_saturation'),
        (8, 'flatline_ge_1s'),
    ]:
        for a, b in intervals((bits & flag) != 0):
            quality_intervals.append(
                dict(
                    record_id=rid,
                    reason=name,
                    start_sample=int(a),
                    end_sample_exclusive=int(b),
                    duration_seconds=(b - a) / 250,
                )
            )
    table(
        destination / 'quality_intervals.csv',
        quality_intervals,
        [
            'record_id',
            'reason',
            'start_sample',
            'end_sample_exclusive',
            'duration_seconds',
        ],
    )
    accepted = []
    rejected = []
    blocks = []
    eligible = 0
    for grid_index, a in enumerate(range(0, n - 1000 + 1, 250)):
        b = a + 1000
        tid = int(tids[a])
        label = int(y[a])
        reasons = []
        phase_ok = tid >= 0 and np.all(tids[a:b] == tid) and np.all(y[a:b] == label)
        if phase_ok:
            trial = trials[tid]
            phase_ok = (
                a * SAMPLE_TICKS >= trial['start_ticks']
                and b * SAMPLE_TICKS <= trial['end_ticks']
            )
        if phase_ok:
            eligible += 1
        else:
            reasons.append('unlabeled_or_cross_phase_boundary')
        flags = int(np.bitwise_or.reduce(bits[a:b]))
        for flag, name in [
            (1, 'BAD_ACQ_SKIP'),
            (2, 'nonfinite'),
            (4, 'ADC_saturation'),
            (8, 'flatline_ge_1s'),
        ]:
            if flags & flag:
                reasons.append(name)
        row = dict(
            record_id=rid,
            grid_window_index=grid_index,
            trial_id=tid,
            label=label if phase_ok else -1,
            start_sample=a,
            end_sample_exclusive=b,
            start_seconds=a / 250,
            decision_seconds=b / 250,
            decision_beijing=ticks_iso(origin + b * SAMPLE_TICKS),
        )
        if reasons:
            rejected.append(dict(row, exclude_reason=';'.join(reasons)))
            continue
        accepted.append(row)
        blocks.append(signal[:, a:b].astype(np.float32))
    assert accepted, (rid, 'no clean labeled windows')
    x = np.stack(blocks)
    physical, model, centers, scales, fractions, keep = transform(x)
    before_count = len(x)
    for row, ok in zip(accepted, keep):
        if not ok:
            rejected.append(dict(row, exclude_reason='second_qc_nonfinite_or_constant'))
    accepted = [row for row, ok in zip(accepted, keep) if ok]
    x, physical, model, centers, scales, fractions = [
        a[keep] for a in [x, physical, model, centers, scales, fractions]
    ]
    yout = np.array([r['label'] for r in accepted], dtype=np.int64)
    start_array = np.array([r['start_sample'] for r in accepted], dtype=np.int64)
    trial_array = np.array([r['trial_id'] for r in accepted], dtype=np.int16)
    assert x.shape == physical.shape == (len(yout), 1, 1000)
    assert model.shape == (len(yout), 1, 1, 1000) and np.isfinite(model).all()
    for row in accepted:
        a, b = row['start_sample'], row['end_sample_exclusive']
        assert (
            np.all(bits[a:b] == 0)
            and np.all(y[a:b] == row['label'])
            and np.all(tids[a:b] == row['trial_id'])
        )
    raw_stats, filtered_stats, model_stats = (
        features(x),
        features(physical),
        features(model[:, 0]),
    )
    diagnostics = []
    paired = []
    for i, row in enumerate(accepted):
        paired.append(
            dict(
                row,
                array_index=i,
                npz_path=str(destination / 'paired_windows.npz'),
                source_array_index=i,
                source_npz=str(destination / 'paired_windows.npz'),
                source_bdf=str(bdf),
            )
        )
        diag = dict(
            record_id=rid,
            array_index=i,
            trial_id=row['trial_id'],
            label=row['label'],
            start_sample=row['start_sample'],
            end_sample_exclusive=row['end_sample_exclusive'],
            norm_center_uv=float(centers[i]),
            norm_scale_uv=float(scales[i]),
            clipped_fraction=float(fractions[i]),
            high_clipping_review=bool(fractions[i] > 0.01),
        )
        for prefix, stats in [
            ('raw', raw_stats),
            ('filtered', filtered_stats),
            ('model', model_stats),
        ]:
            for key, values in stats.items():
                report_key = (
                    {
                        'dc_uv': 'mean',
                        'std_uv': 'std',
                        'slope_uv_per_s': 'slope_per_second',
                    }.get(key, key)
                    if prefix == 'model'
                    else key
                )
                diag[prefix + '_' + report_key] = float(values[i])
        diag['filtered_peak_to_peak_uv'] = float(np.ptp(physical[i]))
        diag['model_max_abs'] = float(np.abs(model[i]).max())
        diagnostics.append(diag)
    np.savez_compressed(
        destination / 'paired_windows.npz',
        X_uv=x,
        X_filtered_uv=physical,
        X_model=model,
        y=yout,
        start_sample=start_array,
        end_sample_exclusive=start_array + 1000,
        trial_id=trial_array,
        norm_center_uv=centers,
        norm_scale_uv=scales,
        clipped_fraction=fractions,
        sfreq=np.array(250),
        channel_names=np.array(['EEG']),
        preprocessed=np.array(True),
    )
    xp, yp = STAGING / (rid + '_X.npy'), STAGING / (rid + '_y.npy')
    np.save(xp, model[:, 0], allow_pickle=False)
    np.save(yp, yout, allow_pickle=False)
    table(destination / 'windows_paired.csv', paired)
    table(destination / 'windows_rejected.csv', rejected)
    table(destination / 'window_diagnostics.csv', diagnostics)
    metadata = dict(
        record_id=rid,
        bdf_path=str(bdf),
        events_path=str(events_path),
        bdf_sha256=sha(bdf),
        events_sha256=sha(events_path),
        time_origin_beijing=ticks_iso(origin),
        origin_ticks_100ns=origin,
        time_origin_source='filename milliseconds; cross-checked T0 annotation',
        header_time_interpretation='Beijing UTC+08:00; MNE UTC tag not used as actual timezone',
        sfreq=250,
        channel_names=['EEG'],
        physical_unit='uV',
        samples=n,
        duration_seconds=n / 250,
        physical_min_uv=pmin,
        physical_max_uv=pmax,
        quantization_step_uv=quantum,
        trial_count=len(trials),
        valid_trials=sum(t['valid'] for t in trials),
        trial_duration_seconds=[t['duration_seconds'] for t in trials],
        alignment_issues=issues,
        candidate_phase_windows=eligible,
        first_qc_kept=before_count,
        second_qc_rejected=before_count - len(yout),
        clean_windows=len(yout),
        class_counts=np.bincount(yout, minlength=2).tolist(),
        high_clipping_review_windows=int(np.sum(fractions > 0.01)),
        bad_samples=int(((bits & 1) != 0).sum()),
        saturation_samples=int(((bits & 4) != 0).sum()),
        flatline_samples=int(((bits & 8) != 0).sum()),
        rejection_reason_counts=dict(
            Counter(
                reason
                for row in rejected
                for reason in row['exclude_reason'].split(';')
            )
        ),
    )
    dump(destination / 'metadata.json', metadata)
    raw.close()
    entry = dict(
        record_id=rid,
        windows=len(yout),
        shape=list(model[:, 0].shape),
        class_counts=np.bincount(yout, minlength=2).tolist(),
        source_npz=str(destination / 'paired_windows.npz'),
        source_sha256=sha(destination / 'paired_windows.npz'),
        source_bdf=str(bdf),
        source_bdf_sha256=metadata['bdf_sha256'],
        source_events=str(events_path),
        source_events_sha256=metadata['events_sha256'],
        x_file=xp.name,
        y_file=yp.name,
        x_sha256=sha(xp),
        y_sha256=sha(yp),
    )
    print(
        'PROCESSED',
        rid,
        'windows',
        len(yout),
        'classes',
        entry['class_counts'],
        'issues',
        len(issues),
        flush=True,
    )
    return entry, metadata, paired


def main():
    assert SOURCE.is_dir() and not AUDIT.exists() and not STAGING.exists()
    records = sorted(p for p in SOURCE.iterdir() if p.is_dir() and p.name.isdigit())
    assert len(records) == 57 and not any(p.name in ['032', '033'] for p in records)
    AUDIT.mkdir()
    STAGING.mkdir()
    config = dict(
        CONFIG,
        dataset_version='EEG_dataset_20261027_filename_t0_v1',
        source_dataset=str(SOURCE),
        time_alignment='filename exact milliseconds, UTC+08:00; integer 100ns event arithmetic',
        window_placement='4s on 1s grid relative to first sample; complete window must be within actual phase interval',
        raw_output='X_uv: original microvolt after quality selection',
        physical_output='X_filtered_uv: causal filtered and mean-centered microvolt',
        model_output='X_model: per-window MAD, clip +/-8 then mean-center',
    )
    dump(STAGING / 'preprocessing.json', config)
    dump(AUDIT / 'preprocessing.json', config)
    entries = []
    metadata = []
    all_windows = []
    for folder in records:
        entry, meta, windows = process(folder)
        entries.append(entry)
        metadata.append(meta)
        all_windows.extend(windows)
        dump(
            AUDIT / 'progress.json',
            dict(
                status='RUNNING',
                processed_records=len(entries),
                total_records=57,
                windows=sum(r['windows'] for r in entries),
            ),
        )
    # 新57条记录重新生成固定划分，不能使用旧48条记录的划分或旧模型。
    order = (
        np.random.RandomState(42)
        .permutation([r['record_id'] for r in entries])
        .tolist()
    )
    validation = sorted(order[:12])
    train = sorted(order[12:])
    split = dict(
        train_records=train,
        validation_records=validation,
        seed=42,
        group='record_id',
        subject_independence_unverified=True,
        dataset_version=config['dataset_version'],
    )
    group_counts = {
        name: sum(r['windows'] for r in entries if r['record_id'] in ids)
        for name, ids in [('train', train), ('validation', validation)]
    }
    counts = np.sum([r['class_counts'] for r in entries], axis=0).tolist()
    for entry in entries:
        entry['split'] = 'train' if entry['record_id'] in train else 'validation'
    for meta in metadata:
        meta.pop('alignment_issues', None)
    table(AUDIT / 'record_summary.csv', metadata)
    table(STAGING / 'source_windows_paired.csv', all_windows)
    dump(STAGING / 'split.json', split)
    manifest = dict(
        dataset_version=config['dataset_version'],
        source=str(AUDIT),
        raw_source=str(SOURCE),
        raw_readme_sha256=sha(SOURCE / 'README.txt'),
        output=str(FINAL),
        records=entries,
        record_count=57,
        windows=sum(r['windows'] for r in entries),
        split_windows=group_counts,
        class_counts=dict(relax=counts[0], task=counts[1]),
        x_spec='float32 [N,1,1000], dimensionless processed X_model',
        y_spec='int64 [N], 0 relax / 1 task',
        sfreq=250,
        seconds=4,
        stride_seconds=1,
        bandpass_hz=[1, 50],
        notch_hz=50,
        source_order_preserved=True,
        signal_processing_performed=True,
        quality_policy='exclude BAD annotations, nonfinite, ADC rails, flatline >=1s; second QC numerical/constant; clipping >1% review only',
        pipeline_source_sha256={
            p.name: sha(p) for p in Path(__file__).parent.glob('*.py')
        },
        filter_core_sha256=sha(
            ROOT / 'data_processing_1_50hz/quality_preprocess_1_50hz.py'
        ),
    )
    dump(STAGING / 'manifest.json', manifest)
    dump(STAGING / 'verification.json', dict(status='PENDING_INDEPENDENT_VERIFICATION'))
    dump(
        AUDIT / 'progress.json',
        dict(
            status='COMPLETED',
            processed_records=57,
            total_records=57,
            windows=manifest['windows'],
            class_counts=manifest['class_counts'],
            split_windows=group_counts,
        ),
    )
    (STAGING / 'README.md').write_text(
        f'''# 更新数据的1–50Hz prepared缓存

来源：{SOURCE}。只使用新57条记录，原点由文件名毫秒时间与T0注释核对，事件按北京时间对齐。按实际trial_start/trial_end配对，阶段外与跨阶段窗口不训练。

4秒/1000点、1秒步长、250Hz、单EEG通道。排除BAD_ACQ_SKIP、非有限值、ADC上下限及持续至少1秒的平直信号所影响的窗口；不以零值本身认定缺失。

每个保留窗口独立执行1–50Hz四阶因果带通、50Hz陷波Q=30、去均值、逐窗口MAD缩放（系数1.4826、尺度下限1微伏）、±8限幅和再次去均值。不读取窗后未来信号，不跨缺失区间传递滤波状态。限幅比例>1%仅作复核标记。

每条记录两个文件：编号_X.npy为float32[N,1,1000]、编号_y.npy为int64[N]；0=relax、1=task。X为缩放后的无量纲模型输入，不是微伏。共{manifest['windows']}窗口，relax={counts[0]}、task={counts[1]}；训练{group_counts['train']}窗口/45记录、验证{group_counts['validation']}窗口/12记录，按记录固定seed42划分，被试独立性尚未确认。

source_windows_paired.csv将array_index对应到原BDF样本位置和处理后NPZ。processing_current中保存每条记录原始微伏、滤波微伏、模型输入、精确时间原点、事件/试次、逐点质量标记、剔除理由和第二次质量诊断。manifest.json记录来源与输出哈希，verification.json记录独立复算结果。

本次重新处理新数据，旧数据训练成绩不适用于此数据版本。后续训练应使用动态加载器读取本manifest/split，不能复用旧48条记录的固定窗口数。
''',
        encoding='utf-8',
    )
    print('DATASET_COMPLETE', manifest['windows'], counts, group_counts, flush=True)


if __name__ == '__main__':
    main()
