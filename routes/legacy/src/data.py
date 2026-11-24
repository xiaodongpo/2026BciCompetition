"""Exact local-clock alignment and conservative gap masks; original files are read-only."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne

FS = 250
NS = 4_000_000

def filename_time(path):
    match = re.search(r'_(\d{8})_(\d{6}\.\d+)_UTC\+0800', Path(path).name)
    if not match:
        raise ValueError(f'Missing exact timestamp: {path}')
    return pd.Timestamp(pd.to_datetime(''.join(match.groups()), format='%Y%m%d%H%M%S.%f')).tz_localize('Asia/Shanghai')

def ceil_sample(time, origin):
    delta = pd.Timestamp(time).value - origin.value
    return -(-delta // NS)

def pair_events(events, origin, n):
    required = {'local_time', 'event', 'phase', 'key'}
    if set(events.columns) != required:
        raise ValueError(f'Unexpected columns {events.columns}')
    times = pd.to_datetime(events.local_time, utc=True, format='ISO8601')
    if not times.is_monotonic_increasing:
        raise ValueError('Events not ordered')
    active = None
    trials, issues = [], []
    for row in events.itertuples(index=False):
        if row.event == 'trial_start':
            if active is not None:
                raise ValueError('Nested trial start')
            if row.phase not in ('relax', 'task'):
                raise ValueError('Unknown phase')
            active = row
        elif row.event == 'trial_end':
            if active is None or active.phase != row.phase:
                raise ValueError('Unmatched trial end')
            a, b = ceil_sample(active.local_time, origin), ceil_sample(row.local_time, origin)
            if b <= a:
                raise ValueError('Nonpositive trial')
            if a < 0 or b > n:
                issues.append(f'trial clipped {a}:{b} to 0:{n}')
            trials.append(dict(phase=row.phase, label=int(row.phase=='task'), start=max(0,a), end=min(n,b),
                               event_start=active.local_time, event_end=row.local_time,
                               duration_s=(pd.Timestamp(row.local_time)-pd.Timestamp(active.local_time)).total_seconds()))
            active = None
    if active is not None:
        issues.append('Unclosed final trial excluded')
    for left,right in zip(trials,trials[1:]):
        if left['end'] > right['start']:
            raise ValueError('Overlapping labeled trials')
    return trials, issues

def read_record(folder):
    folder = Path(folder)
    bdfs, csvs = list(folder.glob('*.bdf')), list(folder.glob('*events.csv'))
    if len(bdfs)!=1 or len(csvs)!=1:
        raise ValueError(f'Unexpected file count {folder}')
    path=bdfs[0]
    raw=mne.io.read_raw_bdf(path, preload=False, verbose='ERROR')
    if not np.isclose(raw.info['sfreq'],FS,rtol=0,atol=1e-6) or len(raw.ch_names)!=1:
        raise ValueError('Expected one channel at 250 Hz')
    origin=filename_time(path)
    x=(raw.get_data()[0]*1e6).astype(np.float32)
    valid=np.isfinite(x)
    gaps=[]
    for onset,duration,description in zip(raw.annotations.onset,raw.annotations.duration,raw.annotations.description):
        if str(description).upper().startswith('BAD'):
            a=max(0,int(np.floor(onset*FS+1e-6)))
            b=min(len(x),int(np.ceil((onset+duration)*FS-1e-6)))
            valid[a:b]=False
            gaps.append(dict(start=a,end=b,description=str(description)))
    events=pd.read_csv(csvs[0],keep_default_na=False)
    trials,issues=pair_events(events,origin,len(x))
    labels=np.full(len(x),-1,np.int8)
    for t in trials:
        labels[t['start']:t['end']]=t['label']
    labels[~valid]=-1
    info=dict(record=folder.name,file=str(path),origin=origin.isoformat(),samples=len(x),duration_s=len(x)/FS,
              sfreq=FS,channel=raw.ch_names[0],channel_type=raw.get_channel_types()[0],
              original_unit=str(raw._orig_units),mne_header_time=str(raw.info['meas_date']),
              header_subject=str(raw.info.get('subject_info')),issues=issues)
    raw.close()
    return x,valid,labels,trials,gaps,events,info

def windows(x, valid, labels, seconds=6, stride=2):
    """Right-edge fixed grid. Labels only select training/evaluation examples, never inference."""
    size,step=int(seconds*FS),int(stride*FS)
    if not 0<size<=1500 or step<=0:
        raise ValueError('Invalid window / stride')
    for end in range(size,len(x)+1,step):
        a=end-size
        y=int(labels[end-1])
        if y>=0 and valid[a:end].all() and (labels[a:end]==y).all():
            yield x[a:end],y,end
