"""Read-only, local dataset validation; output can contain private file metadata."""
import argparse,csv,hashlib,json,math,re,sys
from pathlib import Path
import mne

def file_info(p):
    with p.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'name':p.name,'bytes':p.stat().st_size,'sha256':digest}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--expected-records',type=int,default=57);a=ap.parse_args()
    rows=[]
    for folder in sorted(a.data_root.iterdir()):
        if not folder.is_dir() or not folder.name.isdigit():continue
        b=list(folder.glob('*.bdf'));c=list(folder.glob('*events.csv'))
        if len(b)!=1 or len(c)!=1:raise ValueError('Each record requires exactly one BDF and one events CSV')
        if not re.search(r'_\d{8}_\d{6}\.\d+_UTC\+0800\.bdf$',b[0].name):raise ValueError('Missing filename timestamp with milliseconds and UTC+0800')
        raw=mne.io.read_raw_bdf(b[0],preload=False,verbose='ERROR')
        if not math.isclose(raw.info['sfreq'],250,rel_tol=0,abs_tol=1e-9) or raw.ch_names!=['EEG']:
            raw.close();raise ValueError('Expected one EEG channel at 250 Hz')
        raw.close()
        with c[0].open(encoding='utf-8-sig',newline='') as f:
            if set(csv.DictReader(f).fieldnames or [])!={'local_time','event','phase','key'}:raise ValueError('Unexpected event schema')
        rows.append({'record':folder.name,'files':[file_info(p) for p in [b[0],c[0]]]})
    if len(rows)!=a.expected_records:raise ValueError(f'Expected {a.expected_records} records, found {len(rows)}')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'records':rows,'raw_data_read_only':True},indent=2),encoding='utf8')
    print(json.dumps({'status':'PASS','records':len(rows),'note':'Format and hash audit only; no subject-ID or split independence claim'}))
if __name__=='__main__':main()
