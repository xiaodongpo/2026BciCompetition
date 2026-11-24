"""Compare an authorized local bundle with the historical SHA-256 inventory."""
import argparse, hashlib, json, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--route',required=True,choices=['h1','r0','d2','third_party'])
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'checkpoints/manifest.json')
    a=p.parse_args(); rows=[r for r in json.loads(a.manifest.read_text()) if r['route']==a.route]
    if not rows:p.error('No manifest entries for this route')
    failures=[]
    for row in rows:
        path=a.model/row['filename']
        if not path.is_file():failures.append({'file':row['filename'],'reason':'missing'});continue
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=row['sha256'] or path.stat().st_size!=row['bytes']:
            failures.append({'file':row['filename'],'reason':'digest or size differs'})
    print(json.dumps({'status':'FAIL' if failures else 'PASS','files':len(rows),'failures':failures,'note':'Identity only, not redistribution permission or proof of model quality'},indent=2))
    sys.exit(bool(failures))

if __name__=='__main__':main()
