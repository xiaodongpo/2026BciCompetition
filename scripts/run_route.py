"""Run retained research modules with their own cwd and namespace."""
import argparse,datetime,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path)
    p.add_argument('route',choices=['legacy','d2','h1'])
    p.add_argument('module');p.add_argument('args',nargs=argparse.REMAINDER)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]/'routes'/a.route
    if not a.module.startswith(('scripts.','src.')):p.error('Expected a route scripts.* or src.* module')
    if not (root/Path(*a.module.split('.'))).with_suffix('.py').is_file():p.error('Module not found in selected route')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    if a.data_root:
        if not a.data_root.is_dir():p.error('Data root must exist')
        env['BCI_DATA_ROOT']=str(a.data_root.resolve())
    for d in ['reports','results','experiments','cache','checkpoints']:(root/d).mkdir(exist_ok=True)
    if a.route=='h1':(root/'results/finals').mkdir(exist_ok=True)
    # Execution metadata stays in the ignored route cache, including local data paths.
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    audit_dir=root/'cache/launcher_runs'/stamp;audit_dir.mkdir(parents=True)
    sources={}
    for folder in ['src','scripts','configs','finals/submission']:
        for path in (root/folder).rglob('*'):
            if path.is_file() and path.suffix in {'.py','.json'} and 'model' not in path.relative_to(root).parts:
                sources[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    record=dict(route=a.route,module=a.module,args=a.args,data_root=env.get('BCI_DATA_ROOT','EEG_dataset'),python=platform.python_version(),source_sha256=sources,status='running')
    def save():(audit_dir/'run.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    save();start=time.perf_counter()
    try:
        with (audit_dir/'console.log').open('w',encoding='utf8') as log:
            child=subprocess.Popen([sys.executable,'-u','-m',a.module,*a.args],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace')
            for line in child.stdout:sys.stdout.write(line);sys.stdout.flush();log.write(line);log.flush()
            code=child.wait()
        record.update(status='PASS' if code==0 else 'FAIL',returncode=code)
    except BaseException:
        if 'child' in locals() and child.poll() is None:child.terminate();child.wait()
        record.update(status='INTERRUPTED');raise
    finally:
        record['seconds']=time.perf_counter()-start;save()
    raise SystemExit(code)
if __name__=='__main__':main()
