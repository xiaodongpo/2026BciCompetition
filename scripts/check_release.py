"""Read-only checks of publishable files. Never prints matched secret values."""
import ast,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.venv','__pycache__','.pytest_cache','cache','workspace','EEG_dataset','test_results','experiments','pretrained'}
PRIVATE_SUFFIX={'.bdf','.edf','.pt','.pth','.ckpt','.joblib','.pkl','.safetensors','.npy','.npz','.zip','.pem','.key'}
PATTERNS={
 'github_token':r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})',
 'aws_key':r'AKIA[A-Z0-9]{16}',
 'private_key':r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
 'credential_assignment':r'''(?i)(?:api_key|password|access_token)\s*[:=]\s*["'][A-Za-z0-9+/=_-]{12,}["']''',
 'personal_windows_path':r'(?i)[A-Z]:[\\/]+Users[\\/]+[^\s"\']+',
 'private_host':r'\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3})\b',
}

def public_files():
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(s in SKIP or s.startswith('.venv') or s.endswith('.egg-info') for s in rel.parts):continue
        if p.name=='.release-staging-marker':continue
        if rel.parts[0]=='routes' and any(s in {'model','checkpoints','results','reports','example'} for s in rel.parts):continue
        yield p

def scan():
    errors=[];files=list(public_files())
    for p in files:
        rel=p.relative_to(ROOT).as_posix()
        if p.suffix.lower() in PRIVATE_SUFFIX or p.name.endswith('_events.csv') or p.name.startswith('.env'):
            errors.append(dict(file=rel,kind='private_artifact'));continue
        if p.stat().st_size>5*1024**2:errors.append(dict(file=rel,kind='large_file'))
        try:t=p.read_text(encoding='utf-8-sig')
        except UnicodeError:errors.append(dict(file=rel,kind='unreviewed_binary'));continue
        for kind,pattern in PATTERNS.items():
            for m in re.finditer(pattern,t):errors.append(dict(file=rel,kind=kind,line=t.count('\n',0,m.start())+1))
        try:
            if p.suffix=='.py':ast.parse(t,filename=rel)
            if p.suffix=='.json':json.loads(t)
        except (SyntaxError,ValueError) as e:errors.append(dict(file=rel,kind='parse',detail=str(e)))
        if p.suffix=='.md':
            for link in re.findall(r'\]\(([^)]+)\)',t):
                target=link.split('#')[0]
                if not target or '://' in target or target.startswith('mailto:'):continue
                if not (p.parent/target).exists():errors.append(dict(file=rel,kind='broken_link',target=target))
    return dict(status='PASS' if not errors else 'FAIL',files=len(files),bytes=sum(p.stat().st_size for p in files),findings=errors,scope='current publishable files; runtime caches are excluded and must not enter Git',license_owner_approval_pending='PENDING' in (ROOT/'LICENSE').read_text())

if __name__=='__main__':
    report=scan();print(json.dumps(report,indent=2));sys.exit(bool(report['findings']))
