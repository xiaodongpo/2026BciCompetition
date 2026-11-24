"""Download the author checkpoint via standard urllib proxy handling and verify its published SHA256."""
import urllib.request,json,base64,hashlib,time
from pathlib import Path

def run():
    path=Path('third_party/singlem_downstream_excluded.pt')
    expected=json.loads(Path('third_party/singlem_checksums.json').read_text())[path.name]['sha256']
    urls=['https://raw.githubusercontent.com/ttlabtuat/SingLEM/main/SingLEM/checkpoints/'+path.name,
          'https://api.github.com/repos/ttlabtuat/SingLEM/git/blobs/f9dee1fd4b97d6be6b204a202c1e8b6bc341d789']
    for url in urls:
        for attempt in range(3):
            try:
                req=urllib.request.Request(url,headers={'Accept':'application/vnd.github.raw+json','User-Agent':'BCI-research-checkpoint-download'})
                with urllib.request.urlopen(req,timeout=25) as r:data=r.read()
                if data.startswith(b'{'):
                    obj=json.loads(data);data=base64.b64decode(obj['content'])
                assert hashlib.sha256(data).hexdigest()==expected
                temp=path.with_suffix('.verified');temp.write_bytes(data);temp.replace(path)
                print('Official SingLEM SHA256 verified',expected,len(data),flush=True);return
            except Exception as e:print('Download attempt',attempt,type(e).__name__,str(e)[:120],flush=True);time.sleep(1)
    raise RuntimeError('Official checkpoint download unavailable')

if __name__=='__main__':run()
