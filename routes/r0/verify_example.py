"""比较最小示例输出。CPU/GPU允许小量浮点误差，类别应相同。"""
import argparse,csv,json
from pathlib import Path
import numpy as np

if __name__=='__main__':
    base=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser();ap.add_argument('--actual',default=str(base/'example/recheck.csv'))
    ap.add_argument('--expected',default=str(base/'example/expected_cpu.csv'));ap.add_argument('--atol',type=float,default=1e-4);a=ap.parse_args()
    def rows(p):
        with Path(p).open(encoding='utf8') as f:return list(csv.DictReader(f))
    x,y=rows(a.actual),rows(a.expected)
    assert len(x)==len(y)==32,(len(x),len(y))
    assert [r['end_sample'] for r in x]==[r['end_sample'] for r in y]
    err=float(np.max(np.abs([float(r['score'])-float(s['score']) for r,s in zip(x,y)])))
    assert err<=a.atol,err
    assert [r['state'] for r in x]==[r['state'] for r in y]
    print('EXAMPLE_VERIFY_PASS',json.dumps(dict(rows=len(x),max_abs_score_error=err,tolerance=a.atol)))
