"""Supplemental actual-stream OOF check, after the frozen promotion decision."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.round2_inference import RoundTwoPredictor
from scripts.round2_evaluate import measures


def run():
    protocol=json.loads(Path('configs/round2/protocol.json').read_text())
    summaries={}
    for kind in ('selected','reference'):
        rows=[];coverage=[];timings=[]
        for k,records in enumerate(protocol['outer_folds']):
            path=Path('experiments/round2/record_nested')/f'outer{k}'/(kind+'.joblib')
            predictor=RoundTwoPredictor(path,hop_samples=500)
            for rid in records:
                assert rid not in protocol['excluded_previous_holdout']
                data=np.load('cache/records/'+rid+'.npz');x=data['x'];valid=data['valid'];labels=data['labels']
                predictor.reset();start=time.perf_counter();outputs=[]
                for a in range(0,len(x),4096):outputs.extend(predictor.process(x[a:a+4096],start_sample=a,valid=valid[a:a+4096]))
                timings.append(dict(record=rid,seconds=time.perf_counter()-start))
                eligible=predicted=0
                for r in outputs:
                    end=r['sample_index'];label=int(labels[end])
                    if label<0 or not valid[end]:continue
                    eligible+=1
                    if not r['valid']:continue
                    predicted+=1
                    rows.append(dict(record=rid,end=end,y=label,p=r['p_task'],prediction=r['label'],threshold=predictor.threshold,outer_fold=k))
                coverage.append(dict(record=rid,eligible=eligible,predicted=predicted,coverage=predicted/max(eligible,1)))
            print(kind,'completed outer',k,flush=True)
        frame=pd.DataFrame(rows);frame.to_csv(f'results/round2/continuous_{kind}_oof.csv',index=False)
        record=pd.DataFrame([dict(record=r,**measures(g)) for r,g in frame.groupby('record')])
        record.to_csv(f'results/round2/continuous_{kind}_records.csv',index=False)
        pd.DataFrame(coverage).to_csv(f'results/round2/continuous_{kind}_coverage.csv',index=False)
        summaries[kind]=dict(pooled=measures(frame),record_macro=dict(bacc=float(record.bacc.mean()),auc=float(record.auc.mean())),coverage=sum(r['predicted'] for r in coverage)/sum(r['eligible'] for r in coverage),timings=timings,seconds=sum(t['seconds'] for t in timings))
    summaries['notes']='Supplemental after-lock check, not used for selection/promotion. Actual streaming adapter; 500-sample (2s) emission grid; endpoint labels include mixed-phase windows. Missing/unlabeled endpoints excluded; warmup counted against coverage. Original holdout excluded. Default deployment emits every 125 samples and is separately tested.'
    Path('results/round2/continuous_summary.json').write_text(json.dumps(summaries,indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='timings'} if isinstance(v,dict) else v for k,v in summaries.items()},indent=2))


if __name__=='__main__':run()
