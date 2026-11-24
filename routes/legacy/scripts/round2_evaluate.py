"""Report the frozen nested procedures; never select a candidate using outer scores."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,accuracy_score,f1_score


def measures(frame):
    return dict(bacc=float(balanced_accuracy_score(frame.y,frame.prediction)),
                auc=float(roc_auc_score(frame.y,frame.p)),
                accuracy=float(accuracy_score(frame.y,frame.prediction)),
                f1=float(f1_score(frame.y,frame.prediction)),n=len(frame))


def collect(name,kind):
    frame=pd.read_csv(f'results/round2/{name}_{kind}_oof.csv',dtype={'record':str})
    protocol=json.loads(Path('configs/round2/protocol.json').read_text())
    assert set(frame.record)==set(protocol['development'])
    assert not set(frame.record)&set(protocol['excluded_previous_holdout'])
    assert not frame.duplicated(['record','end']).any()
    assert ((frame.p>=frame.threshold).astype(int)==frame.prediction).all()
    records=pd.DataFrame([dict(record=r,outer_fold=int(g.outer_fold.iloc[0]),**measures(g)) for r,g in frame.groupby('record')])
    records['score']=(records.bacc+records.auc)/2
    records.to_csv(f'results/round2/{name}_{kind}_records.csv',index=False)
    folds=[]
    for k,g in frame.groupby('outer_fold'):
        m=records[records.outer_fold==k]
        folds.append(dict(outer_fold=int(k),records=len(m),selected_id=g.selected_id.iloc[0],threshold=float(g.threshold.iloc[0]),macro_bacc=float(m.bacc.mean()),macro_auc=float(m.auc.mean()),macro_score=float(m.score.mean()),**measures(g)))
    pd.DataFrame(folds).to_csv(f'results/round2/{name}_{kind}_folds.csv',index=False)
    return dict(pooled=measures(frame),record_macro=dict(bacc=float(records.bacc.mean()),auc=float(records.auc.mean()),score=float(records.score.mean())),folds=folds),records


def run():
    report={};all_records={}
    for name in ('record_nested','date_nested'):
        report[name]={};all_records[name]={}
        for kind in ('selected','reference'):
            report[name][kind],all_records[name][kind]=collect(name,kind)
    a=all_records['record_nested']['selected'].set_index('record')
    b=all_records['record_nested']['reference'].set_index('record').loc[a.index]
    d=a.score.to_numpy()-b.score.to_numpy()
    rng=np.random.default_rng(20261103);boots=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
    low,high=np.quantile(boots,[.025,.975])
    gains=[x['macro_score']-y['macro_score'] for x,y in zip(report['record_nested']['selected']['folds'],report['record_nested']['reference']['folds'])]
    date_gain=report['date_nested']['selected']['record_macro']['score']-report['date_nested']['reference']['record_macro']['score']
    report['promotion']=dict(passed=bool(low>0 and min(gains)>0 and date_gain>=0),record_mean_gain=float(d.mean()),record_bootstrap_95=[float(low),float(high)],record_fold_gains=gains,date_macro_score_gain=date_gain,bootstrap_replicates=10000,seed=20261103,uncertainty_unit='record; shared participants are unknown, so not a subject-level confidence interval')
    pd.DataFrame(dict(record=a.index,selected_score=a.score.to_numpy(),reference_score=b.score.to_numpy(),gain=d)).to_csv('results/round2/paired_record_gains.csv',index=False)
    report['final_inner_selection']=json.loads(Path('configs/round2/selected_model.json').read_text())
    rows=[]
    for p in Path('experiments/round2').glob('*/*/*_inner.json'):
        r=json.loads(p.read_text());rows.append(dict(protocol=p.parent.parent.name,outer=p.parent.name,**{k:r[k] for k in ('id','inner_score','threshold','seconds','input_features','classifier_parameters')}))
    for p in Path('experiments/round2/final_selection').glob('*_inner.json'):
        r=json.loads(p.read_text());rows.append(dict(protocol='final_selection',outer='all45',**{k:r[k] for k in ('id','inner_score','threshold','seconds','input_features','classifier_parameters')}))
    table=pd.DataFrame(rows);table.to_csv('results/round2/experiments.csv',index=False)
    report['executed']=dict(candidate_inner_searches=len(table),inner_fits=len(table)*3,outer_fits=14,final_fit=1,total_fits=len(table)*3+15,inner_fit_and_prediction_seconds=float(table.seconds.sum()))
    package=Path('submission_20261103.zip')
    report['original_submission']=dict(path=str(package),sha256=hashlib.sha256(package.read_bytes()).hexdigest(),bytes=package.stat().st_size)
    assert report['original_submission']['sha256']=='f842bc85ce379645abe49e08063d098e549e58fe34e021eef46fdb5a5ecf762a'
    report['limitations']=['Development data had been studied previously; this is not a new independent test set.','No verified participant, electrode or device metadata; no subject-wise generalization claim.','Acquisition date is an imperfect group proxy, never an identity or input feature.','Record bootstrap cannot account for unknown shared participants.','Foundation-model corpus overlap with this competition cannot be independently excluded.','Only the fixed normalized-input SingLEM probe was tested; no universal conclusion about the pretrained model.']
    Path('results/round2/summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':run()
