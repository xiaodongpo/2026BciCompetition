"""Evaluate frozen outer predictions, without retroactive candidate selection."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.round2_evaluate import measures


def collect(name,kind):
    p=json.loads(Path('configs/round5/protocol.json').read_text())
    f=pd.read_csv(f'results/round5/{name}_{kind}_oof.csv',dtype={'record':str})
    assert set(f.record)==set(p['development']) and not set(f.record)&set(p['excluded_previous_holdout'])
    assert not f.duplicated(['record','end']).any()
    assert ((f.p>=f.threshold).astype(int)==f.prediction).all()
    records=pd.DataFrame([dict(record=r,outer_fold=int(g.outer_fold.iloc[0]),**measures(g)) for r,g in f.groupby('record')])
    records['score']=(records.bacc+records.auc)/2;records.to_csv(f'results/round5/{name}_{kind}_records.csv',index=False)
    folds=[]
    for k,g in f.groupby('outer_fold'):
        r=records[records.outer_fold==k]
        folds.append(dict(outer_fold=int(k),selected_id=g.selected_id.iloc[0],threshold=float(g.threshold.iloc[0]),records=len(r),macro_bacc=float(r.bacc.mean()),macro_auc=float(r.auc.mean()),macro_score=float(r.score.mean()),pooled=measures(g)))
    return dict(pooled=measures(f),record_macro=dict(bacc=float(records.bacc.mean()),auc=float(records.auc.mean()),score=float(records.score.mean())),folds=folds),records


def run():
    report={};records={}
    for name in ('record_nested','date_nested'):
        report[name]={};records[name]={}
        for kind in ('selected','reference'):report[name][kind],records[name][kind]=collect(name,kind)
    a=records['record_nested']['selected'].set_index('record');b=records['record_nested']['reference'].set_index('record').loc[a.index]
    gain=a.score.to_numpy()-b.score.to_numpy();rng=np.random.default_rng(20261104)
    lo,hi=np.quantile(gain[rng.integers(0,len(gain),(10000,len(gain)))].mean(1),[.025,.975])
    folds=[x['macro_score']-y['macro_score'] for x,y in zip(report['record_nested']['selected']['folds'],report['record_nested']['reference']['folds'])]
    date_gain=report['date_nested']['selected']['record_macro']['score']-report['date_nested']['reference']['record_macro']['score']
    bacc_gain=float((a.bacc-b.bacc).mean())
    report['promotion']=dict(passed=bool(lo>0 and min(folds)>0 and date_gain>=0 and bacc_gain>=0),record_mean_score_gain=float(gain.mean()),record_bootstrap_95=[float(lo),float(hi)],record_fold_score_gains=folds,date_macro_score_gain=date_gain,record_macro_bacc_gain=bacc_gain,bootstrap_replicates=10000,seed=20261104,uncertainty_unit='record, not verified participant')
    pd.DataFrame(dict(record=a.index,selected_score=a.score.to_numpy(),reference_score=b.score.to_numpy(),gain=gain)).to_csv('results/round5/paired_record_gains.csv',index=False)
    rows=[]
    for p in Path('experiments/round5').rglob('*.json'):
        r=json.loads(p.read_text())
        if 'fits' not in r or p.name=='selection.json':continue
        rows.append(dict(context=p.parent.relative_to('experiments/round5').as_posix(),id=r['id'],features=r['config']['features'],model=r['config']['model'],inner_score=r['inner_score'],threshold=r['threshold'],seconds=r['seconds'],fit_count=len(r['fits'])))
    table=pd.DataFrame(rows);table.to_csv('results/round5/experiments.csv',index=False)
    assert len(table)==64,len(table)
    report['executed']=dict(candidate_inner_searches=len(table),inner_fits=int(table.fit_count.sum()),outer_fits=14,final_fit=1,total_research_fits=int(table.fit_count.sum())+15,inner_fit_prediction_logging_seconds=float(table.seconds.sum()),training_invocation=json.loads(Path('experiments/round5/completed.json').read_text()))
    report['final_inner_selection']=json.loads(Path('configs/round5/selected_model.json').read_text())
    old=json.loads(Path('reports/round5/prior_archives.json').read_text())
    for name,digest in old.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    report['prior_archives_unchanged']=old
    report['limitations']=['Previously studied development records; no new independent test.','Unknown participant, electrode and device identities.','Date groups do not identify devices or participants.','Record bootstrap may be optimistic if records share participants.','Entropy is not a direct measure of cognitive load; morphology and artifact contributions remain unresolved.']
    Path('results/round5/summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(promotion=report['promotion'],record_nested={k:v['record_macro'] for k,v in report['record_nested'].items()},date_nested={k:v['record_macro'] for k,v in report['date_nested'].items()},selected=report['final_inner_selection']['id'],executed=report['executed']),indent=2))


if __name__=='__main__':run()
