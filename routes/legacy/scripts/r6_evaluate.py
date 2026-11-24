"""Aggregate executed ablations; choose no settings from these outer metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.transition_evaluation import summary,paired_gate


def aggregate(frames,threshold,name):
    f=pd.concat(frames).sort_values(['record','end']).reset_index(drop=True);meta=pd.read_csv('cache/round6/windows.csv',dtype={'record':str})
    assert f[['record','end','y']].equals(meta[['record','end','y']])
    metrics,records=summary(f,f.p.to_numpy(),threshold);foldmap=f.groupby('record').outer_fold.first();records['outer_fold']=records.record.map(foldmap)
    pure=f.pure.to_numpy();pure_m,_=summary(f[pure],f.p.to_numpy()[pure],threshold)
    f.to_csv(f'results/round6/{name}_oof.csv',index=False);records.to_csv(f'results/round6/{name}_records.csv',index=False)
    return dict(full=metrics,pure=pure_m),records


def run():
    assert Path('experiments/round6/lr_complete.json').exists()
    all_rows=[];result={};tables=[]
    for protocol,nfolds in [('record',3),('date',4)]:
        outpaths=[Path(f'experiments/round6/lr/{protocol}/outer{k}') for k in range(nfolds)];done=[json.loads((p/'complete.json').read_text()) for p in outpaths]
        ids=[a['id'] for a in done[0]['ablations']];summaries={};records={}
        for arm in [*ids,'selected_procedure']:
            frames=[]
            for k,(path,d) in enumerate(zip(outpaths,done)):
                actual=d['selected'] if arm=='selected_procedure' else arm
                frames.append(pd.read_csv(path/(actual+'.csv'),dtype={'record':str}))
            summaries[arm],records[arm]=aggregate(frames,.46,f'lr_{protocol}_{arm}')
            tables.append(dict(model='lr',protocol=protocol,condition=arm,**summaries[arm]['full'],pure_bacc=summaries[arm]['pure']['macro_bacc'],pure_auc=summaries[arm]['pure']['macro_auc']))
        gate=paired_gate(records['selected_procedure'],records['pure_reference'])
        for k,d in enumerate(done):
            for a in d['ablations']:all_rows.append(dict(model='lr',protocol=protocol,outer_fold=k,stage='outer',condition=a['id'],config=json.dumps(a['config']),train_rows=a['samples']['retained_rows'],task_fraction=a['samples']['task_fraction'],validation_rows=a['full']['pooled']['n'],macro_bacc=a['full']['macro_bacc'],macro_auc=a['full']['macro_auc'],score=a['full']['score'],pure_macro_bacc=a['pure']['macro_bacc'],pure_macro_auc=a['pure']['macro_auc'],seconds=a['seconds']))
        result[protocol]=dict(conditions=summaries,promotion=gate,selected_configs=[dict(outer_fold=k,**next(a['config'] for a in d['ablations'] if a['id']==d['selected'])) for k,d in enumerate(done)])
    result['lr_promotion']={**result['record']['promotion'], 'date_gain':result['date']['promotion']['mean_gain']}
    result['lr_promotion']['passed']=bool(result['record']['promotion']['record_passed'] and result['date']['promotion']['mean_gain']>=0)
    result['lr_executed']=json.loads(Path('experiments/round6/lr_complete.json').read_text())
    if Path('experiments/round6/cnn_complete.json').exists():
        root=Path('experiments/round6/cnn');names=[p.name for p in (root/'outer0').iterdir() if p.is_dir()];neural={};neural_records={}
        selected=[]
        for k in range(3):
            choices=[json.loads((root/f'outer{k}'/name/'complete.json').read_text()) for name in names]
            choice=max(choices,key=lambda c:c['inner']['best_inner_score']);selected.append(choice['condition'])
            for c in choices:all_rows.append(dict(model='cnn',protocol='record',outer_fold=k,stage='outer',condition=c['condition'],config=json.dumps(c['config']),train_rows=c['refit']['training_policy_counts']['retained_rows'],task_fraction=c['refit']['training_policy_counts']['task_fraction'],validation_rows=c['full']['pooled']['n'],macro_bacc=c['full']['macro_bacc'],macro_auc=c['full']['macro_auc'],score=c['full']['score'],pure_macro_bacc=c['pure']['macro_bacc'],pure_macro_auc=c['pure']['macro_auc'],seconds=c['inner']['seconds_this_invocation']+c['refit']['seconds_this_invocation'],actual_inner_epochs=c['inner']['actual_epochs'],best_epoch=c['inner']['best_epoch'],parameters=c['inner']['parameters'],gpu_peak_mb=c['inner']['gpu_peak_mb']))
        for name in [*names,'selected_procedure']:
            frames=[pd.read_csv(root/f'outer{k}'/(selected[k] if name=='selected_procedure' else name)/'outer_predictions.csv',dtype={'record':str}) for k in range(3)]
            neural[name],neural_records[name]=aggregate(frames,.5,f'cnn_{name}')
            tables.append(dict(model='cnn',protocol='record',condition=name,**neural[name]['full'],pure_bacc=neural[name]['pure']['macro_bacc'],pure_auc=neural[name]['pure']['macro_auc']))
        reference=pd.read_csv('results/round6/lr_record_pure_reference_records.csv',dtype={'record':str})
        result['cnn']=dict(conditions=neural,selected_conditions=selected,promotion=paired_gate(neural_records['selected_procedure'],reference),long_vs_short=paired_gate(neural_records['long150_pure'],neural_records['legacy6_pure']),executed=json.loads(Path('experiments/round6/cnn_complete.json').read_text()))
    for path in Path('experiments/round6/lr').rglob('*.json'):
        if path.name in ('selection.json','complete.json','splits.json','split.json'):continue
        r=json.loads(path.read_text())
        if 'fits' not in r:continue
        all_rows.append(dict(model='lr',protocol=path.parent.as_posix(),outer_fold=None,stage='inner',condition=r['id'],config=json.dumps(r['config']),macro_bacc=r['metrics']['macro_bacc'],macro_auc=r['metrics']['macro_auc'],score=r['inner_score'],seconds=r['seconds'],fits=len(r['fits'])))
    pd.DataFrame(all_rows).to_csv('results/transition_experiments.csv',index=False);pd.DataFrame(tables).drop(columns='pooled').to_csv('results/round6/ablation_summary.csv',index=False)
    Path('results/round6/summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(lr=result['lr_promotion'],cnn=result.get('cnn',{}).get('promotion'),cnn_long_vs_short=result.get('cnn',{}).get('long_vs_short')),indent=2),flush=True)


if __name__=='__main__':run()
