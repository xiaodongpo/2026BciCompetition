"""Audit prior DL budgets; missing validation-loss/LR logs stay missing."""
import json
from pathlib import Path
import pandas as pd


def run():
    rows=[];curves=[]
    for path in sorted(Path('experiments').glob('*/training.jsonl')):
        if not any(k in path.parent.name for k in ('cnn','tcn','biot')):continue
        logs=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        frame=pd.DataFrame(logs)
        for fold,g in frame.groupby('fold'):
            last=int(g.epoch.max());best=g.loc[((g.bacc+g.auc)/2).idxmax()] if 'bacc' in g and 'auc' in g else None
            rows.append(dict(experiment=path.parent.name,fold=int(fold),completed=(path.parent/'summary.json').exists(),actual_epochs=last,checkpoint_epoch=last,best_observed_outer_epoch=int(best.epoch) if best is not None else None,best_is_diagnostic_not_selected=True,training_loss_first=float(g.loss.iloc[0]),training_loss_last=float(g.loss.iloc[-1]),validation_loss_logged='val_loss' in g,lr_logged='lr' in g,early_stopping=False))
        frame['experiment']=path.parent.name;curves.append(frame)
    pd.DataFrame(rows).to_csv('reports/round6/prior_neural_audit.csv',index=False)
    pd.concat(curves).to_csv('results/round6/prior_neural_curves.csv',index=False)
    print(pd.DataFrame(rows)[['experiment','fold','completed','actual_epochs','best_observed_outer_epoch']].to_string(index=False))


if __name__=='__main__':run()
