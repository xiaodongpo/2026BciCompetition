"""Pre-register the new search before inspecting any new validation outcomes."""
import json,hashlib
from pathlib import Path

def ranked(records,salt):return sorted(records,key=lambda r:hashlib.sha256((salt+r).encode()).hexdigest())

def run():
    old=json.loads(Path('configs/splits.json').read_text());dev=old['development']
    order=ranked(dev,'round2-20261103-outer-')
    folds=[sorted(order[k::3]) for k in range(3)]
    candidates=[dict(id='legacy_C10',features='legacy',C=10,record_weight=False)]
    for family in ('legacy','robust','filterbank','singlem','singlem_random','fusion'):
        for c in (.1,1,10):candidates.append(dict(id=f'{family}_record_C{c}',features=family,C=c,record_weight=True))
    config=dict(round=2,seed=20261103,development=dev,excluded_previous_holdout=old['holdout'],outer_folds=folds,
        inner_folds=3,inner_split_salt='round2-inner-',candidates=candidates,
        fixed_reference=dict(features='legacy',C=10,record_weight=False,threshold=.49),
        selection='Mean of per-record bACC@0.5 and per-record AUC from inner OOF; deterministic candidate order breaks ties',
        threshold='Inner OOF record-and-class-balanced bACC on 0.35:0.01:0.65; nearest 0.5 wins ties',
        date_stress='Nested leave-one-acquisition-date-out on the same 45 development records; dates are groups, never subject/device identity or input features',
        promotion='Promote only if record-paired bootstrap 95% lower bound of mean(bACC,AUC) gain > 0, all 3 outer folds improve that score, and date-stress mean score does not decrease',
        limitations='Previously studied development records: nested CV prevents current-fold tuning leakage but cannot undo prior adaptive research. No new pristine holdout and no subject-generalization claim.',
        raw_support_samples=1500,fs=250,stride_samples=500)
    p=Path('configs/round2/protocol.json')
    if p.exists():assert json.loads(p.read_text())==config,'Do not mutate preregistration'
    else:p.write_text(json.dumps(config,indent=2))
    Path('reports/round2/protocol.md').write_text('''# 第二轮预先固定协议

用户确认无法取得受试者、电极及设备信息。仅采用真实记录ID和文件名日期分组，不推断身份。原12条留出完全排除，不参与任何本轮特征提取、预训练、阈值选择或评价。

原45条开发记录按固定哈希划分3个外折；每个外折的训练记录再做3折内层验证。19个候选的特征、正则与权重选项在configs/round2/protocol.json冻结。所有候选的scaler/分类器只拟合当前训练记录。外折预测仅生成内层选定的模型和预先固定的上一轮基线，不通过外折挑另一个方案。

选择指标为记录宏平均的(bACC@0.5+AUC)/2，阈值从内层OOF选取。新候选采用记录和类别等权训练，减少长记录占比差异；旧基线保留原窗口级类别权重作为固定对照。主结果描述的是整个内层选模流程，而不是挑最高外折成绩的模型。

另外独立执行按采集日期留一组的嵌套压力测试，每个日期外折仍只在其训练日期上选模型。日期不作为模型特征，也不被称为设备/受试者标签。由于开发数据已经在上一轮研究过，新的嵌套评价仍带历史研究偏倚，不能重命名为新的独立测试集。

升级门槛：记录级配对bootstrap的平均(bACC,AUC)差值95%区间下界>0，3个记录外折同向改善，且日期压力测试平均分不下降；否则保留现有正式提交包，将本轮模型作为研究候选交付。所有判断在运行前固定。
''',encoding='utf8')
    print('Frozen',len(candidates),'candidates; excluded:',old['holdout'])

if __name__=='__main__':run()
