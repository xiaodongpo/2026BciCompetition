"""Generate the three requested reports and standalone figures from executed artifacts."""
import json,hashlib,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.r6_diagnostics import LABELS

ROOT=Path('reports/round6')
COLORS={'relax_to_task':'#216b9b','task_to_relax':'#bd5a33'}


def link(name):return (ROOT/name).resolve().as_posix()
def write(name,lines):Path('reports',name).write_text('\n'.join(lines)+'\n',encoding='utf-8')
def ci(r):return f"{r['mean']:+.4f} [{r['low']:+.4f}, {r['high']:+.4f}]"


def plots(s):
    bins=pd.read_csv('results/round6/directional_bins.csv');curves=pd.read_csv('results/round6/aligned_transition_curves.csv')
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,(direction,color) in enumerate(COLORS.items()):
        g=curves[(curves.direction==direction)&(curves.variable=='p')].sort_values('relative_bin');ax=axs[0,col]
        xx=g.relative_bin.to_numpy()+1;ax.plot(xx,g['mean'],color=color);ax.fill_between(xx,g.low,g.high,color=color,alpha=.15);ax.axvline(0,color='#333',lw=1);ax.axvspan(0,6,color='#999',alpha=.15);ax.axhline(.46,color='#666',ls='--',lw=.8);ax.set(title=direction.replace('_',' '),xlabel='Seconds from event start (2s bins)',ylabel='Cross-fitted P(task)',ylim=(.1,.9))
        ax=axs[1,col]
        for scope,style in [('all','-'),('pure','--')]:
            g=bins[(bins.scope==scope)&(bins.direction==direction)&(bins.variable=='error')].set_index('bin').reindex(LABELS)
            ax.plot(range(9),g['mean'],style,marker='o',color=color,alpha=1 if scope=='all' else .6,label=scope)
            if scope=='pure' and np.isfinite(g.loc['4-6','mean']):
                ax.annotate(f"pure: n={int(g.loc['4-6','n_records'])} records",(2,g.loc['4-6','mean']),xytext=(2.6,min(.94,g.loc['4-6','mean']+.08)),fontsize=8,color='#555',arrowprops=dict(arrowstyle='-',color='#888',lw=.7))
        ax.set_xticks(range(9),LABELS,rotation=35);ax.set(xlabel='Seconds after phase start',ylabel='Record-macro error rate',ylim=(0,1));ax.legend()
    fig.suptitle('Bidirectional transitions: record-level uncertainty; shaded first 6s can mix phases');fig.savefig(ROOT/'transition_probability.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,var in zip(axs.flat,['log_theta','log_alpha','log_beta','log_variance','spectral_entropy','pe_m3_d4']):
        for direction,color in COLORS.items():
            g=bins[(bins.scope=='all')&(bins.direction==direction)&(bins.variable==var)].set_index('bin').reindex(LABELS)
            xx=np.arange(9);ax.plot(xx,g['mean'],label=direction,color=color);ax.fill_between(xx,g.low,g.high,color=color,alpha=.12)
        ax.set_xticks(range(9),LABELS,rotation=35);ax.set(title=var,xlabel='Seconds after phase start')
    axs[0,0].legend(fontsize=8);fig.suptitle('Offline EEG descriptors: trial then record averaging, not physiological ground truth');fig.savefig(ROOT/'transition_features.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(14,12),layout='constrained');names=list(s['cnn']['conditions']);names.remove('selected_procedure')
    colors=plt.cm.tab10(np.arange(len(names)))
    for k in range(3):
        for name,color in zip(names,colors):
            h=pd.read_json(f'experiments/round6/cnn/outer{k}/{name}/inner/history.jsonl',lines=True)
            axs[k,0].plot(h.epoch,h.train_hard_loss,label=name,color=color);axs[k,1].plot(h.epoch,h.val_score,label=name,color=color)
        axs[k,0].set(title=f'Outer {k}: training hard-label probe BCE',xlabel='Epoch',ylabel='BCE (fixed training subset)')
        axs[k,1].set(title=f'Outer {k}: inner record validation',xlabel='Epoch',ylabel='(bACC@0.5 + AUC) / 2')
    axs[0,0].legend(fontsize=8);fig.suptitle('CNN budget and supervision ablations: outer validation never drives stopping');fig.savefig(ROOT/'cnn_learning_curves.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(14,11),layout='constrained')
    for k in range(3):
        for name,color in zip(names,colors):
            h=pd.read_json(f'experiments/round6/cnn/outer{k}/{name}/inner/history.jsonl',lines=True)
            axs[k,0].plot(h.epoch,h.val_loss,label=name,color=color);axs[k,1].semilogy(h.epoch,h.lr,label=name,color=color)
        axs[k,0].set(title=f'Outer {k}: complete inner validation BCE',xlabel='Epoch',ylabel='Hard official-label BCE')
        axs[k,1].set(title=f'Outer {k}: learning rate actually used',xlabel='Epoch',ylabel='Learning rate')
    axs[0,0].legend(fontsize=8);fig.suptitle('Validation loss and learning-rate histories, including early stopping');fig.savefig(ROOT/'cnn_validation_loss_lr.png',dpi=160);plt.close(fig)
    psd=pd.read_csv('results/round6/directional_log_psd.csv');freq=np.load('cache/round6/diagnostic_psd.npz')['frequencies'];fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,direction in zip(axs,COLORS):
        for name in ['6-10','10-15','20-30','60+']:
            g=psd[(psd.direction==direction)&(psd.bin==name)]
            if len(g):ax.plot(freq,g.iloc[0,2:].to_numpy(float),label=name)
        ax.set(title=direction,xlabel='Hz',ylabel='Mean log10 PSD (filtered microvolt signal)');ax.legend(title='Seconds')
    fig.savefig(ROOT/'transition_psd.png',dpi=170);plt.close(fig)


def transition_report(s):
    prepared=json.loads(Path('cache/round6/prepared.json').read_text());bins=pd.read_csv('results/round6/directional_bins.csv');contrast=pd.read_csv('results/round6/early_stable_contrasts.csv');inventory=pd.read_csv('reports/round6/transition_inventory_all57.csv',dtype={'record':str});latent=pd.read_csv('results/round6/latent_distance_summary.csv')
    lines=['# 双向状态切换分析（2026-11-05完成）','',
        '**直接观察：两个方向的阶段初期都较难分类，而且差异持续到整窗已经进入新阶段之后。方向曲线不相同，但错误率增量的不对称性尚不足以确定。以下不能证明生理过渡时长，也不能排除运动、接触变化或未记录的采集时延。**','',
        '## 对齐与评价对象','',
        f"重新读取57份BDF及事件，配对得到{len(inventory)}个完整阶段。使用文件名的北京时间毫秒锚点，以真实事件时间计算预测末端相对时间；窗口范围仍为末端已到达的1500点。相邻异类阶段间隙0–1秒定义有效切换，不依赖严格交替假设。全部记录审计表见`reports/round6/transition_inventory_all57.csv`。",'',
        f"45条开发记录包含135次relax→task和90次task→relax；最初45个relax阶段不当作任务恢复。旧12条留出只参与文件和事件结构审计，不作新特征或性能分析。完整末端标签集合{prepared['windows']:,}窗，其中纯阶段{prepared['pure_windows']:,}窗；{prepared['mixed_windows']}窗含另一阶段、{prepared['unlabeled_history_windows']}窗含未标注历史，两类可重叠。纯阶段子集与原缓存逐点一致。",'',
        '主评价保留所有有官方末端标签且整窗信号有效的输出；不因过渡困难而删除。缺失、无标签末端及不足6秒的输出不计性能。模型输入没有事件或时间变量。以下使用每条记录未参与拟合的第二轮固定配置外折预测；不加载全45记录模型来评价同45记录。','',
        '## 双向概率和错误','',f'![双向模型轨迹]({link("transition_probability.png")})','',
        '曲线先在trial内汇总，再在记录内汇总，阴影为记录bootstrap区间；不能将重叠窗当独立观测。前6秒存在窗口混合，灰色区域仅提示信号支持范围。单方向均为单一真实类别，不单独计算AUC。','',
        '| 时间段（秒） | task Recall | task平均P(task) | relax Specificity | relax平均P(task) |','|---|---:|---:|---:|---:|']
    for bin_name in LABELS:
        cells=[]
        for direction in COLORS:
            g=bins[(bins.scope=='all')&(bins.direction==direction)&(bins.bin==bin_name)].set_index('variable');cells.extend([1-g.loc['error','mean'],g.loc['p','mean']])
        lines.append('| '+bin_name+' | '+' | '.join(f'{v:.4f}' for v in cells)+' |')
    lines+=['','## 混合窗以外的证据与不对称性','',
        '以下只比较完全同阶段窗口：新阶段6–20秒减去该阶段60秒后，先trial再record汇总；两方向再按同一记录配对。60秒后只是预先定义的晚期参照，未被验证为生理稳定期。少量纯窗末端处于5.996–6秒附近，因此4–6秒箱也可能有纯窗；图中注明其较小记录数，不与完整45记录曲线等同解释。', '',
        '| 对比 | 错误率增量及95%记录区间 |','|---|---:|']
    for direction in ['relax_to_task','task_to_relax','directional_asymmetry']:
        r=contrast[(contrast.variable=='error')&(contrast.direction==direction)].iloc[0];lines.append(f'| {direction} | {ci(r)} |')
    lines+=['','两方向早期错误增量均高于零，说明问题不能仅用6秒窗口跨界解释。方向差的区间跨零，不支持确定谁在生理上更慢。task概率在30–60秒段已有变化，relax在该段仍有较高误报，这只是群体模型响应；模型阈值、训练偏倚和信号变化都可能影响它。未强行拟合个体过渡长度或时间常数。','',
        '## 特征、PSD及潜在表征','',f'![特征动态]({link("transition_features.png")})','',f'![PSD]({link("transition_psd.png")})','',
        '复用既有窗内FIR、鲁棒频谱、Hjorth和排列熵；Delta/Theta/Alpha/Beta绝对及相对功率、Theta/Alpha和Beta/Alpha比、时域方差、复杂度和熵的完整曲线均保存。功率是1–45Hz预处理后的微伏信号描述，不能理解为已分离的神经源活动。','',
        '| 方向 | 变量 | 纯早期减稳定期及95%区间 |','|---|---|---:|']
    for direction in COLORS:
        for var in ['log_theta','log_alpha','log_beta','log_variance','pe_m3_d4']:
            r=contrast[(contrast.direction==direction)&(contrast.variable==var)].iloc[0];lines.append(f'| {direction} | {var} | {ci(r)} |')
    lines+=['','这些特征变化与概率变化共同支持“阶段初期的数据分布不同”的观察，但不确定其原因。频带结果为探索性多变量分析，未作确认性多重比较推断。也不能据此认定官方标签错误。','',
        'CNN潜在表征另在`results/round6/latent_distance_summary.csv`汇总：仅在同一外折模型、同一trial内计算到60秒后平均表征的距离，再按记录汇总。**此距离使用未来晚期段，仅作离线诊断，绝不进入模型推理或选模。** 各网络坐标不直接跨折平均，也不将距离当作生理状态。', '',
        '| 方向 | 6–10秒平均表征距离 | 60秒后平均表征距离 |','|---|---:|---:|']
    for direction in COLORS:
        g=latent[latent.direction==direction].set_index('bin');lines.append(f"| {direction} | {g.loc['6-10','mean']:.4f} | {g.loc['60+','mean']:.4f} |")
    lines+=['','两方向早期到晚期中心的平均距离较大；该统计依赖模型尺度和中心估计，且晚期样本参与了中心计算，不能作为独立的过渡检验。', '',
        '## 观察、解释与未验证假设','',
        '- 数据直接支持：纯阶段早期仍有更高错误，功率等描述也随阶段时间变化；两方向概率轨迹不同。',
        '- 可能解释：认知建立/恢复、动作和肌电、接触/增益变化、事件指令与实际操作间隔或未记录采集延迟，都可能贡献；当前单通道和缺失元数据无法区分。',
        '- 尚未验证：真实生理过渡时长、稳定的个体不对称性，以及辅助隐状态是否有独立预测价值。没有自动移动事件时间或把模型判断当真实标签。','',
        '相关原始研究在其他多任务实验中观察过方向与频段有关的切换差异：[Bowers等，2014](https://doi.org/10.1177/1541931214581046)。这只支持研究假设，不是本数据的生理证明。所有结果仍来自反复使用过的开发集，记录级区间并非受试者区间。']
    write('bidirectional_transition_analysis.md',lines)


def ablation_report(s):
    table=pd.read_csv('results/round6/ablation_summary.csv');gate=s['lr_promotion'];e=s['lr_executed'];c=e['selected']['config'];phase=pd.read_csv('results/round6/phase_metrics.csv')
    lines=['# 双向过渡期训练消融（2026-11-05完成）','',
        '**本轮LR训练策略带来小幅平均增益，但记录配对区间跨零，未通过升级门槛。继续保留第二轮提交包；本轮候选仅作研究。**','',
        '## 隔离与方法','',
        '所有方案使用相同完整外层验证窗，包含跨界及阶段初期，不剔除困难验证样本。固定对照为第二轮125维LR(C10)、阈值0.46，在各外折训练记录上重训。3记录外折与4日期外折均包含3内折，最终45记录另做内层选择。参数从内层完整验证集的记录宏平均(bACC+AUC)/2选择，外层仅作报告。','',
        '先单独搜索T_task/T_relax∈{0,6,15,30}，再组成独立组合，并与对称15秒、15/30及30/15秒剔除比较；线性降权考察单向/双向与最低权重0.25/0.5；软标签从0.5过渡到官方标签。初始阶段不套用方向参数。原请求0/2/4/6/10/15/20/30秒全部做样本影响统计，实际拟合只用上述缩减网格，避免完整笛卡尔积。', '',
        '剔除后重新平衡记录/官方类别总权重，scaler只看保留训练行；软标签以加权双标签样本实现soft BCE等价目标。官方验证标签从未修改。每个内外折的保留样本数、类别比例、权重、配置和概率均保留，完整实验索引为`results/transition_experiments.csv`。','',
        '## 完整验证集结果','',
        '| 协议 | 流程 | 记录宏bACC | 记录宏AUC | 纯阶段bACC | 纯阶段AUC |','|---|---|---:|---:|---:|---:|']
    for protocol in ('record','date'):
        for name in ('pure_reference','full_hard','selected_procedure'):
            r=table[(table.model=='lr')&(table.protocol==protocol)&(table.condition==name)].iloc[0];lines.append(f'| {protocol} | {name} | {r.macro_bacc:.6f} | {r.macro_auc:.6f} | {r.pure_bacc:.6f} | {r.pure_auc:.6f} |')
    lines+=['',f"内层选择流程相对固定对照的记录综合分增量{gate['mean_gain']:+.6f}，95%记录配对区间[{gate['bootstrap95'][0]:+.6f},{gate['bootstrap95'][1]:+.6f}]；三个外折增量为"+', '.join(f'{v:+.6f}' for v in gate['fold_gains'])+f"。日期增量{gate['date_gain']:+.6f}，记录bACC增量{gate['bacc_gain']:+.6f}。",'',
        '预定升级要求区间下界>0、三个外折同向改善、日期综合分与记录bACC不下降。本轮区间条件未通过。所有指标仅是历史开发集上的记录级证据，不能称为新的独立测试。','',
        '| 方案 | 记录宏bACC | 记录宏AUC | 综合分 |','|---|---:|---:|---:|']
    for r in table[(table.model=='lr')&(table.protocol=='record')].itertuples(index=False):lines.append(f'| {r.condition} | {r.macro_bacc:.6f} | {r.macro_auc:.6f} | {r.score:.6f} |')
    lines+=['','上述固定消融表不用于事后替换嵌套选择流程。independent系列的两个T由各外折自己的内层数据选择，不保证跨折相同。','',
        '## 两方向参数与局部表现','', '| 协议 | 外折 | 选择方案 | T_task | T_relax |','|---|---:|---|---:|---:|']
    for protocol in ('record','date'):
        for r in s[protocol]['selected_configs']:lines.append(f"| {protocol} | {r['outer_fold']} | {r['id']} | {r['T_task']} | {r['T_relax']} |")
    lines+=[f"\n完整45记录最终内层选择`{c['id']}`，T_task={c['T_task']}秒、T_relax={c['T_relax']}秒。只代表本轮候选范围内的训练处理参数，不能解释为生理时长。不同外折可偏向不同不对称关系，不能据此指定统一的建立/恢复速度顺序。",'',
        '| 模型 | 方向 | 阶段秒数 | Recall或Specificity |','|---|---|---|---:|']
    for name in ('lr_record_pure_reference','lr_record_selected_procedure'):
        for direction in COLORS:
            for period in ('0-6','6-20','60+'):
                r=phase[(phase.model==name)&(phase.scope=='all')&(phase.direction==direction)&(phase.period==period)].iloc[0];lines.append(f'| {name} | {direction} | {period} | {r["mean"]:.4f} |')
    counts=pd.read_csv('results/transition_experiments.csv');counts=counts[(counts.model=='lr')&(counts.stage=='outer')]
    lines+=['',f"外折训练保留行数范围{int(counts.train_rows.min())}–{int(counts.train_rows.max())}，原始保留行中task比例范围{counts.task_fraction.min():.4f}–{counts.task_fraction.max():.4f}；实际损失仍作记录/类别等总权重。不同外折训练记录数不同，行数范围不能直接解读为单一策略删样量。每折精确值见实验CSV与对应JSON。",'',
        '完整阶段指标另见`results/round6/phase_metrics.csv`；同时包含两类的阶段集合才计算AUC，见`phase_combined_class_metrics.csv`。局部性能改善不能替代完整集合的主结论。','',
        '## 执行、验证与范围','',
        f"完成{e['planned_inner_fits']}次内层、{e['outer_fits']}次外层消融和1次最终拟合，共{e['total_research_fits']}次LR研究拟合，耗时{e['seconds']:.2f}秒。683个检查点的训练记录、scaler样本数和均值、完整验证行及日志概率均核验；最终候选复拟合系数和概率逐位一致。23个实际LR保存模型通过逐样本、任意分块、未来/过早历史隔离与间断重置，API拒绝phase参数。",'',
        'CNN课程学习及训练轮数消融见`deep_learning_training_optimization.md`。未追加隐式多状态辅助任务：本轮尚不能确认生理过渡标签，也没有证据支持把模型错误转为额外真值。保存的研究候选为`checkpoints/round6/lr_candidate.joblib`，未部署。原始114份BDF/事件文件与两份旧包SHA保持不变。', '',
        '限制：既有开发集复用、未知参与者关系、单通道伪迹可能性、只搜索有限T和权重函数、软标签只是训练假设。未检验的指数/Sigmoid、250/400轮或更大模型不报告性能。']
    write('transition_ablation.md',lines)


def neural_report(s):
    n=s['cnn'];t=pd.read_csv('results/round6/ablation_summary.csv');audit=json.loads((ROOT/'cnn_integrity_audit.json').read_text());prior=pd.read_csv(ROOT/'prior_neural_audit.csv');complete=prior[prior.completed]
    lines=['# 深度学习训练预算与过渡监督优化（2026-11-05完成）','',
        '本轮实际训练了原SmallCNN的6种条件，在3个记录外折中分别以内层整记录验证选择轮数，再重训外层训练记录。模型输入只有1500点EEG，最终概率不读取事件时间。','',
        '## 历史审计','',
        '已完成历史实验：CNN固定6轮、TCN固定4轮、BIOT部分/全量微调固定3轮；均未使用本轮的内层早停。中断的TCN试运行单独标记为未完成。历史CNN/TCN日志有训练损失与外层bACC/AUC，但没有每轮验证BCE、LR或内层最佳checkpoint；BIOT只有每轮训练损失，不能补造未记录曲线。SingLEM为冻结特征探针，没有本地反向训练epoch。','',
        '历史某一epoch的外层分数高低仅作回顾，未用它选择本轮训练轮数。审计表与曲线数据在`reports/round6/prior_neural_audit.csv`、`results/round6/prior_neural_curves.csv`。','',
        '## 本轮控制与训练','',
        '保持原CNN结构、width16、dropout0.25、batch128、AdamW(lr0.001,weight_decay0.01)、gradient clipping=1。legacy6_pure与long150_pure使用相同纯训练窗、原global pos_weight和恒定LR；三外折前6轮权重逐位相同，增加预算的比较有匹配起点。long150_full只加入完整末端标签训练窗；scheduled150_full再加入ReduceLROnPlateau。过渡/课程条件另使用记录类别等权监督，属于监督策略组合。','',
        '上限150轮，patience25；Plateau监视内层硬标签BCE，factor0.5/patience8/min_lr1e-5。最佳epoch按完整内层记录宏平均(bACC@0.5+AUC)/2，随后在外层训练记录上重训到该轮并重放内层LR列表。外层标签从不驱动学习率或停止。课程前15轮只看切换30秒以外，第16–30轮引入过渡样本，31轮以后才允许最佳选择。','',
        '采用一个固定整记录内层epoch验证划分，因此该部分不是三内折平均的epoch选择。过渡策略另由LR的内层选择提供；两种内部调参数据会重叠，但外层验证记录始终隔离。没有声称比较穷尽全部训练预算或网络。','',
        '## 完整验证结果','', '| 条件 | 记录宏bACC | 记录宏AUC | 纯阶段bACC | 纯阶段AUC |','|---|---:|---:|---:|---:|']
    for r in t[t.model=='cnn'].itertuples(index=False):lines.append(f'| {r.condition} | {r.macro_bacc:.6f} | {r.macro_auc:.6f} | {r.pure_bacc:.6f} | {r.pure_auc:.6f} |')
    gate=n['promotion'];gain=n['long_vs_short']
    lines+=['',f"仅延长纯窗CNN训练相对固定6轮的综合分增量{gain['mean_gain']:+.6f}，95%记录配对区间[{gain['bootstrap95'][0]:+.6f},{gain['bootstrap95'][1]:+.6f}]。",'',
        f"CNN内层选择流程相对固定LR的综合分增量{gate['mean_gain']:+.6f}，95%区间[{gate['bootstrap95'][0]:+.6f},{gate['bootstrap95'][1]:+.6f}]。记录级升级条件通过状态：{gate['record_passed']}。本轮未通过时，不追加日期测试或新包，也不声称CNN具有日期泛化证据。",'',
        '| 外折 | 条件 | 实际内层epoch | 最佳epoch / 外折重训epoch | 早停 |','|---:|---|---:|---:|---|']
    rows=n['executed']['results']
    for r in rows:lines.append(f"| {r['outer_fold']} | {r['condition']} | {r['inner']['actual_epochs']} | {r['inner']['best_epoch']} | {r['inner']['early_stopped']} |")
    lines+=['',f'![学习曲线]({link("cnn_learning_curves.png")})','',
        '训练曲线的hard-label BCE来自训练行每8行取1的固定探针，不冒充全部训练行的BCE；实际优化目标及完整验证BCE、bACC、AUC、LR逐epoch都保留。图中可见训练拟合持续改善时验证表现常已停滞或波动，这支持过拟合风险的观察，不能只据低训练损失判定成功。早停允许选择较早epoch，并不保证增加上限提高外层泛化。','',
        f'![验证损失和学习率]({link("cnn_validation_loss_lr.png")})','',
        '学习率图展示实际每轮使用值；调度器只读取内层验证BCE。降低学习率没有保证验证损失或记录综合分改善。对于课程方案，实际训练目标随样本引入而变化，因此不同阶段的优化损失不能直接当作相同样本集合上的拟合程度比较；上图的硬标签探针和完整验证指标另行提供参照。','',
        '## 资源、恢复和推理','',
        f"18次内层训练与18次外层重训完成，共36个训练过程；总计记录内层epoch数{sum(r['inner']['actual_epochs'] for r in rows)}、重训epoch数{sum(r['refit']['actual_epochs'] for r in rows)}。本轮CNN调用耗时{n['executed']['seconds']:.2f}秒；最大已分配显存{max(r['inner']['gpu_peak_mb'] for r in rows):.2f}MiB；模型参数量{rows[0]['inner']['parameters']}。仅使用本地RTX4060，没有付费或外部资源。",'',
        '每轮保存模型、优化器、调度器、Python/NumPy/Torch/CUDA随机状态，保留best/last及epoch6快照。额外恢复试验以两条开发记录从第6轮恢复至第8轮，与不间断8轮的参数逐位一致（审计额外10个训练epoch，未计入36个正式过程）。18个外折模型逐一核对内部训练/验证分组、重训轮数和LR列表。18个实际CNN适配器通过CPU缓存/输出核对、逐样本、未来与历史支持及缺失重置测试。', '',
        '模型潜在表征的阶段变化仅作离线诊断，见双向分析报告；到未来稳定段中心的距离没有进入训练选择或正式推理。推理程序为`src/transition_inference.py`，仅接受EEG块、样本序号、单位和有效性mask，明确拒绝phase。','',
        '## 结论边界','',
        '是否增加轮数有益以完整外层结果为准；最佳epoch只反映本数据、结构与优化器。没有据本轮结果推断所有深度模型无效。TCN和大型预训练模型本轮只审计旧日志，未追加150轮以上训练；250/400轮、Cosine、warmup和复杂辅助任务未执行。优先把本地预算用于同一CNN的可归因消融，已能观察长训和过拟合问题。', '',
        '所有结果仍复用历史开发记录，未知参与者关联可能让记录区间偏乐观。调度器语义按本机版本核对：[PyTorch 2.7文档](https://docs.pytorch.org/docs/2.7/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)。']
    write('deep_learning_training_optimization.md',lines)


def run():
    s=json.loads(Path('results/round6/summary.json').read_text());assert 'cnn' in s
    plots(s);transition_report(s);ablation_report(s);neural_report(s)
    files=[*Path('scripts').glob('r6*.py'),Path('src/transition_training.py'),Path('src/transition_evaluation.py'),Path('src/transition_inference.py'),Path('tests/test_transition.py'),ROOT/'protocol.md']
    with zipfile.ZipFile('experiments/round6/postprocessing_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p)
    artifacts=[Path('reports')/name for name in ('bidirectional_transition_analysis.md','transition_ablation.md','deep_learning_training_optimization.md')]
    (ROOT/'report_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*files,*artifacts,*ROOT.glob('*.png'),Path('results/transition_experiments.csv'),Path('results/round6/summary.json')]},indent=2));print('Wrote all three requested reports and figures.',flush=True)


if __name__=='__main__':run()
