# 评价协议

`P(task)` 越大越偏向task。分类阈值通常为0.5，早期第二轮保留0.46。bACC是两类召回率均值，AUC衡量排序；决赛的ACC是普通准确率，不能替换成bACC。

| 路线 | 划分与评价 | 边界 |
| --- | --- | --- |
| 早期1–6轮 | 45开发/12历史留出，开发集3折或嵌套/日期压力测试 | 不再用旧留出选型；无核实受试者ID |
| D2 | 46开发/11锁箱，开发集5折×20顺序及原顺序 | 20排列复用信号，非20独立数据集 |
| R0 | 代码提供57记录最终拟合；历史报告另有采集重叠组评估 | 当前包未提供完整R0分组CV调度器，不声称重建了其历史CV |
| H1 | 46开发5折，调用真实solution接口，每记录3次参考抽取 | 与特征级模拟分开；平均应按记录等权 |
| H1_dev历史锁箱 | 用46记录训练，11记录历史结果 | 已经暴露，不能再参与选择 |
| H1x5 | 57记录最终拟合，网络各5种子 | 不是H1_dev，不把历史锁箱分数冒充最终权重成绩 |

初赛在线适应与决赛固定参考并非同一协议，跨路线不能按AUC直接排名。记录时间有重叠、参与者ID未知，记录级验证不支持未经限定的“跨受试者泛化”结论。

## H1开发集真实接口复现

先准备两类缓存，再训练H1的每折网络（各3种子），然后每折导出和评价。下面以第0折为例，其余折使用1、2、3、4。每折的网络名保持相同，输出为独立fold目录。

```bash
python scripts/run_route.py --data-root /path/to/authorized_data h1 src.training.train_hybrid_fin --name hybfin_base_aug --wave base --folds 0 --seeds 3 --epochs 30 --device cuda
python scripts/run_route.py --data-root /path/to/authorized_data h1 src.training.train_hybrid_fin --name hybfin_bpn_aug --wave bpn --folds 0 --seeds 3 --epochs 30 --device cuda
python scripts/run_route.py --data-root /path/to/authorized_data h1 scripts.finals.train_fin --recipe H1 --which fold0 --out checkpoints/fin/H1_fold0 --jobs 1
python scripts/run_route.py --data-root /path/to/authorized_data h1 scripts.finals.eval_solution --model checkpoints/fin/H1_fold0 --recs fold0 --seeds 14,15,16 --out results/finals/true_H1_fold0.json --predictions-dir cache/evaluation/H1_fold0
```

`build_cache.py`可对所有授权训练记录做无监督逐记录滤波/特征缓存；成员先验、拟合和训练增强只用训练折。缓存存在不等于可以把验证特征用于先验统计。

`--predictions-dir`保存全部计划待测窗的索引、终点标签、评分及是否纳入纯阶段计分；含非公开标签，只存本地cache。重复运行使用新目录，避免覆盖已有预测。统一启动器在对应路线`cache/launcher_runs/`保存命令、源码/配置SHA-256、控制台日志、状态与耗时。

真实接口评价按官方参考抽取和乱序调用程序执行；标签仅用于离线计分。代码只评分完全属于一个已标注阶段的窗口，和第六轮“官方标签端点且历史可混合”的评价口径不同。

## 独立预测表评价

```bash
python scripts/evaluate_predictions.py examples/sample_output/metric_fixture.csv
```

CSV需有 `y,p_task`，可有 `valid`。双类、非空、有限[0,1]概率为前提。该工具输出 pooled 指标；不能用其直接代替决赛按记录等权的汇总。真实记录请分别计分后等权平均。合成表只测试公式，不提供模型性能证据。
