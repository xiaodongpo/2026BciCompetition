# 从源码到验证结果

这是本地发布候选。目标GitHub地址、代码许可、模型再分发权限尚待确认；当前没有可声称已发布的仓库URL。以下使用用户提供的真实地址替换占位值：

```powershell
$RepoUrl = 'REPLACE_WITH_CONFIRMED_GITHUB_URL'
git clone $RepoUrl industrial-eeg
cd industrial-eeg
```

本地检查可直接进入已有发布目录，无需克隆。依次执行以下步骤。

1. 按[安装说明](installation.md)建立Python 3.11独立环境并运行`pip check`。
2. `python -m pytest -q`执行无需真实数据、无需GPU的测试。H1、D2与R0均覆盖训练优化步骤、模型保存/重载和预测；早期测试验证有限窗口，R0/D2验证未来扰动和分块一致性，H1验证顺序独立与参考缓存隔离。
3. `python scripts/h1_synthetic_smoke.py --out cache/h1_demo`生成四成员结构的极小测试模型并验证接口。每类网络只训练一个种子和一个优化步骤，**不是H1x5训练结果**。
4. `python scripts/synthetic_smoke.py --out cache/r0_demo`生成R0合成模型、输入和逐决策输出。`python scripts/evaluate_predictions.py examples/sample_output/metric_fixture.csv`检查独立指标入口；该手工概率表的AUC不代表模型性能。
5. 取得合法比赛数据，按[数据说明](dataset.md)存放。`python scripts/validate_data.py --data-root /path/to/authorized_data --output cache/data_manifest.json`只读检查57记录、BDF/事件结构并记录SHA-256。该文件含原始文件元数据，应留在本地。核对对应路线的数据版本，不能混用原始与修订数据。
6. 按[训练说明](training.md)建立H1两类缓存。先用[评价说明](evaluation.md)的逐折命令重建46记录开发集H1（五折、每类网络三种子），按记录等权汇总，保留每次运行配置、划分、模型、预测与耗时。不要查看锁箱以决定新参数。
7. 最终配置已经冻结在`routes/h1/configs/fin_recipes.json`。执行H1x5两类网络各五种子、全57记录拟合与树模型导出命令。**最终全量模型不再有这57记录中的独立留出集**。本次整理不重开模型选择。
8. 使用自己的新模型时保留训练生成的配置与全部模型文件。若取得已授权原模型，运行`python scripts/verify_weights.py --route h1 --model /path/to/model`核对历史清单；自行重训的权重不要求与历史SHA逐字节相同。
9. 按[推理说明](inference.md)设置`FIN_MODEL_DIR`并调用H1接口，或运行R0连续流CLI。可用`python scripts/h1_synthetic_smoke.py --model /path/to/model --out cache/h1_bundle_check`对完整H1模型做合成接口测试。
10. 真实评价保存每记录指标、参考种子14/15/16及预测，再按协议汇总。结果与[历史实验](experiments.md)分开保存，记录软件/硬件、数据哈希和配置差异，不覆盖原表。

## 本次验证到哪一层

| 层次 | 本次证据 | 尚未证明 |
| --- | --- | --- |
| 代码可运行 | 新建独立环境、14项测试、原H1x5权重合成推理、原/重写协议逐元素比对 | 所有可选研究脚本与所有平台都可运行 |
| 小规模流程可复现 | H1/D2/R0合成训练导出；R0一条真实记录完整数据处理、独立核验、三种子各1轮、2轮树、模型重载 | H1/D2完整真实数据训练与所有历史搜索 |
| 结果可复现 | 固定合成指标表公式验证；历史JSON/CSV来源与汇总可核验 | 本次没有重跑完整CV、锁箱或官方隐藏测试，不宣称竞赛指标已重新复现 |

R0一条真实记录验证在原有项目环境运行；独立环境执行的是CPU合成与接口测试。两者不能混写为“独立环境全量复现”。具体执行证据见[发布报告](../reports/github_release_report.md)。

本地初始提交的干净克隆不包含原工作区缓存或权重，仍通过全部14项测试（35.61秒），见[克隆验收](../results/validation/clean_checkout.json)。它验证了Git纳入文件的完整性，不代表已经测试远程网络克隆或Linux。

## 已知边界

记录编号不是受试者ID；重复/重叠记录可能使记录级划分偏乐观。早期45/12和D2/H1的46/11划分不同，旧留出已有历史暴露，均不可再当新锁箱调参。R0当前包提供最终拟合入口，缺少其历史完整分组CV调度器，不能保证重建全部R0研究指标。

H1固定参考和全局因果IIR、R0/D2无标签历史适应均超过端到端六秒支持；早期严格窗口路线须单独讨论。因果与有限感受野不是同一个约束。随机种子控制不保证不同硬件和库版本逐位一致。没有设备、电极、可靠参与者身份信息，不能补造元数据或将窗口数当独立样本数。
