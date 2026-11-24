# 本地GitHub发布候选验收报告

日期：2026-11-23。独立发布目录：`2026BciCompetition_GitHub`。本次目标是整理、核验并准备发布，未继续大规模训练或搜索。最终主模型核实为用户补充工程中的**H1x5**，历史R0/D2及早期研究分别保留。

## 已交付内容

中英文README、数据/安装/训练/评价/推理/架构/实验/复现/规则九份详细文档、四条隔离研究路线、统一命令入口、固定直接依赖与独立环境快照、合成输入生成及输出示例、CPU测试、Windows/Linux Actions配置、模型哈希清单、源码映射与安全/许可审计。准确目录与文件见[release_files.json](release_files.json)。

源目录对应关系、排除项及工程变更详见[审计报告](github_release_audit.md)。原文件只读保留；复制时及验收时对所选源文件做哈希核验。新增产物置于发布目录或原项目cache。未把原始项目整目录初始化成Git。

## 实际执行证据

| 检查 | 实际结果 | 证据与边界 |
| --- | --- | --- |
| 新独立venv安装 | PASS；Python3.11.16、PyTorch2.7.1+cu128；CPU执行；pip check无冲突 | [环境JSON](../results/validation/environment.json)；从本地wheel缓存安装，无继承site-packages；全新在线下载未完成 |
| 独立环境CPU测试 | 14 passed，76.05秒 | [测试JSON](../results/validation/tests.json)；包括嵌套legacy测试、H1/D2/R0合成训练及R0正式入口smoke |
| 干净本地克隆 | 14 passed，35.61秒；无发布目录运行缓存 | [克隆验收](../results/validation/clean_checkout.json)；从初始Git提交克隆，使用同一独立venv；不是网络克隆测试 |
| 实际H1x5权重 | 合成输入的顺序独立、参考重排、缓存隔离、坏窗回退、因果滤波与不改权重检查通过 | [原权重接口JSON](../results/validation/h1_original_weights.json)；原项目环境；不是泛化评价 |
| 协议辅助工具替换 | 5种子×450123点，参考和乱序窗口数组逐元素一致 | [协议比对](../results/validation/protocol_equivalence.json)；与原示例的行为比对，不复制其文件 |
| H1数据接口 | 57记录、114个BDF/事件文件格式及SHA核验通过 | [数据摘要](../results/validation/data_interface.json)；只读，不新算任何锁箱指标 |
| R0真实数据缩小流程 | 1记录；1753清洁窗；1892流式窗；三CNN种子各1轮；树2轮；六文件模型导出/重载；32个推理决策 | [端到端JSON](../results/validation/r0_one_record.json)；原项目环境；不是全57拟合或新性能结论 |
| 新环境R0示例 | 3种子各1优化步、165维特征、树2轮、32决策 | [合成JSON](../results/validation/r0_synthetic.json)与[示例CSV](../examples/sample_output/r0_synthetic_predictions.csv)；数据完全合成 |
| 指标公式 | 二分类方向、AUC、ACC与bACC差异及非法输入测试通过 | `tests/test_release.py`；手工表AUC=1、ACC=0.75、bACC=5/6，不是模型分数 |
| 来源与安全 | 所选原源码未修改；候选全文扫描与清单核验 | [来源摘要](../results/validation/source_integrity.json)、[安全检查](../results/validation/release_scan.json) |
| 命令与模型清单 | H1训练/导出/评价帮助入口、R0 CLI重放、三套完整正式包哈希核验通过 | [命令验收](../results/validation/commands.json)；R0 CLI 32个评分与直接调用一致 |
| 授权公开权重后的版本 | 17 passed，45.12秒；发布目录H1x5完整权重合成接口检查通过 | [公开版本验收](../results/validation/publication.json)；34个授权文件哈希匹配 |

CPU合成测试覆盖模型加载、特征维度、配置、优化、概率范围/方向、任意分块、未来扰动、缓冲上限、重置、缺失与边界、参考缓存隔离。R0冻结适应模式测试尾窗支持；R0/D2默认历史适应及H1参考协议不被据此错误证明为端到端六秒。

安装日志、完整JUnit、原始验收报告、源清单与敏感位置详情留在本地私有审计缓存；公共JSON保留必要摘要和原日志SHA。执行错误也未隐去：在线下载停滞后使用缓存wheel；初版采样率校验因精确浮点比较误报，修正后完整57记录重验通过。最终发布内容不存在这项已修复误报。

## 可复现性分级

**代码可运行：** Windows独立环境及GitHub-hosted Windows/Ubuntu的CPU链路已执行。两类CI完成在线安装、pip check、17项测试、三套权重核验及H1x5正式权重合成检查。可选BIOT/SingLEM扩展未在新环境安装执行。

**小规模流程可复现：** H1/D2/R0合成训练导出与推理通过；R0一条真实记录完成正式训练链的缩小验收。新增逐窗口CSV导出属于可选日志功能，本次未重跑真实H1评价来产生新分数。

**完整竞赛结果未重新复现：** 历史表取自保存的日志/JSON/CSV，未重跑全部CV、超参数搜索、锁箱或官方隐藏测试。R0历史分组CV调度器在当前最终包中不完整，已明确披露。H1_dev锁箱与H1x5正式权重不同，不能借用前者分数声称后者的新验证成绩。

## 实验与版本真实性

保留第3、4、5轮未晋级结果，BIOT预训练未优于随机探针的对应结果、第六轮各条件完整汇总、D2最差顺序、H2未改善H1的结果。开发和历史锁箱指标分别注明协议与种子；没有官方隐藏测试或排名数字。所有者提供的电极、设备和可靠受试者身份仍缺失，不补造。

## 所有者授权与发布范围

所有者已于2026-11-23明确确认：目标[GitHub仓库](https://github.com/xiaodongpo/2026BciCompetition)公开，采用MIT，公开最终权重，不创建Release。34个H1x5/R0/D2最终权重和拟合config共52,228,136字节，以普通Git文件纳入；第三方预训练2项仅保留哈希。原始数据/事件、个人路径、内部连接配置、会话、完整环境与原官方材料不进入历史。BIOT/SingLEM原MIT通知保留，依赖许可另有元数据清单。

已建立本地main分支和初始提交`a346ae8`，并从该提交的干净克隆完成14项测试。公开权重时增加精确文件SHA白名单，以及授权文件可接受、权重被修改会拒绝、未授权BDF仍拒绝三项回归检查。算法训练/推理源码未因发布授权改变。最新本地与远端提交及最终文件数由Git和交付回执记录，避免循环引用本报告自身哈希。

发布前已确认目标仓库为空且为public，当前账号具有admin/push权限，SSH推送预检成功；不覆盖任何已有远端历史，不创建Release或稳定标签。用户第13.5节要求的发布确认已取得，见[授权范围](publication_authorization.json)。CI执行状态以GitHub Actions及后续验收记录为准，不将本机测试冒充CI结果。

## 远端发布验收

公开仓库的首次发布提交为`0a78da746b99f4f01c062e70d02041f3de4a6c93`。远端334个文件的Git blob SHA及字节数与本地审计树完全一致；GitHub识别MIT，匿名下载的H1x5网络文件通过SHA-256核验，Release与tag数量均为0，见[首次上传核验](../results/validation/github_upload_0a78da7.json)。

[GitHub Actions运行45347242025](https://github.com/xiaodongpo/2026BciCompetition/actions/runs/45347242025)已成功：Windows 17项测试19.33秒，Ubuntu 17项测试17.11秒；两个job的在线安装、pip check、安全扫描、三套权重哈希、H1x5正式权重合成检查及R0合成训练全部通过。执行元数据见[CI记录](../results/validation/github_actions.json)。

发布目录内的R0和D2正式权重也通过合成检查：[R0 32决策](../results/validation/r0_published_weights.json)、[D2 35决策且分块一致](../results/validation/d2_published_weights.json)。这些不是新增分类性能结果。

后续提交只追加本节、平台状态说明和验收JSON；算法、测试、依赖、CI配置及模型文件与已通过CI的提交逐字节一致。该文档提交使用`[skip ci]`避免重复运行相同计算，仍执行本地完整发布扫描和文件清单核验。最新文件数由release_files.json记录，不能用首次上传的334推断后续元数据文件数。
