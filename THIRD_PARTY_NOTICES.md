# 第三方来源与许可边界

| 组件 | 来源 | 本地检查及处理 |
| --- | --- | --- |
| BIOT适配代码 | [作者仓库](https://github.com/ycq091044/BIOT) | 上游MIT；保留 `routes/legacy/third_party/BIOT_LICENSE`，保留作者版权行；权重不分发 |
| SingLEM适配代码 | [作者仓库](https://github.com/ttlabtuat/SingLEM) | 上游MIT；保留 `routes/legacy/third_party/SINGLEM_LICENSE`；本地参照实现不等同于权重许可证明 |
| R0 / HybridNet | 用户提供的训练与部署工程 | HybridNet注释记载从另一本地研究工程迁入；项目所有者已明确确认MIT公开发布；原来源注释保留 |
| NumPy/SciPy/pandas/scikit-learn/MNE/LightGBM/PyTorch等 | 通过依赖安装 | 不将这些库源码或环境整体打包；实际发行版许可清单由环境元数据审计生成 |
| 官方规则、数据、示例与比赛资料 | 赛事方 | 未发现明确再分发授权，排除原文件和示例数据 |

2026-11-23核对了[BIOT LICENSE](https://raw.githubusercontent.com/ycq091044/BIOT/main/LICENSE)及[SingLEM LICENSE](https://raw.githubusercontent.com/ttlabtuat/SingLEM/main/LICENSE)。代码许可证不能自动证明训练数据、预训练语料或权重可再分发。保留的本地文件SHA用于追溯，不能替代授权。

项目所有者于2026-11-23明确确认采用MIT并公开最终H1x5/R0/D2权重。根目录`LICENSE`已落实MIT，第三方原版权及MIT通知保留。代码库包含研究工具，不作医疗诊断用途承诺。

可选预训练权重应由使用者从作者仓库的官方指引合法获取，并核对 `checkpoints/manifest.json`；不提供未经确认的镜像下载地址。公开仓库不包含原始EEG、事件标签、第三方预训练权重或内部服务器连接配置。所有者授权的34个最终模型/拟合配置文件按清单公开；该授权不扩展到比赛原始数据。
