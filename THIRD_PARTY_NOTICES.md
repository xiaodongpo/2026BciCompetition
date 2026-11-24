# 第三方来源与许可边界

| 组件 | 来源 | 本地检查及处理 |
| --- | --- | --- |
| BIOT适配代码 | [作者仓库](https://github.com/ycq091044/BIOT) | 上游MIT；保留 `routes/legacy/third_party/BIOT_LICENSE`，保留作者版权行；权重不分发 |
| SingLEM适配代码 | [作者仓库](https://github.com/ttlabtuat/SingLEM) | 上游MIT；保留 `routes/legacy/third_party/SINGLEM_LICENSE`；本地参照实现不等同于权重许可证明 |
| R0 / HybridNet | 用户提供的训练与部署工程 | HybridNet注释记载从另一本地研究工程迁入；发布前由权利方确认原创代码及团队贡献的授权范围 |
| NumPy/SciPy/pandas/scikit-learn/MNE/LightGBM/PyTorch等 | 通过依赖安装 | 不将这些库源码或环境整体打包；实际发行版许可清单由环境元数据审计生成 |
| 官方规则、数据、示例与比赛资料 | 赛事方 | 未发现明确再分发授权，排除原文件和示例数据 |

2026-11-23核对了[BIOT LICENSE](https://raw.githubusercontent.com/ycq091044/BIOT/main/LICENSE)及[SingLEM LICENSE](https://raw.githubusercontent.com/ttlabtuat/SingLEM/main/LICENSE)。代码许可证不能自动证明训练数据、预训练语料或权重可再分发。保留的本地文件SHA用于追溯，不能替代授权。

建议对经权利方确认的项目原创代码采用MIT，保留第三方MIT通知。许可证尚待用户确认，根目录 `LICENSE` 当前是待确认声明，尚不是MIT授权。代码库包含研究工具，不作医疗诊断用途承诺。

可选预训练权重应由使用者从作者仓库的官方指引合法获取，并核对 `checkpoints/manifest.json`；不提供未经确认的镜像下载地址。公开仓库不包含原始EEG、事件标签、训练权重或内部服务器连接配置。
