# 模型文件

所有正式模型权重均默认排除。`manifest.json`记录当前找到的H1x5、R0、D2及历史预训练文件的大小和SHA-256，用于本地来源核验；拥有哈希不代表拥有再分发许可。

完整原模型包可运行`python scripts/verify_weights.py --route h1 --model /path/to/model`核验，其他路线替换为`r0`或`d2`。清单共36项，包含H1/R0/D2的拟合配置；不把含先验统计的配置直接公开。自行重训的模型不要求与历史哈希相同。

H1x5必须有完整 `config.json`、两个成员各3个LightGBM文本模型、两个HybridNet成员各5个 `.pt`。配置还包含训练先验和LR参数，不能只拿网络权重运行。通过 `FIN_MODEL_DIR`指定目录。

R0需要 `config.json`、`energy_seed42.pth`、`energy_seed2024.pth`、`energy_seed3407.pth`、`lightgbm.txt`、`statistics.npz`。D2需配置及其引用的全部树文件；LR和先验存于配置中。

没有已授权的公开权重下载地址。请按[训练说明](../docs/training.md)自行从合法数据训练，或在作者确认后获取模型。大型权重即使通过Git LFS或Releases分发，也仍需数据和模型权利方授权。

无数据快速测试使用 `python scripts/synthetic_smoke.py` 或 `python scripts/h1_synthetic_smoke.py --out cache/h1_demo`。生成的模型仅用于接口测试，不能报告分类性能。
