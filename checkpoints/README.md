# 最终模型权重

项目所有者于2026-11-23明确同意公开最终权重并采用MIT。仓库提供三个完整最终包，共34个文件、52,228,136字节（约52.2 MB）：

| 模型 | 目录 | 文件数 | 字节数 |
| --- | --- | ---: | ---: |
| 决赛H1x5 | `checkpoints/h1/` | 17 | 22069467 |
| 初赛R0 LAG-EnergyProto | `checkpoints/r0/` | 6 | 734986 |
| 历史D2最终拟合 | `checkpoints/d2/` | 11 | 29423683 |

使用普通Git文件随源码获取，无需Git LFS，也不创建GitHub Release。原始EEG、事件标签和可选第三方预训练权重不包含。`manifest.json`仍记录36项来源：其中34项有`owner_approved=true`及`published_path`，另2项第三方历史权重仅记录哈希，不分发。

```bash
git clone https://github.com/xiaodongpo/2026BciCompetition.git
cd 2026BciCompetition
python scripts/verify_weights.py --route h1 --model checkpoints/h1
python scripts/verify_weights.py --route r0 --model checkpoints/r0
python scripts/verify_weights.py --route d2 --model checkpoints/d2
python scripts/h1_synthetic_smoke.py --model checkpoints/h1 --out cache/h1_published_check
```

H1x5需要config及全部6个树、10个网络文件；配置含先验和LR参数。R0需要config、3个CNN、LightGBM和statistics。D2需要config及10个树文件。不要混合不同版本的配置与权重。完整哈希核验通过后按[推理说明](../docs/inference.md)使用。

这些均为全57记录最终拟合模型，不能把其中记录当作独立测试集。H1_dev历史锁箱成绩不能当成H1x5权重的新评价结果。自行重训不要求逐字节匹配历史权重。

MIT许可仅覆盖本仓库权利人授权的代码与模型文件，并不授予原始赛事数据或第三方预训练文件的再分发权。保留[第三方声明](../THIRD_PARTY_NOTICES.md)。合成模型生成命令仍可用于无数据训练验收，不能据其报告比赛性能。
