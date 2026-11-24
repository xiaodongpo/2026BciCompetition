# 训练入口

以下命令均从仓库根目录执行。`run_route.py` 在独立子进程中切换到路线目录，保持原有 `src` 和 `scripts` 导入关系。`--data-root` 通过 `BCI_DATA_ROOT` 传入数据加载器，算法本身不读取个人绝对路径。输出落在对应路线的 `cache/`、`experiments/`、`checkpoints/`，不进入Git。

## H1x5 决赛最终配置

真实配方在 `routes/h1/configs/fin_recipes.json`。H1x5为四成员：参考分位树0.20、带通陷波严格树/LR 0.10、原波形HybridNet 0.35、带通陷波HybridNet 0.35。两个树成员各3个种子，两个网络成员各5个种子。网络训练随机初始化，正式模型不依赖BIOT或SingLEM预训练权重。

```bash
python scripts/run_route.py --data-root /path/to/authorized_data h1 scripts.finals.build_cache
python scripts/run_route.py --data-root /path/to/authorized_data h1 scripts.finals.build_cache --bpn
python scripts/run_route.py --data-root /path/to/authorized_data h1 src.training.train_hybrid_fin --name hybfin_base_aug5 --wave base --train_orders aug --lag 30 --guard 15 --epochs 30 --seeds 5 --export all --device cuda
python scripts/run_route.py --data-root /path/to/authorized_data h1 src.training.train_hybrid_fin --name hybfin_bpn_aug5 --wave bpn --train_orders aug --lag 30 --guard 15 --epochs 30 --seeds 5 --export all --device cuda
python scripts/run_route.py --data-root /path/to/authorized_data h1 scripts.finals.train_fin --recipe H1x5 --which all --out checkpoints/fin/H1x5_all --jobs 1
```

`--device cpu` 可执行网络训练，但全量训练可能很慢。正式配方为30轮、batch256、AdamW初始学习率0.001、weight decay0.001、OneCycle峰值学习率0.002；每轮随机阶段重排、重新抽取参考、幅值扰动±10%、记录均衡损失权重。不使用测试标签或测试折早停。

Lag30取窗口末端前30秒的官方标签，再检查回溯标签连续稳定性，保护跨度为6+15秒。切换后的前30秒可沿用上一阶段目标，约30–51秒区间屏蔽；这不等于删除切换后的前51秒。具体网格边界由 `lagged` 的逐秒计数器决定。该处理仅用于训练。

`--export all` 合并57条记录最终拟合，只用于交付，不能再把其中11条记录称为该模型的留出集。模型选择已冻结，以上流程不是新的调参建议。原训练入口具有文件存在即跳过的恢复逻辑；修改配置时使用新 `--name`，不要在旧目录混合轮次。树导出入口会写文件，应使用空输出目录。

## R0 初赛 LAG-EnergyProto

```bash
python routes/r0/train_from_scratch.py --stage smoke --device cpu --work-dir cache/r0_smoke
python routes/r0/train_from_scratch.py --stage all --raw-data /path/to/r0_authorized_data --device cuda --workers 1 --work-dir cache/r0_full
```

完整入口顺序执行预处理、独立数组核验、连续流缓存、165维特征、三个CNN种子42/2024/3407各50轮、200轮LightGBM及源原型。参数见 `routes/r0/training/recipe.json`。正式R0用全部57条记录拟合，不能在训练记录上报告独立测试成绩。工作目录必须为空或为同配方已有工作目录。

## D2 流式参考路线

```bash
python scripts/run_route.py --data-root /path/to/authorized_data d2 scripts.local_prepare
python scripts/run_route.py --data-root /path/to/authorized_data d2 scripts.local_reproduce
# 最终拟合，与CV缓存/模型分开，无锁箱评分：
python scripts/run_route.py --data-root /path/to/authorized_data d2 scripts.refit_final
```

`local_prepare`只生成46条开发记录特征；`local_reproduce`按冻结5折训练，成员各3种子，重放20种阶段顺序和原顺序。最终拟合为全部57条记录、成员各5种子。0.8在线分位树+0.2严格树/LR，阈值0.5。`refit_final`由原项目最终拟合入口移植，去除了跨工程导入路径修补。

## 早期基线与过渡研究

```bash
python scripts/run_route.py --data-root /path/to/authorized_data legacy scripts.audit
python scripts/run_route.py legacy scripts.prepare
python scripts/run_route.py legacy scripts.train_classical
python scripts/run_route.py legacy scripts.train_neural --model cnn --epochs 6
python scripts/run_route.py legacy scripts.train_neural --model tcn --epochs 4
```

历史45/12划分、各轮协议位于 `routes/legacy/configs/`。第二至五轮入口为 `scripts.roundN_prepare/train/evaluate`，第六轮为 `scripts.r6_*`；保留其研究实现，未在本次发布中重跑搜索。BIOT/SingLEM属于可选历史实验，需另装 `requirements-transfer.txt` 并合法取得相应权重。正式H1x5无需这些依赖。

所有完整训练命令均需合法数据及足够资源；本次轻量验收不等于重跑这些完整实验。
