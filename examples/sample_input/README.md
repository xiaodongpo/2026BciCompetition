# 合成输入

运行 `python scripts/synthetic_smoke.py --out cache/synthetic_smoke` 会生成35秒、250 Hz、单通道微伏数组和测试用模型。固定随机种子，信号由正弦项与随机噪声合成，不包含真实受试者数据。

运行 `python scripts/h1_synthetic_smoke.py --out cache/h1_demo` 会生成小型四成员模型并构造十段30秒参考，检查训练、重载及无跨调用状态。它缩减了种子数和树轮数，不能代替H1x5完整训练。
