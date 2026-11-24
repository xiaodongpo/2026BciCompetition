# 推理接口

所有接口以微伏输入单通道250 Hz EEG；高分代表task。权重未获再分发授权，默认不随仓库提供。可先执行合成冒烟训练产生接口测试用权重，或按训练文档生成正式模型。

## H1x5 决赛主方案

`routes/h1/finals/submission/solution.py`提供 `filter_eeg(bdf_path)` 与 `predict(window, reference)`。通过 `FIN_MODEL_DIR` 指定完整模型目录。

```python
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path('routes/h1/finals/submission').resolve()))
os.environ['FIN_MODEL_DIR'] = str(Path('routes/h1/checkpoints/fin/H1x5_all').resolve())
import solution
# window: (1500,), reference: (10,7500), both from the same causally filtered record.
score = solution.predict(window, reference)
state = int(score >= 0.5)
```

先对整条记录做二阶100 Hz因果低通，保持采样点一一对应；这一步不读事件、BDF标注、标签或记录时间。随后由评测流程提供六秒窗和十段30秒固定参考；不能把其他待测窗口当作参考。全局因果滤波有IIR历史，固定参考也超出六秒，所以H1x5不是严格端到端六秒路线。

参考仅用于分位数/Z标准化。唯一跨调用缓存以参考内容哈希为键，不在线更新分类器。清洁窗不足750点返回0.5；清洁片段不足完整1500点时禁用网络成员，剩余树成员重新归一化融合。参考不足20个清洁子窗时回退到训练先验。0.5是中性占位，不能理解成确定task。

## R0连续流式方案

```bash
python routes/r0/run_inference.py --input cache/synthetic_smoke/raw_eeg_uv.npy --model cache/synthetic_smoke/model --output cache/r0_predictions.csv --device cpu
```

```python
import sys
sys.path.insert(0, 'routes/r0')
from r0 import R0Decoder
decoder = R0Decoder('path/to/complete_r0_model', device='cpu')
for chunk_uv, valid_mask in incoming_chunks:
    for result in decoder.push(chunk_uv, valid_mask):
        print(result['score'], result['state'], result['valid'])
decoder.reset()  # 新记录时重置，记录内阶段切换不重置
```

前4秒不输出，之后每秒判断。原始缓冲最多1500点；CNN4秒、有限评分聚合最多覆盖6秒，在线归一化、logit中心化和原型使用更早的无标签历史。先预测后更新。无效4秒窗输出 `valid=False,score=0.5,state=1`，该state不可靠；4秒可用但6秒不可用时退回深度分支。必须使用valid字段。

## D2及早期路线

D2的 `src.inference.decoder.StreamingDecoder` 在其路线目录导入，`push`接受任意大小连续块，`Decision.p_task` 为评分。使用历史分位数，异常时可能沿用此前评分；`evaluated=False`表示沿用，不能等同于当前窗新预测。其历史支持超过6秒。

早期 `routes/legacy/src/inference.py` 的 `StreamingPredictor` 只使用当前尾随窗，既有测试验证未来扰动及6秒之外信号扰动不影响当前输出。每条记录重置，不得删掉缺口后拼接。

加载权重前核对 `checkpoints/manifest.json`。不要加载来源不可信的pickle/joblib或任意PyTorch检查点。合成测试权重带有明确的synthetic标记，不能用来报告比赛成绩。
