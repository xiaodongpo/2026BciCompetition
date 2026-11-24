# 安装与运行平台

使用Python 3.11。直接依赖来自代码导入与实际运行，不是全局环境的整份导出。`requirements.txt`固定核心版本，`requirements-dev.txt`加入pytest，可选迁移实验使用`requirements-transfer.txt`。

## Windows PowerShell

在仓库根目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pip check
python scripts/check_release.py
python -m pytest -q
python scripts/h1_synthetic_smoke.py --out cache/h1_demo
```

无需激活也可把`python`替换成`.\.venv\Scripts\python.exe`。若系统限制激活脚本，直接使用解释器路径，无需修改系统执行策略。脚本应从仓库根目录启动；不要把多个路线的`src`同时加入PYTHONPATH。

## Linux（命令及CI已配置，尚未实机验证）

```bash
python3.11 -m venv .venv
source .venv/bin/activate
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
```

## GPU与可选历史模型

H1x5完整网络训练可选择`--device cuda`。与CUDA 12.8及相应驱动兼容的机器，可将上面的PyTorch安装行替换为：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

安装后用`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`核对。无可用GPU时使用`--device cpu`，不要假定安装CUDA wheel就具备GPU。本次没有验证新环境的GPU训练。

BIOT/SingLEM仅属于历史路线，另外执行`python -m pip install -r requirements-transfer.txt`。权重许可独立核验，见[第三方说明](../THIRD_PARTY_NOTICES.md)。本次独立环境没有安装或重跑这一可选扩展。

## 本次实际安装证据与限制

2026-11-23在Windows新建了独立venv，未继承原环境site-packages。在线下载PyTorch CPU wheel及部分依赖停滞，因此改用本机既有的官方PyTorch 2.7.1+cu128 wheel和pip缓存中的兼容wheel，通过`pip --no-index --find-links`安装同一依赖清单。不是复制原虚拟环境。`pip check`通过，14项CPU测试通过（76.05秒）。

完整安装快照见[独立环境锁定文件](../requirements-validation-windows.txt)和[验证摘要](../results/validation/environment.json)。该快照只记录新建验收环境，不能当作Linux或所有GPU平台的锁文件。全新在线下载尚未完成验证；GitHub Actions尚未运行。缓存wheel及环境本身不随仓库发布。

常见问题：缺少模型文件时先运行合成示例或自行训练；缓存与数据版本不一致时换用新的工作目录；无效路径通过`--data-root`解决；LightGBM依赖缺失应检查平台运行库，不应直接忽略导入错误。
