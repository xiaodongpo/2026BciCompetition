"""决赛提交：滤波函数 filter_eeg 与预测函数 predict。

- filter_eeg：读取 BDF 的 EEG 通道（µV），做单向因果低通滤波（2 阶 Butterworth，100 Hz），
  输出与原始采样点一一对应。此处只做滤波，不做归一化、特征提取或缺口处理。
- predict：输入一个 6 秒窗口和十段 30 秒参考 EEG，输出 0–1 分数（越接近 1 越倾向任务状态）。
  模型参数固定；参考 EEG 只用于归一化（分位数 / z 分数）。除“参考统计量的缓存”（参考数组的
  纯函数）外不保留任何跨调用状态，结果与调用顺序无关。

源代码：fin/filtering.py（滤波）、fin/model.py（预测）、fin/features.py（特征）、
fin/quality.py（坏样本检测）、fin/net.py（网络结构）。模型文件在 model/ 目录。

本发布版仅保留团队预测核心，并使用独立编写的本地协议重放工具。
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import mne
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fin.filtering import causal_lowpass  # noqa: E402
from fin.model import FinalsModel  # noqa: E402

FS = 250
WINDOW = 6 * FS
REFERENCE = 30 * FS
REFERENCE_COUNT = 10
HORIZON = 10 * 60 * FS

_MODEL = None


def _model() -> FinalsModel:
    """模型只加载一次；参数在推理期间不变。"""
    global _MODEL
    if _MODEL is None:
        _MODEL = FinalsModel(os.environ.get("FIN_MODEL_DIR", str(_HERE / "model")))
    return _MODEL


# 以下两个函数由参赛队伍提供。
def filter_eeg(bdf_path: str) -> np.ndarray:
    """读取 BDF 并返回等长一维因果滤波数组，单位为微伏（µV）。"""
    raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose="ERROR")
    try:
        if "EEG" not in raw.ch_names or not np.isclose(raw.info["sfreq"], FS, rtol=0, atol=1e-9):
            raise ValueError("需要名为 EEG 的通道，采样率为 250 Hz")
        eeg = raw.get_data(picks=["EEG"], units="uV")[0]
    finally:
        raw.close()
    # 2 阶 Butterworth 低通 100 Hz，单向因果滤波；初始状态取首个采样点的稳态值。
    return causal_lowpass(eeg)


def predict(window: np.ndarray, reference: np.ndarray) -> float:
    """输入 (1500,) 目标窗口及 (10, 7500) 参考 EEG，返回 0–1 分数（越大越倾向任务状态）。"""
    return _model().predict(window, reference)


# Local protocol utilities are independently implemented, not copied organiser files.
from protocol import segment_and_shuffle

def checked_predict(window, reference):
    if np.shape(window)!=(WINDOW,) or np.shape(reference)!=(REFERENCE_COUNT,REFERENCE):
        raise ValueError('Expected window (1500,) and reference (10,7500)')
    value=float(predict(np.array(window,copy=True),np.array(reference,copy=True)))
    if not np.isfinite(value) or not 0<=value<=1:
        raise ValueError('Model score must be a finite probability')
    return value

def check_order_independence(windows,reference):
    sample=windows[:5]
    forward=[checked_predict(w,reference) for w in sample]
    backward=[checked_predict(w,reference) for w in sample[::-1]][::-1]
    np.testing.assert_allclose(forward,backward,atol=1e-6,rtol=1e-6)

def main():
    parser=argparse.ArgumentParser(description='Local H1 protocol replay; no evaluation labels read.')
    parser.add_argument('bdf_path');parser.add_argument('output_csv');parser.add_argument('--seed',type=int,default=14);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args();output=Path(args.output_csv)
    if output.exists():raise FileExistsError('Choose a fresh output file')
    ref,wins=segment_and_shuffle(filter_eeg(args.bdf_path),args.seed)
    scores=[checked_predict(w,ref) for w in wins]
    if args.self_test:check_order_independence(wins,ref)
    with output.open('x',encoding='utf8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['shuffled_index','score']);writer.writerows(enumerate(scores))
    print(f'PREDICTION_PASS windows={len(scores)}')

if __name__=='__main__':main()
