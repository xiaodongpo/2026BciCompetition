"""LAG-EnergyProto完整训练入口。

作用：将随包源码复制到独立工作目录，统一写入路径、设备及轮次配置，
顺序执行新57记录预处理、独立验证、连续流缓存、特征、从零CNN/树/原型拟合。
不加载随包最终权重进行训练；断点恢复只读取本次工作目录的检查点。
"""
import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def materialize(work, raw, device, workers, epochs=50, tree_rounds=200):
    """复制运行源码并保存配置，不通过替换源码字符串修改路径。"""
    marker = work / 'materialized_source_manifest.json'
    if work.exists() and any(work.iterdir()) and not marker.exists():
        raise ValueError('请使用空工作目录或本入口先前生成的目录，避免覆盖已有工程')
    work.mkdir(parents=True, exist_ok=True)
    settings = dict(raw_data=str(raw), epochs=epochs, tree_rounds=tree_rounds,
                    device=device, workers=workers)
    settings_file = work / 'pipeline_config.json'
    if settings_file.exists():
        previous = json.loads(settings_file.read_text(encoding='utf-8'))
        if any(previous.get(key) != settings[key] for key in ['raw_data', 'epochs', 'tree_rounds']):
            raise ValueError('工作目录的数据来源或训练轮次与本次不同，请使用新的空工作目录')
    manifest = {}
    for src in (PACKAGE / 'training/source').rglob('*.py'):
        rel = src.relative_to(PACKAGE / 'training/source')
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        manifest[rel.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    (work / 'materialized_source_manifest.json').write_text(json.dumps(manifest, indent=2))
    settings_file.write_text(json.dumps(settings, indent=2), encoding='utf-8')


def run(work, relative):
    """独立进程执行各阶段，避免同名模块和全局路径串用。"""
    env = dict(os.environ)
    paths = [str(work), str(work / 'online_adapt')]
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    env['PYTHONUTF8'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run([sys.executable, str(work / relative)], cwd=str(work), env=env, check=True)


def prepare(work):
    """新数据已是北京时间原点，沿用已核验的文件名/T0策略，不再人为平移8小时。"""
    final = work / 'prepared_1_50hz'
    staging = work / 'prepared_1_50hz_new_staging'
    if not final.exists():
        if not staging.exists():
            run(work, 'data_processing_current/preprocess.py')
        run(work, 'data_processing_current/verify.py')
        staging.rename(final)
    assert json.loads((final / 'verification.json').read_text())['status'] == 'PASS'
    cache = work / 'runs/energy_stream_training_50epochs/stream_cache'
    if not (cache / 'metadata.json').exists():
        if cache.exists():
            raise RuntimeError('存在未完成流缓存，请使用新工作目录；程序不自动删除数据')
        cache.parent.mkdir(parents=True, exist_ok=True)
        run(work, 'stream_energy_data.py')
    run(work, 'online_adapt/gen_stream_features.py')
    print('PREPARATION_SUCCESS', flush=True)


def smoke(work):
    """只验证训练接口、梯度、LAG边界及训练/推理同构，不是正式训练。"""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    sys.path[:0] = [str(work), str(work / 'online_adapt'), str(PACKAGE)]
    from model_representation import EnergyFusionCNN as TrainingModel
    from proto_student_teacher import lag_targets
    from r0.model import EnergyFusionCNN as InferenceModel
    from strict6s_pipeline import extract_batch as original_features
    from r0.signal import extract_batch as packaged_features
    elapsed = np.tile([0., 14., 15., 44., 45., 60.], 2)
    q = dict(elapsed=elapsed, y=np.repeat([0, 1], 6), starts=np.arange(12), eval=np.arange(12))
    target, keep, _ = lag_targets(q)
    assert keep.tolist() == [False, False, False, False, True, True, True, True, False, False, True, True]
    assert target[6:8].tolist() == [0, 0]
    torch.manual_seed(42)
    model = TrainingModel()
    x = torch.randn(6, 1, 1000)
    packed = torch.cat([x[:, 0], torch.zeros(6, 6)], 1)
    model.eval()
    deployed = InferenceModel().eval()
    deployed.load_state_dict(model.state_dict(), strict=True)
    assert torch.allclose(model(packed), deployed(x), atol=1e-7)
    before = model.energy_filters.weight.detach().clone()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    loss = torch.nn.functional.cross_entropy(model(packed), torch.tensor([0, 1, 0, 1, 0, 1]))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    opt.step()
    assert torch.isfinite(loss) and not torch.equal(before, model.energy_filters.weight)
    rng = np.random.default_rng(42)
    raw = 10000 + rng.normal(size=(2, 1500)) * 20
    a, _ = original_features(raw, np.ones(2, bool))
    b, _ = packaged_features(raw, np.ones(2, bool))
    assert np.array_equal(a, b)
    # 确认原始训练入口及其所有顶层导入可以在独立目录解析。
    importlib.import_module('train_r0_all57')
    report = dict(status='TRAINING_SMOKE_PASS', finite_loss=float(loss.detach()),
                  lag_boundaries=True, training_inference_state_dict_and_logits=True,
                  one_optimizer_step=True, feature_exact_equality=True,
                  full_50_epoch_retraining_performed=False)
    (work / 'training_smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['prepare', 'train', 'all', 'smoke'], required=True)
    ap.add_argument('--raw-data', type=Path, default=PACKAGE / 'data/EEG_dataset')
    ap.add_argument('--work-dir', type=Path, default=PACKAGE / 'workspace/full_training')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=50, help='每个CNN种子的训练轮次；正式复现为50')
    ap.add_argument('--tree-rounds', type=int, default=200, help='树模型轮次；正式复现为200')
    args = ap.parse_args()
    work, raw = args.work_dir.resolve(), args.raw_data.resolve()
    workspace = PACKAGE / 'workspace'
    if work == PACKAGE or (PACKAGE in work.parents and work != workspace and workspace not in work.parents):
        raise ValueError('包内工作目录必须放在workspace下，不能覆盖代码、原始数据或model')
    if work == raw or raw in work.parents or work in raw.parents:
        raise ValueError('工作目录与原始数据目录须互相独立')
    if args.workers < 1:
        raise ValueError('workers必须为正整数')
    if args.epochs < 1 or args.tree_rounds < 1:
        raise ValueError('训练轮次必须为正整数')
    if args.stage != 'smoke' and not raw.is_dir():
        raise FileNotFoundError('找不到原始数据目录：%s' % raw)
    materialize(work, raw, args.device, args.workers, args.epochs, args.tree_rounds)
    if args.stage == 'smoke':
        smoke(work)
        return
    if args.stage in ['prepare', 'all']:
        prepare(work)
    if args.stage in ['train', 'all']:
        run(work, 'train_r0_all57.py')
        output = work / 'runs/r0_all57_submission_20261107/model'
        (output / 'TRAINING_FINISHED.txt').write_text(
            'R0: three seeds x %d epochs, tree %d rounds; fresh weights unless resuming this work directory.\n'
            % (args.epochs, args.tree_rounds))
        print('TRAINING_SUCCESS model_dir=' + str(output), flush=True)


if __name__ == '__main__':
    main()
