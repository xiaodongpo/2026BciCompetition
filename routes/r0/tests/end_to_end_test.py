"""用随包001原始记录做CPU全链路验收，不是57记录正式训练。

处理原始BDF/事件、独立复算、建流与特征、随机初始化三种子各1轮、树2轮、
计算源原型、导出新模型并加载推理。只在独立workspace目录写入测试结果。
"""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
sys.dont_write_bytecode = True
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir', type=Path, default=PACKAGE / 'workspace/end_to_end_check')
    ap.add_argument('--raw-data', type=Path, default=PACKAGE / 'data/EEG_dataset')
    args = ap.parse_args()
    work = args.work_dir.resolve()
    raw = args.raw_data.resolve()
    if work.exists() and any(work.iterdir()):
        raise ValueError('验收目录必须为空；请用--work-dir指定一个新的目录')
    if work == PACKAGE or work == raw or raw in work.parents or work in raw.parents:
        raise ValueError('验收目录不能覆盖代码包或原始数据')
    if PACKAGE in work.parents and PACKAGE / 'workspace' not in work.parents:
        raise ValueError('包内验收输出只能放在workspace的子目录')
    from train_from_scratch import materialize
    materialize(work, raw, 'cpu', 1, epochs=1, tree_rounds=2)
    sys.path[:0] = [str(work / 'data_processing_current'), str(work / 'online_adapt'), str(work)]
    import numpy as np
    import preprocess as prep
    import common
    assert (raw / '001').is_dir(), '缺少随包001原始记录'
    prep.AUDIT.mkdir(); prep.STAGING.mkdir()
    entry, _, windows = prep.process(raw / '001')
    entry['split'] = 'train'
    common.table(prep.STAGING / 'source_windows_paired.csv', windows)
    common.dump(prep.STAGING / 'split.json', dict(train_records=['001'], validation_records=[]))
    common.dump(prep.STAGING / 'manifest.json', dict(
        record_count=1, raw_source=str(raw), records=[entry], windows=entry['windows'],
        class_counts=dict(zip(['relax', 'task'], entry['class_counts'])),
        split_windows=dict(train=entry['windows'], validation=0)))

    # 仅将独立核验main中的记录数约束57/45/12调整为1/1/0。
    # 时间、标签、逐点质量、滤波及逐元素比较主体均保持不变，正式verify.py不改动。
    verify_path = work / 'data_processing_current/verify.py'
    tree = ast.parse(verify_path.read_text(encoding='utf-8'))
    class OneRecordScope(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name != 'main':
                return node
            class Counts(ast.NodeTransformer):
                def visit_Constant(self, value):
                    if type(value.value) is int and value.value in {57, 45, 12}:
                        return ast.copy_location(ast.Constant({57: 1, 45: 1, 12: 0}[value.value]), value)
                    return value
            return Counts().visit(node)
    scope = dict(__name__='one_record_verifier', __file__=str(verify_path))
    exec(compile(ast.fix_missing_locations(OneRecordScope().visit(tree)), str(verify_path), 'exec'), scope)
    scope['main']()
    prep.STAGING.rename(prep.FINAL)

    import stream_energy_data as stream
    stream.CACHE.parent.mkdir(parents=True, exist_ok=True)
    stream.prepare()
    from online_adapt import gen_stream_features as gen
    _, stream_windows = gen.work('001')
    import train_r0_all57 as trainer
    # 单记录验收只放宽训练ID数量；正式入口ids()仍严格使用全部57记录。
    trainer.ids = lambda: ['001']
    trainer.ROOT.mkdir(parents=True, exist_ok=True)
    for seed in trainer.SEEDS:
        trainer.cnn(seed)
    trainer.tree_and_prior()
    model_dir = trainer.ROOT / 'model'
    config = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    assert config['epochs'] == 1 and config['tree_rounds'] == 2
    assert config['training_records'] == ['001'] and config['training_records_count'] == 1
    config['acceptance_test_only'] = True
    (model_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    from r0 import R0Decoder
    import r0.decoder as deployed_module
    from run_inference import load_signal
    bdf = next((raw / '001').glob('*.bdf'))
    signal, mask = load_signal(bdf, 'EEG', 250)
    decoder = R0Decoder(model_dir, device='cpu')
    results = []
    for start in range(0, min(len(signal), 8750), 137):
        end = min(start + 137, 8750)
        results.extend(decoder.push(signal[start:end], mask[start:end]))
    assert len(results) == 32 and all(np.isfinite(r['score']) for r in results)
    origins = {name: str(Path(module.__file__).resolve()) for name, module in
               [('preprocess', prep), ('common', common), ('stream', stream),
                ('features', gen), ('trainer', trainer), ('decoder', deployed_module)]}
    assert all(PACKAGE in Path(path).parents or work in Path(path).parents for path in origins.values())
    (work / 'predictions.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    report = dict(status='END_TO_END_PASS', raw_record='001',
                  clean_windows=entry['windows'], class_counts=entry['class_counts'],
                  independent_quality_labels_and_arrays=True, stream_windows=stream_windows,
                  cnn_seeds=trainer.SEEDS, epochs_each=1, tree_rounds=2,
                  model_files=sorted(x.name for x in model_dir.iterdir()),
                  new_model_reload_and_inference=True, inference_decisions=len(results),
                  model_directory=str(model_dir), formal57_training_performed=False)
    report['algorithm_module_origins'] = origins
    report['all_algorithm_modules_inside_package_or_materialized_workdir'] = True
    (work / 'END_TO_END_PASS.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('END_TO_END_PASS', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
