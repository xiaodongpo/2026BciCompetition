"""训练流程的统一路径与运行参数，不依赖某台服务器的绝对路径。

工作副本从同目录pipeline_config.json读取入口保存的配置；直接导入随包源码时，
默认数据与输出均定位到本工程内部。这里只读取配置，不创建或删除目录。
"""
import json
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SOURCE_DIR / 'pipeline_config.json'

if CONFIG_FILE.is_file():
    SETTINGS = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
    WORK_DIR = SOURCE_DIR
    RAW_DATA = Path(SETTINGS['raw_data']).resolve()
else:
    # 随包布局：工程根目录/training/source/pipeline_paths.py。
    PACKAGE_DIR = SOURCE_DIR.parents[1]
    if not (PACKAGE_DIR / 'train_from_scratch.py').is_file():
        raise RuntimeError('缺少pipeline_config.json，请通过工程根目录的训练入口运行')
    WORK_DIR = PACKAGE_DIR / 'workspace/full_training'
    RAW_DATA = PACKAGE_DIR / 'data/EEG_dataset'
    SETTINGS = {}

DEVICE = SETTINGS.get('device', 'cuda')
WORKERS = int(SETTINGS.get('workers', 2))
EPOCHS = int(SETTINGS.get('epochs', 50))
TREE_ROUNDS = int(SETTINGS.get('tree_rounds', 200))

if DEVICE not in ('cpu', 'cuda') or min(WORKERS, EPOCHS, TREE_ROUNDS) < 1:
    raise ValueError('pipeline_config.json中的设备或运行参数无效')
