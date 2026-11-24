from pathlib import Path
import numpy as np
from r0 import R0Decoder

root=Path(__file__).resolve().parent
decoder=R0Decoder(root/'model',device='cpu')
decoder.reset()  # 每条独立记录开始时调用，阶段切换时不调用。
eeg=np.load(root/'example/raw_eeg_uv.npy')
for start in range(0,len(eeg),137):
    for result in decoder.push(eeg[start:start+137]):
        print(result['time_sec'],result['state'],result['score'],result['valid'])
