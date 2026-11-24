"""仅用于接口复现的确定性合成原始EEG，不带状态标签，不用于宣称分类性能。"""
from pathlib import Path
import numpy as np

if __name__=='__main__':
    p=Path(__file__).resolve().parent/'example';p.mkdir(exist_ok=True)
    rng=np.random.default_rng(20261107);t=np.arange(35*250)/250
    x=40000+20*np.sin(2*np.pi*10*t)+(5+8*np.sin(t/10)**2)*np.sin(2*np.pi*6*t)+120*np.sin(2*np.pi*50*t)+rng.normal(0,3,len(t))
    np.save(p/'raw_eeg_uv.npy',x.astype(np.float64))
    print('EXAMPLE_CREATED shape=',x.shape,'fs=250 units=uV seconds=35 synthetic=True')
