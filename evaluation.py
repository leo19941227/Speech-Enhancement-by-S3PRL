import numpy as np
from pesq import pesq
from pystoi import stoi

def pesq_nb_eval(src, tar, sr=16000):
    assert src.ndim == 1 and tar.ndim == 1
    if np.allclose(src.sum(), 0.0, atol=1e-10) or np.allclose(tar.sum(), 0.0, atol=1e-10):
        print(f'[Evaluation] wav values too small: src {src.sum()}, tar {tar.sum()}')
    mos_lqo = pesq(sr, tar, src, 'nb')
    return mos_lqo

def pesq_wb_eval(src, tar, sr=16000):
    assert src.ndim == 1 and tar.ndim == 1
    if np.allclose(src.sum(), 0.0, atol=1e-10) or np.allclose(tar.sum(), 0.0, atol=1e-10):
        print(f'[Evaluation] wav values too small: src {src.sum()}, tar {tar.sum()}')
    mos_lqo = pesq(sr, tar, src, 'wb')
    return mos_lqo

def stoi_eval(src, tar, sr=16000):
    assert src.ndim == 1 and tar.ndim == 1
    return stoi(tar, src, sr, extended=False)

def estoi_eval(src, tar, sr=16000):
    assert src.ndim == 1 and tar.ndim == 1
    return stoi(tar, src, sr, extended=True)