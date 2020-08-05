import numpy as np
from pesq import pesq as pesq1
from pypesq import pesq as pesq2
from pystoi import stoi

def pesq_nb_eval(src, tar, sr=16000):
    assert src.ndim == 1 and tar.ndim == 1
    assert not np.allclose(src.sum(), 0.0, atol=1e-6) and not np.allclose(tar.sum(), 0.0, atol=1e-6)
    mos_lqo = pesq1(sr, tar, src, 'nb')
    return mos_lqo

def pesq_wb_eval(src, tar, sr=16000):
    assert src.ndim == 1 and tar.ndim == 1
    assert not np.allclose(src.sum(), 0.0, atol=1e-6) and not np.allclose(tar.sum(), 0.0, atol=1e-6)
    mos_lqo = pesq1(sr, tar, src, 'wb')
    return mos_lqo

def pypesq_eval(src, tar, sr=16000):
    assert src.ndim == 1 and tar.ndim == 1
    assert not np.allclose(src.sum(), 0.0, atol=1e-6) and not np.allclose(tar.sum(), 0.0, atol=1e-6)
    raw_pesq = pesq2(tar, src, sr)
    mos_lqo = 0.999 + 4.0 / (1.0 + np.exp(-1.4945 * raw_pesq + 4.6607))
    return mos_lqo

def stoi_eval(src, tar, sr=16000):
    assert src.ndim == 1 and tar.ndim == 1
    return stoi(tar, src, sr, extended=False)

def estoi_eval(src, tar, sr=16000):
    assert src.ndim == 1 and tar.ndim == 1
    return stoi(tar, src, sr, extended=True)