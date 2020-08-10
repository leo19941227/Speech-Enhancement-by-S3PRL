import torch
import torch.nn as nn
import math
import numpy as np
import scipy
from scipy.signal.windows import hann as hanning
from torch import Tensor


class SISDR(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, predicted, linear_tar, stft_length_masks, **kwargs):
        # length_masks: (batch_size, max_time)
        # src, tar: (batch_size, max_time, feat_dim)
        src = predicted * stft_length_masks.unsqueeze(-1)
        tar = linear_tar * stft_length_masks.unsqueeze(-1)

        src = src.flatten(start_dim=1).contiguous()
        tar = tar.flatten(start_dim=1).contiguous()

        alpha = torch.sum(src * tar, dim=1) / (torch.sum(tar * tar, dim=1) + self.eps)
        ay = alpha.unsqueeze(1) * tar
        norm = torch.sum((ay - src) * (ay - src), dim=1) + self.eps
        loss = -10 * torch.log10(torch.sum(ay * ay, dim=1) / norm + self.eps)
        
        return loss.mean()


class L1(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps
        self.fn = torch.nn.L1Loss()

    def forward(self, predicted, linear_tar, stft_length_masks, **kwargs):
        # length_masks: (batch_size, max_time)
        # src, tar: (batch_size, max_time, feat_dim)

        src = predicted * stft_length_masks.unsqueeze(-1)
        tar = linear_tar * stft_length_masks.unsqueeze(-1)
        l1 = self.fn(src, tar)
        
        return l1
