import torch
import torch.nn as nn
import math
import numpy as np
import scipy
from scipy.signal.windows import hann as hanning
from torch import Tensor
from functools import partial
from utils import *


class SISDR(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, predicted, linear_tar, stft_length_masks, **kwargs):
        # stft_length_masks: (batch_size, max_time)
        # predicted, linear_tar: (batch_size, max_time, feat_dim)
        src = predicted * stft_length_masks.unsqueeze(-1)
        tar = linear_tar * stft_length_masks.unsqueeze(-1)

        src = src.flatten(start_dim=1).contiguous()
        tar = tar.flatten(start_dim=1).contiguous()

        alpha = torch.sum(src * tar, dim=1) / (torch.sum(tar * tar, dim=1) + self.eps)
        ay = alpha.unsqueeze(1) * tar
        norm = torch.sum((ay - src) * (ay - src), dim=1) + self.eps
        loss = -10 * torch.log10(torch.sum(ay * ay, dim=1) / norm + self.eps)
        
        return loss.mean(), {}


class L1(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps
        self.fn = torch.nn.L1Loss()

    def forward(self, predicted, linear_tar, stft_length_masks, **kwargs):
        # stft_length_masks: (batch_size, max_time)
        # predicted, linear_tar: (batch_size, max_time, feat_dim)

        src = predicted * stft_length_masks.unsqueeze(-1)
        tar = linear_tar * stft_length_masks.unsqueeze(-1)
        l1 = self.fn(src, tar)
        
        return l1, {}


class WSD(nn.Module):
    def __init__(self, alpha=0.5, db_interval=30, eps=1e-10):
        super().__init__()
        self.db_interval = db_interval
        self.alpha = alpha
        self.eps = eps

    def forward(self, linear_inp, offset, linear_tar, stft_length_masks, **kwargs):
        device = linear_inp.device
        S, G = linear_tar, offset
        N = torch.max(linear_inp - linear_tar, torch.zeros(1).to(device))
        
        energy = S.sum(dim=-1, keepdim=True)
        db_thres = 10 * torch.log10(energy.max()) - self.db_interval
        voice_mask = ((10 * torch.log10(energy)) > db_thres).long()

        speech_diff = (S - (G * S)) * voice_mask * stft_length_masks.unsqueeze(-1)
        speech_diff_powsum = speech_diff.pow(2).sum(-1).sum(-1)
        speech_loss =  speech_diff_powsum.mean()

        noise_loss = (G * N * stft_length_masks.unsqueeze(-1)).pow(2).sum(-1).sum(-1).mean()

        def logger_tmp(log, global_step, S, voice_mask, **kwargs):
            fig = plot_spectrogram(S[0].log())
            log.add_figure('WSD_speech', fig, global_step)
            fig = plot_spectrogram((S * voice_mask)[0].log())
            log.add_figure('WSD_voice_mask', fig, global_step)
        logger = partial(logger_tmp, **locals())

        return self.alpha * speech_loss + (1 - self.alpha) * noise_loss, {'logger': logger}