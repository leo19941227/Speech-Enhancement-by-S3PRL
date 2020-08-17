import torch
import torch.nn as nn
import math
import numpy as np
import scipy
from scipy.signal.windows import hann as hanning
from torch import Tensor
from functools import partial
from utils import *
from asteroid.losses.sdr import SingleSrcNegSDR
from asteroid.losses.stoi import NegSTOILoss
from asteroid.losses.pmsqe import SingleSrcPMSQE


class stoi(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = NegSTOILoss(sample_rate = 16000)

    def forward(self, wav_predicted, wav_tar, length_masks, **kwargs):
        # stft_length_masks: (batch_size, max_time)
        # predicted, linear_tar: (batch_size, max_time, feat_dim)
       
        src = wav_predicted * length_masks
        tar = wav_tar * length_masks
        loss = self.fn(src, tar).mean()
        
        return loss, {}


class estoi(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = NegSTOILoss(sample_rate = 16000, extended=True)

    def forward(self, wav_predicted, wav_tar, length_masks, **kwargs):
        # stft_length_masks: (batch_size, max_time)
        # predicted, linear_tar: (batch_size, max_time, feat_dim)
       
        src = wav_predicted * length_masks
        tar = wav_tar * length_masks
        loss = self.fn(src, tar).mean()
        
        return loss, {}


class pmsqe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = SingleSrcPMSQE()
        self.fn.nbins = 400

    def forward(self, predicted, linear_tar, stft_length_masks, **kwargs):
        # stft_length_masks: (batch_size, max_time)
        # predicted, linear_tar: (batch_size, max_time, feat_dim)

        src = predicted * stft_length_masks.unsqueeze(-1)
        tar = linear_tar * stft_length_masks.unsqueeze(-1)
        loss = self.fn(src, tar)
        
        return loss, {}


class sisdr(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = SingleSrcNegSDR("sisdr", zero_mean=False, reduction='mean')

    def forward(self, predicted, linear_tar, stft_length_masks, **kwargs):
        # stft_length_masks: (batch_size, max_time)
        # predicted, linear_tar: (batch_size, max_time, feat_dim)

        src = predicted * stft_length_masks.unsqueeze(-1)
        tar = linear_tar * stft_length_masks.unsqueeze(-1)
        loss = self.fn(src, tar)
        
        return loss, {}


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
    def __init__(self, log=False, eps=1e-10):
        super().__init__()
        self.log = log
        self.eps = eps
        self.fn = torch.nn.L1Loss()

    def forward(self, predicted, linear_tar, stft_length_masks, **kwargs):
        # stft_length_masks: (batch_size, max_time)
        # predicted, linear_tar: (batch_size, max_time, feat_dim)

        src = predicted * stft_length_masks.unsqueeze(-1)
        tar = linear_tar * stft_length_masks.unsqueeze(-1)

        if self.log:
            l1 = self.fn(src.log(), tar.log())
        else:
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
        db_thres = 10 * torch.log10(energy.max() + self.eps) - self.db_interval
        voice_mask = ((10 * torch.log10(energy + self.eps)) > db_thres).long()

        speech_diff = (S - (G * S)) * voice_mask * stft_length_masks.unsqueeze(-1)
        speech_diff_powsum = speech_diff.pow(2).sum(-1).sum(-1)
        speech_loss =  speech_diff_powsum.mean()

        noise_loss = (G * N * stft_length_masks.unsqueeze(-1)).pow(2).sum(-1).sum(-1).mean()

        def logger_tmp(log, global_step, S, voice_mask, energy, **kwargs):
            fig = plot_spectrograms([
                (S[0] + self.eps).log(),
                (linear_inp[0] + self.eps).log(),
                (energy.expand_as(S)[0] + self.eps).log(),
                (S * voice_mask + self.eps)[0].log(),
                (N[0] + self.eps).log(),
            ])
            log.add_figure('WSD_variables', fig, global_step)
        logger = partial(logger_tmp, S=S, voice_mask=voice_mask, energy=energy)

        return self.alpha * speech_loss + (1 - self.alpha) * noise_loss, {'logger': logger}