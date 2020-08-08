import torch
import matplotlib.pyplot as plt

def masked_mean(batch, length_masks, keepdim=False, eps=1e-8):
    # batch: (batch_size, max_time)
    means = (batch * length_masks).sum(dim=-1, keepdim=keepdim) / (length_masks.sum(dim=-1, keepdim=keepdim) + eps)
    if not keepdim:
        means = means.squeeze(-1)
    return means

def masked_normalize_decibel(audio, target, length_masks, eps=1e-8):
    # audio: (batch_size, max_time)
    # length_masks: (batch_size, max_time)

    if type(target) is float:
        # target: fixed decibel level
        target = torch.ones(len(audio)).to(device=audio.device) * target
    elif type(target) is torch.Tensor and target.dim() > 1:
        # target: reference audio for decibel level
        target = 10.0 * torch.log10(masked_mean(target.pow(2), length_masks, keepdim=False))
    assert type(target) is torch.Tensor and target.dim() == 1
    # target: (batch_size, ), each utterance has a target decibel level

    scalar_square = (10.0 ** (target.unsqueeze(-1) / 10.0)) / (masked_mean(audio.pow(2), length_masks, keepdim=True) + eps)        
    scalar = scalar_square.pow(0.5)
    return audio * scalar

def plot_spectrogram(spec, height=2):
    h, w = spec.size(0), spec.size(1)
    scaling = height / h
    fig = plt.figure(figsize=(round(w * scaling), round(h * scaling)))
    plt.imshow(spec)
    return fig