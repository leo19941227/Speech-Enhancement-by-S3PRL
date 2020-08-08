import torch

def masked_mean(batch, length_masks, keepdim=False):
    means = (batch * length_masks).sum(dim=-1, keepdim=True) / (length_masks.sum(dim=-1, keepdim=True) + self.eps)
    if not keepdim:
        means = means.squeeze(-1)
    return means

def masked_normalize_decibel(audio, target, length_masks):
    # audio: (batch_size, max_time)
    # length_masks: (batch_size, max_time)

    if type(target) is torch.Tensor:
        target = 10.0 * torch.log10(masked_mean(reference_audio.pow(2), length_masks, keepdim=True))
    assert type(target) is float

    scalar = (10.0 ** (target / 10.0)) / (masked_mean(audio.pow(2), length_masks, keepdim=True) + self.eps)        
    return audio * scalar