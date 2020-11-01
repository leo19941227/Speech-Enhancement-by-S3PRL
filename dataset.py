import os
import re
import glob
import copy
import random

import torch
import librosa
import torchaudio
from librosa.util import find_files
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

ACTIVE_BUFFER_NUM = 4


class PseudoDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(1000, 16000, 2)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
        # return: (time, 2)


def filestrs2list(filestrs, fileroot=None, sample_num=0, select_sampled=False, **kwargs):
    print(f'[filestrs2list] - Parsing filestrs: {filestrs}.')

    if type(filestrs) is not list:
        filestrs = [filestrs]

    all_files = []
    for filestr in filestrs:
        if os.path.isdir(filestr):
            all_files += sorted(find_files(filestr))
        elif os.path.isfile(filestr):
            with open(filestr, 'r') as handle:
                all_files += sorted([f'{fileroot}/{line[:-1]}' for line in handle.readlines()])
        else:
            all_files += sorted(glob.glob(filestr))

    all_files = sorted(all_files)
    random.seed(0)
    random.shuffle(all_files)
    all_files = all_files[:sample_num] if select_sampled else all_files[sample_num:]

    print(f'[filestrs2list] - Complete parsing: {len(all_files)} files found.')
    return all_files


def add_noise(speech, noise, snrs, eps=1e-10):
    # speech, noise: (batch_size, seqlen)
    # snrs: (batch_size, )
    if speech.size(-1) >= noise.size(-1):
        times = speech.size(-1) // noise.size(-1)
        remainder = speech.size(-1) % noise.size(-1)
        noise_expanded = noise.unsqueeze(-2).expand(-1, times, -1).reshape(speech.size(0), -1)
        noise = torch.cat([noise_expanded, noise[:, :remainder]], dim=-1)
    else:
        noise = noise[:, :speech.size(-1)]
    assert noise.size(-1) == speech.size(-1)

    snr_exp = 10.0 ** (snrs / 10.0)
    speech_power = speech.pow(2).sum(dim=-1, keepdim=True)
    noise_power = noise.pow(2).sum(dim=-1, keepdim=True)
    scalar = (speech_power / (snr_exp * noise_power + eps)).pow(0.5)
    scaled_noise = scalar * noise
    noisy = speech + scaled_noise

    assert torch.isnan(noisy).sum() == 0 and torch.isinf(noisy).sum() == 0 
    return noisy, scaled_noise


class OnlineDataset(Dataset):
    def __init__(self, speech, noise, sample_rate, max_time, min_time=0,
                 target_level=-25, snrs=[3], infinite=False, half_noise=None,
                 pseudo_modes=None, pseudo_clean=None, pseudo_noise=None,
                 seed=0, eps=1e-8, **kwargs):
        self.sample_rate = sample_rate
        self.max_time = max_time
        self.min_time = min_time
        self.target_level = target_level
        self.infinite = infinite
        self.half_noise = half_noise
        self.pseudo_modes = pseudo_modes
        self.pseudo_clean = pseudo_clean
        self.pseudo_noise = pseudo_noise
        self.eps = eps

        self.filepths = filestrs2list(**speech)
        self.all_noises = filestrs2list(**noise)
        self.all_snrs = snrs

        random.seed(0)
        self.fixed_noises = random.choices(self.all_noises, k=len(self.filepths))

        random.seed(0)
        self.fixed_snrs = random.choices(self.all_snrs, k=len(self.filepths))

        # This mapping directly decide how many data points are in the dataset
        self.id_mapping = list(range(len(self.filepths)))

    def normalize_wav_decibel(self, audio):
        '''Normalize the signal to the target level'''
        rms = audio.pow(2).mean().pow(0.5)
        scalar = (10 ** (self.target_level / 20)) / (rms + 1e-10)
        audio = audio * scalar
        return audio

    def load_data(self, wav_path):
        wav, sr = librosa.load(wav_path, sr=self.sample_rate)
        wav = torch.FloatTensor(wav)
        
        maxpoints = int(sr / 1000) * self.max_time
        minpoints = int(sr / 1000) * self.min_time
        if len(wav) < minpoints:
            times = minpoints // len(wav) + 1
            wav = wav.unsqueeze(0).expand(times, -1).reshape(-1)
        if len(wav) > maxpoints:
            wav = wav[:maxpoints]

        return wav

    def __getitem__(self, idx):
        idx = self.id_mapping[idx]
        if self.pseudo_modes is not None:
            case = random.choice(self.pseudo_modes)

        # speech
        speech_pth = self.filepths[idx]
        if 'case' in locals() and (case == 2 or case == 3) and self.pseudo_clean is not None:
            speech = random.choice(self.pseudo_clean)
        else:
            speech = self.load_data(speech_pth)
        speech = self.normalize_wav_decibel(speech)

        # noise
        noise_pth = random.choice(self.all_noises) if self.infinite else self.fixed_noises[idx]
        if 'case' in locals() and (case == 0 or case == 3) and self.pseudo_noise is not None:
            noise = random.choice(self.pseudo_noise)
        else:
            noise = self.load_data(noise_pth)
        
        if self.half_noise:
            middle = len(noise) // 2
            if self.half_noise == 'front':
                noise = noise[:middle]
            elif self.half_noise == 'end':
                noise = noise[middle:]

        noise = self.normalize_wav_decibel(noise)

        # noisy
        snr = random.choice(self.all_snrs) if self.infinite else self.fixed_snrs[idx]
        noisy, scaled_noise = add_noise(speech.unsqueeze(0), noise.unsqueeze(0), torch.ones(1) * snr, self.eps)
        noisy, scaled_noise = noisy.squeeze(0), scaled_noise.squeeze(0)

        wavs = torch.stack([noisy, speech, scaled_noise], dim=-1)
        if 'case' in locals():
            return wavs, case
        return wavs

    def __len__(self):
        return len(self.id_mapping)

    def collate_fn(self, samples):
        if type(samples[0]) is torch.Tensor:
            wavs = samples
        else:
            wavs, cases = [[samples[i][j] for i in range(len(samples))] for j in range(len(samples[0]))]

        lengths = torch.LongTensor([len(s) for s in wavs])
        wavs = pad_sequence(wavs, batch_first=True).transpose(-1, -2).contiguous()
        if 'cases' not in locals():
            return lengths, wavs
        return lengths, wavs, torch.LongTensor(cases)

    def get_subset(self, n_file=100):
        subset = copy.deepcopy(self)
        subset.infinite = False
        random.seed(0)
        random.shuffle(subset.id_mapping)
        subset.id_mapping = subset.id_mapping[:n_file]
        return subset


class NoisyCleanDataset(Dataset):
    # This dataset identify the clean/noisy pair by the regex pattern in the filename
    # Each directory in roots should contain two sub-directories: clean & noisy
    # eg. The noisy file 'root/noisy/fileid_0.wav' has the label of the clean file 'root/clean/fileid_0.wav'
    def __init__(self, roots, noisy_channel=0, clean_channel=1, seed=1227, sample_ratio=1.0,
                 select_sampled=True, sample_num=None, regex='fileid_\d+', max_sec=10.0):
        random.seed(seed)

        clean_pths = []
        for root in roots:
            clean_pths.extend(find_files(os.path.join(root, 'clean')))
        clean_pths = sorted(clean_pths)

        sampled = random.sample(clean_pths, round(len(clean_pths) * sample_ratio))
        if select_sampled:
            self.clean_pths = sampled
        else:
            self.clean_pths = [pth for pth in clean_pths if pth not in sampled]
        assert len(self.clean_pths) > 0

        if sample_num is not None:
            if len(self.clean_pths) >= sample_num:
                self.clean_pths = self.clean_pths[:sample_num]
            else:
                times = sample_num // len(self.clean_pths) + 1
                self.clean_pths = (self.clean_pths * times)[:sample_num]

        self.noisy_channel = noisy_channel
        self.clean_channel = clean_channel
        self.regex_searcher = re.compile(regex)
        self.max_sec = max_sec

    def __getitem__(self, idx):
        clean_pth = self.clean_pths[idx]
        result = self.regex_searcher.search(clean_pth)
        assert result is not None
        fileid = result.group()
        noisy_dir = os.path.dirname(clean_pth).replace('clean', 'noisy')
        noisy_pths = glob.glob(f'{noisy_dir}/*{fileid}*')
        file_searcher = re.compile(fileid + '\D')
        noisy_pths = [pth for pth in noisy_pths if file_searcher.search(pth) is not None]
        assert len(noisy_pths) == 1, f'{noisy_pths}'
        noisy_pth = noisy_pths[0]
       
        clean, sr1 = torchaudio.load(clean_pth)
        noisy, sr2 = torchaudio.load(noisy_pth)
        assert sr1 == sr2
        assert clean.size(-1) == noisy.size(-1)

        max_length = round(self.max_sec * sr1)
        if clean.size(-1) > max_length:
            start = random.randint(0, clean.size(-1) - max_length - 1)
            clean = clean[:, start:start + max_length]
            noisy = noisy[:, start:start + max_length]
        
        return torch.stack([noisy, clean], dim=-1).view(-1, 2)
        # return: (time, 2)

    def __len__(self):
        return len(self.clean_pths)

    def get_subset(self, ratio=0.2, sample_seed=None):
        subset = copy.deepcopy(self)
        clean_pths = sorted(subset.clean_pths)
        subset_num = round(len(clean_pths) * ratio)
        if sample_seed is None:
            clean_pths = clean_pths[:subset_num]
        else:
            random.seed(sample_seed)
            clean_pths = random.sample(clean_pths, subset_num)
        subset.clean_pths = clean_pths
        return subset
