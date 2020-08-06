import os
import re
import glob
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from librosa.util import find_files


class PseudoDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(1000, 16000, 2)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
        # return: (time, 2)


class NoisyCleanDataset(Dataset):
    # This dataset identify the clean/noisy pair by the regex pattern in the filename
    # Each directory in roots should contain two sub-directories: clean & noisy
    # eg. The noisy file 'root/noisy/fileid_0.wav' has the label of the clean file 'root/clean/fileid_0.wav'
    def __init__(self, roots, noisy_channel=0, clean_channel=1, sample_seed=None, sample_ratio=0.8, select_sampled=True, regex='fileid_\d+'):
        clean_pths = []
        for root in roots:
            clean_pths.extend(find_files(os.path.join(root, 'clean')))
        clean_pths = sorted(clean_pths)

        if sample_seed is None:
            self.clean_pths = clean_pths
        else:
            random.seed(sample_seed)
            sampled = random.sample(clean_pths, round(len(clean_pths) * sample_ratio))
            if select_sampled:
                self.clean_pths = sampled
            else:
                self.clean_pths = [pth for pth in clean_pths if pth not in sampled]
        assert len(self.clean_pths) > 0

        self.noisy_channel = noisy_channel
        self.clean_channel = clean_channel
        self.regex_searcher = re.compile(regex)

    def __getitem__(self, idx):
        clean_pth = self.clean_pths[idx]
        result = self.regex_searcher.search(clean_pth)
        assert result is not None
        fileid = result.group()
        noisy_dir = os.path.dirname(clean_pth).replace('clean', 'noisy')
        noisy_pths = glob.glob(f'{noisy_dir}/*{fileid}*')
        file_searcher = re.compile(fileid + '\D')
        noisy_pths = [pth for pth in noisy_pths if file_searcher.search(pth) is not None]
        assert len(noisy_pths) == 1
        noisy_pth = noisy_pths[0]
       
        clean, sr1 = torchaudio.load(clean_pth)
        noisy, sr2 = torchaudio.load(noisy_pth)
        assert sr1 == sr2
        
        return torch.stack([noisy, clean], dim=-1).view(-1, 2)
        # return: (time, 2)

    def __len__(self):
        return len(self.clean_pths)