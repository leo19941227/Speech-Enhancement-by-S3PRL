import os
import time
import glob
import math
import copy
import random
import signal
import multiprocessing as mp
from functools import partial
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from torch.optim import Adam
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# packages in S3PRL
from downstream.solver import get_optimizer
from utility.preprocessor import OnlinePreprocessor

from dataset import *
from evaluation import *
from objective import *
from utils import *

MAX_POSITIONS_LEN = 16000 * 50
ACTIVE_BUFFER_NUM = 4


def get_length_masks(lengths, ascending):
    # lengths: (batch_size, ) in cuda
    ascending = ascending[:lengths.max().item()].unsqueeze(0).expand(len(lengths), -1)
    length_masks = (ascending < lengths.unsqueeze(-1)).long()
    return length_masks


def mixing(cleans, noises, norm_fn, collate_fn, snrs, query_num=32):
    cleans = random.choices(cleans, k=query_num)
    noises = random.choices(noises, k=query_num)

    wavs = []
    for clean, noise in zip(cleans, noises):
        clean = norm_fn(torch.FloatTensor(clean))
        noise = torch.FloatTensor(noise)
        snr = random.choice(snrs)
        noisy, scaled_noise = add_noise(clean.unsqueeze(0), noise.unsqueeze(0), torch.ones(1) * snr)
        noisy, scaled_noise = noisy.squeeze(0), scaled_noise.squeeze(0)
        wav = torch.stack([noisy, clean, scaled_noise], dim=-1)
        wavs.append(wav)
    
    return collate_fn(wavs)


def scoring(args, config, preprocessor, model, criterion, ascending, lengths, wavs):
    feats_for_upstream, feats_for_downstream, linear_inp, phase_inp, linear_tar, phase_tar = preprocessor(wavs)

    if args.from_waveform:
        # nn_transformer will take care of feature extraction
        down_inp = wavs.transpose(1, 2)
    elif args.from_rawfeature:
        down_inp = feats_for_downstream

    stft_lengths = lengths // preprocessor._win_args['hop_length'] + 1
    stft_length_masks = get_length_masks(stft_lengths, ascending)

    predicted, model_results = model(features=down_inp, linears=linear_inp)
    log_predicted = model_results['log_predicted']

    def chunk(tensor):
        return tensor.chunk(tensor.size(0), dim=0)

    grads = []
    for pre, log_pre, tar, stft in zip(chunk(predicted), chunk(log_predicted), chunk(linear_tar), chunk(stft_length_masks)):
        loss, objective_results = criterion(predicted=pre, log_predicted=log_pre, linear_tar=tar, stft_length_masks=stft)
        model.zero_grad()
        loss.backward(retain_graph=True)

        grad = []
        for para in model.parameters():
            grad.append(para.grad.view(-1))
        grad = torch.cat(grad, dim=0)
        grads.append(grad.detach())

    grads = torch.stack(grads, dim=0)
    # grads: (batch_size, model_parameters)
    return grads


def matching(query_scores, key_scores):
    return torch.mm(key_scores, query_scores.mean(dim=0).unsqueeze(1)).reshape(-1)


def thresholding(match_scores):
    return match_scores > 0


def find_active_samples(parent_msg,
                        child_msg,
                        buffers,
                        args, config,
                        preprocessor, model, criterion,
                        pseudo_clean, pseudo_noise):

    torch.multiprocessing.set_sharing_strategy('file_system')
    def handler(buffers, current_buffers, n_sample, signum, frame):
        print('[Active] - Signal handler called with signal', signum)
        print('[Active] - Writing data to buffers in manager... ', end='')
        for key in current_buffers.keys():
            buffers[key] = current_buffers[key][:n_sample]
            current_buffers[key] = []
        print('done. Exiting')
        exit(0)

    current_buffers = {0: [], 1: [], 2: [], 3: []}
    print('[Active] - Register SIGTERM handler.')
    signal.signal(signal.SIGTERM, partial(handler, buffers, current_buffers, config['runner']['active_sample_num']))

    if args.active_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.active_device)
        device = 'cuda'
    else:
        device = 'cpu'

    preprocessor.to(device=device)
    model.to(device=device)
    criterion = criterion.to(device=device)

    # build scoring template
    ascending = torch.arange(MAX_POSITIONS_LEN).to(device)
    scoring_tmp = partial(scoring, args, config, preprocessor, model, criterion, ascending)

    # build pseudo waveforms
    pseudo_clean = [torch.FloatTensor(wav) for wav in pseudo_clean]
    pseudo_noise = [torch.FloatTensor(wav) for wav in pseudo_noise]

    # build queries
    query_set = eval(args.trainset)(**config[f'{args.trainset}_train'],
                                    pseudo_modes=[3],
                                    pseudo_clean=pseudo_clean,
                                    pseudo_noise=pseudo_noise)
    query_loader = DataLoader(query_set, batch_size=config['runner']['active_query_num'], shuffle=True,
                              num_workers=args.n_jobs, collate_fn=query_set.collate_fn)
    query_lengths, query_wavs, _ = next(iter(query_loader))
    query_scores = scoring_tmp(query_lengths.to(device), query_wavs.to(device))

    # build candidates
    train_set = eval(args.trainset)(**config[f'{args.trainset}_train'],
                                    pseudo_modes=list(range(ACTIVE_BUFFER_NUM)),
                                    pseudo_clean=pseudo_clean,
                                    pseudo_noise=pseudo_noise)
    
    parent_msg.put('start active sampling')
    while True:
        print('[Active] - Set up new dataloader.')
        train_loader = DataLoader(train_set, batch_size=config['dataloader']['batch_size'], shuffle=True,
                                  num_workers=args.n_jobs, collate_fn=train_set.collate_fn)

        print(f'[Active] - Dataloader batch num: {len(train_loader)}')
        for bszid, (lengths, wavs, cases) in enumerate(train_loader):
            lengths = lengths.to(device)
            wavs = wavs.to(device)
            scores = scoring_tmp(lengths, wavs)

            # matching
            match_scores = matching(query_scores, scores)
            is_match = thresholding(match_scores).nonzero().view(-1)

            for idx in is_match:
                current_buffers[cases[idx].item()].append({
                    'wavs': wavs[idx][:lengths[idx].cpu(), :].detach().cpu(),
                    'match_score': match_scores[idx].detach().cpu(),
                })

            try:
                message = child_msg.get_nowait()
            except:
                message = None

            if message is not None:
                print(f'[Active] - get message {message}')
                for key in list(current_buffers.keys()):
                    buffers[key] = current_buffers[key][:config['runner']['active_sample_num']]
                    current_buffers[key] = []
                print('[Active] - Finish preparing active samples.')
                parent_msg.put('finish')
                print('[Active] - Go on for finding new active samples.')

        print(f'[Active] - Dataloader exhuasted.')