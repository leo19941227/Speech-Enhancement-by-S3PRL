import os
import math
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from tensorboardX import SummaryWriter
from downstream.solver import get_optimizer
from joblib import Parallel, delayed
from evaluation import *
from objective import *
from utils import *

OOM_RETRY_LIMIT = 10
MAX_POSITIONS_LEN = 16000 * 50

class Runner():
    ''' Handler for complete training and evaluation progress of downstream models '''
    def __init__(self, args, runner_config, preprocessor, upstream, downstream, expdir, eps=1e-6):
        self.device = torch.device('cuda') if (args.gpu and torch.cuda.is_available()) else torch.device('cpu')
        if torch.cuda.is_available(): print('[Runner] - CUDA is available!')
        self.model_kept = []
        self.global_step = 1
        self.log = SummaryWriter(expdir)

        self.args = args
        self.config = runner_config
        self.preprocessor = preprocessor
        self.upstream_model = upstream.to(self.device)
        self.downstream_model = downstream.to(self.device)
        self.grad_clip = self.config['gradient_clipping']
        self.expdir = expdir
        self.metrics = [eval(f'{m}_eval') for m in runner_config['eval_metrics']]
        self.criterion = None
        self.ascending = torch.arange(MAX_POSITIONS_LEN).to(self.device)
        self.eps = eps

        if self.config['loss'] == 'si_sdr':
            self.criterion = SI_SDR()
        elif self.config['loss'] == 'stoi':
            self.criterion = Stoi(self.device)
        elif self.config['loss'] == 'estoi':
            self.criterion = Estoi(self.device)
        elif self.config['loss'] == 'l1':
            self.criterion = L1().to(device=self.device)

        assert self.metrics is not None
        assert self.criterion is not None

    def set_model(self):
        if self.args.fine_tune:
            self.upstream_model.train()
            param_optimizer = list(self.upstream_model.named_parameters()) + list(self.downstream_model.named_parameters())
            self.optimizer = get_optimizer(params=param_optimizer,
                                           lr=float(self.config['learning_rate']), 
                                           warmup_proportion=float(self.config['warmup_proportion']),
                                           training_steps=int(self.config['total_steps']))
        else:
            self.upstream_model.eval()
            self.optimizer = Adam(self.downstream_model.parameters(), lr=float(self.config['learning_rate']), betas=(0.9, 0.999))
        
        self.downstream_model.train()

    def save_model(self, name='states', save_best=None):
        all_states = {
            'Upstream': self.upstream_model.state_dict() if self.args.fine_tune else None,
            'Downstream': self.downstream_model.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'Global_step': self.global_step,
            'Settings': {
                'Config': self.config,
                'Paras': self.args,
            },
        }

        if save_best is not None:
            model_path = f'{self.expdir}/{save_best}.ckpt'
            torch.save(all_states, model_path)
            return

        model_path = f'{self.expdir}/{name}-{self.global_step}.ckpt'
        torch.save(all_states, model_path)
        self.model_kept.append(model_path)

        if len(self.model_kept) >= int(self.config['max_keep']):
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)

    def _get_length_masks(self, lengths):
        # lengths: (batch_size, ) in cuda
        ascending = self.ascending[:lengths.max().item()].unsqueeze(0).expand(len(lengths), -1)
        length_masks = (ascending < lengths.unsqueeze(-1)).long()
        return length_masks

    def train(self, trainloader, subtrainloader=None, devloader=None, testloader=None):
        total_steps = int(self.config['epochs'] * len(trainloader))
        pbar = tqdm(total=total_steps)

        variables = locals()
        eval_splits = self.config['eval_splits']
        eval_metrics = self.config['eval_metrics']
        eval_settings = [(split_name, eval(f'{split_name}loader', None, variables), torch.zeros(len(self.metrics)))
                          for split_name in eval_splits]
        # eval_settings: [(split_name, split_loader, split_current_best_metrics), ...]
        
        def eval_and_log():
            for split_name, split_loader, metrics_best in eval_settings:
                if split_loader is None:
                    continue
                print(f'[Runner] - Evaluating on {split_name} set')
                loss, scores, *eval_wavs = self.evaluate(split_loader)
                self.log.add_scalar(f'{split_name}_loss', loss.item(), self.global_step)
                for score, metric_name in zip(scores, eval_metrics):
                    self.log.add_scalar(f'{split_name}_{metric_name}', score.item(), self.global_step)
                for idx, wavs in enumerate(zip(*eval_wavs)):
                    for tag, wav in zip(['noisy', 'clean', 'enhanced'], wavs):
                        marker = f'{split_name}-{tag}-{idx}'
                        
                        # log audio
                        self.log.add_audio(f'{marker}.wav', wav.reshape(-1, 1), global_step=self.global_step,
                                           sample_rate=self.preprocessor._sample_rate)

                        # log spectrogram
                        feat = {'feat_type': 'linear', 'log': True}
                        linear = self.preprocessor(wav.reshape(1, 1, -1).to(self.device), [feat])[0]
                        linear = linear.detach().cpu().squeeze().transpose(0, 1).flip(dims=[0])
                        fig = plot_spectrogram(linear)
                        self.log.add_figure(f'{marker}.png', fig, global_step=self.global_step)

                if (scores > metrics_best).sum() > 0:
                    metrics_best.data = torch.max(scores, metrics_best).data
                    self.save_model(save_best=f'best_{split_name}')
        
        # start training
        loss_sum = 0
        while self.global_step <= total_steps:
            for lengths, wavs in trainloader:
                # wavs: (batch_size, channel, max_len)
                try:
                    if self.global_step > total_steps:
                        break
                    
                    lengths = lengths.to(device=self.device)
                    wavs = wavs.to(device=self.device)
                    wav_inp = wavs[:, self.preprocessor.channel_inp, :]
                    wav_tar = wavs[:, self.preprocessor.channel_tar, :]
                    
                    feats_for_upstream, linear_inp, phase_inp, linear_tar, phase_tar = self.preprocessor(wavs)
                    # all features are already in CUDA and in shape: (batch_size, max_time, feat_dim)
                    # For reconstruction waveform from linear spectrogram ((power=2)) and phase, use the following:
                    # wav = self.preprocessor.istft(linear, phase)

                    if self.args.fine_tune:
                        features = self.upstream_model(feats_for_upstream)
                    else:
                        with torch.no_grad():
                            features = self.upstream_model(feats_for_upstream)
                    # features: (batch_size, max_time, feat_dim)

                    stft_lengths = lengths // self.preprocessor._win_args['hop_length'] + 1
                    stft_length_masks = self._get_length_masks(stft_lengths)
                    assert stft_length_masks.size(-1) == features.size(-2)
                    # stft_length_masks: (batch_size, max_time)

                    predicted = self.downstream_model(features, linears=linear_inp)
                    loss = self.criterion(predicted, linear_tar, length_masks=stft_length_masks)
                    loss.backward()
                    loss_sum += loss.item()

                    # gradient clipping
                    up_paras = list(self.upstream_model.parameters())
                    down_paras = list(self.downstream_model.parameters())
                    if self.args.fine_tune: 
                        grad_norm = torch.nn.utils.clip_grad_norm_(up_paras + down_paras, self.grad_clip)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(down_paras, self.grad_clip)

                    # update parameters
                    if math.isnan(grad_norm) or math.isinf(grad_norm):
                        print('[Runner] - Error : grad norm is nan/inf at step ' + str(self.global_step))
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                    # log
                    if self.global_step % int(self.config['log_step']) == 0:
                        loss_avg = loss_sum / self.config['log_step']
                        self.log.add_scalar('loss', loss_avg, self.global_step)
                        self.log.add_scalar('gradient norm', grad_norm, self.global_step)
                        pbar.set_description('Loss %.5f' % (loss_avg))
                        loss_sum = 0

                    # evaluate and save the best
                    if self.global_step % int(self.config['eval_step']) == 0:
                        eval_and_log()

                except RuntimeError as e:
                    if not 'CUDA out of memory' in str(e): raise
                    print('[Runner] - CUDA out of memory at step: ', self.global_step)
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()

                pbar.update(1)
                self.global_step += 1

        eval_and_log()
        pbar.close()
        self.log.close()

    def evaluate(self, dataloader):
        torch.cuda.empty_cache()
        self.upstream_model.eval()
        self.downstream_model.eval()
        
        data_num = len(dataloader)
        sampled_wav_num = int(self.config['eval_log_wavs_num'])
        sample_interval = int(data_num / sampled_wav_num)
        sample_indices = list(range(0, data_num, sample_interval))[:sampled_wav_num]
        noisy_wavs, clean_wavs, enhanced_wavs = [], [], []

        loss_sum = 0
        oom_counter = 0
        scores_sum = torch.zeros(len(self.metrics))
        for indice, (lengths, wavs) in enumerate(tqdm(dataloader, desc="Iteration")):
            with torch.no_grad():
                try:
                    lengths = lengths.to(device=self.device)
                    wavs = wavs.to(device=self.device)
                    wav_inp = wavs[:, self.preprocessor.channel_inp, :]
                    wav_tar = wavs[:, self.preprocessor.channel_tar, :]
                    # wav: (batch_size, time)

                    feats_for_upstream, linear_inp, phase_inp, linear_tar, phase_tar = self.preprocessor(wavs)
                    features = self.upstream_model(feats_for_upstream)
                    # features: (batch_size, max_time, feat_dim)

                    stft_lengths = lengths // self.preprocessor._win_args['hop_length'] + 1
                    stft_length_masks = self._get_length_masks(stft_lengths)
                    assert stft_length_masks.size(-1) == features.size(-2)
                    # stft_length_masks: (batch_size, max_time)

                    predicted = self.downstream_model(features, linears=linear_inp)
                    loss = self.criterion(predicted, linear_tar, length_masks=stft_length_masks, noisy=linear_inp)
                    loss_sum += loss

                    wav_predicted = self.preprocessor.istft(predicted, phase_inp)
                    length_masks = self._get_length_masks(lengths)
                    wav_predicted = masked_normalize_decibel(wav_predicted, wav_tar, length_masks)

                    # split batch into list of utterances and duplicate N_METRICS times
                    batch_size = len(wav_predicted)
                    wav_predicted_list = wav_predicted.detach().cpu().chunk(batch_size) * len(self.metrics)
                    wav_tar_list = wav_tar.detach().cpu().chunk(batch_size) * len(self.metrics)
                    lengths_list = lengths.detach().cpu().tolist() * len(self.metrics)

                    # prepare metric function for each utterance in the duplicated list
                    ones = torch.ones(batch_size).long().unsqueeze(0).expand(len(self.metrics), -1)
                    metric_ids = ones * torch.arange(len(self.metrics)).unsqueeze(-1)
                    metric_fns = [self.metrics[idx.item()] for idx in metric_ids.reshape(-1)]
                    
                    def calculate_metric(length, predicted, target, metric_fn):
                        return metric_fn(predicted.squeeze()[:length], target.squeeze()[:length])
                    scores = Parallel(n_jobs=self.args.n_jobs)(delayed(calculate_metric)(l, p, t, f)
                                      for l, p, t, f in zip(lengths_list, wav_predicted_list, wav_tar_list, metric_fns))
                    
                    scores = torch.FloatTensor(scores).view(len(self.metrics), batch_size).mean(dim=1)
                    scores_sum += scores

                    if indice in sample_indices:
                        noisy_wavs.append(wav_inp[0].detach().cpu())
                        clean_wavs.append(wav_tar[0].detach().cpu())
                        enhanced_wavs.append(wav_predicted[0].detach().cpu())

                except RuntimeError as e:
                    if not 'CUDA out of memory' in str(e): raise
                    if oom_counter >= OOM_RETRY_LIMIT: 
                        oom_counter = 0
                        break
                    oom_counter += 1
                    torch.cuda.empty_cache()
        
        n_sample = len(dataloader)
        loss_avg = loss_sum / n_sample
        scores_avg = scores_sum / n_sample
        
        self.upstream_model.train()
        self.downstream_model.train()
        torch.cuda.empty_cache()

        return loss_avg, scores_avg, noisy_wavs, clean_wavs, enhanced_wavs