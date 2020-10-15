import os
import time
import glob
import math
import copy
import torch
import random
from functools import partial
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from tensorboardX import SummaryWriter
from downstream.solver import get_optimizer
from joblib import Parallel, delayed
from dataloader import OnlineDataset
from utility.preprocessor import OnlinePreprocessor
from evaluation import *
from objective import *
from utils import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import multiprocessing as mp

OOM_RETRY_LIMIT = 10
MAX_POSITIONS_LEN = 16000 * 50


def collate_fn(samples):
    if type(samples[0]) is torch.Tensor:
        wavs = samples
    else:
        parse = [[samples[i][j] for i in range(len(samples))] for j in range(len(samples[0]))]
        wavs, channel3s = parse
    lengths = torch.LongTensor([len(s) for s in wavs])
    samples = pad_sequence(wavs, batch_first=True).transpose(-1, -2).contiguous()
    channel3_lengths = torch.LongTensor([len(s) for s in channel3s]) if 'channel3s' in locals() else lengths
    channel3_samples = pad_sequence(channel3s, batch_first=True) if 'channel3s' in locals() else samples[:, 0, :]
    return lengths, samples, channel3_lengths, channel3_samples

def get_dataloader(args, config):
    train_set = eval(args.trainset)(**config[f'{args.trainset}_train'])
    dlconf = config['dataloader']
    train_loader = DataLoader(train_set, batch_size=dlconf['batch_size'], shuffle=True, num_workers=args.n_jobs, collate_fn=collate_fn)
    return train_loader

def find_active_samples(handshake, active_samples, args, config, model, preprocessor, gpu2=1):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu2)
    model = model.cuda()
    preprocessor = preprocessor.cuda()
    trainloader = get_dataloader(args, config)

    pivot = 0
    for lengths, wavs, channel3_lengths, channel3_wavs in trainloader:
        handshake.put(0)
        active_samples[pivot] = torch.randn(16000, 2)
        pivot = (pivot + 1) % len(active_samples)


def logging(logger, step, tag, data, mode='scalar', preprocessor=None):
    if type(data) is torch.Tensor:
        data = data.detach().cpu()

    if mode == 'scalar':
        # data is a int or float
        logger.add_scalar(tag, data, global_step=step)
    elif mode == 'audio':
        # data: (seqlen, )
        assert preprocessor is not None
        data = data / data.abs().max().item()
        # log wavform
        logger.add_audio(f'{tag}.wav', data.reshape(-1, 1), global_step=step, sample_rate=preprocessor._sample_rate)
        # log log-linear-scale spectrogram
        feat_config = OnlinePreprocessor.get_feat_config(feat_type='linear', log=True)
        linear = preprocessor(data.reshape(1, 1, -1), [feat_config])[0]
        figure = plot_spectrogram(linear)
        logger.add_figure(f'{tag}.png', figure, global_step=step)
    else:
        raise NotImplementedError


class Runner():
    ''' Handler for complete training and evaluation progress of downstream models '''
    def __init__(self, args, config, preprocessor, upstream, downstream, expdir, eps=1e-6):
        self.device = torch.device('cuda') if (args.gpu and torch.cuda.is_available()) else torch.device('cpu')
        if torch.cuda.is_available(): print('[Runner] - CUDA is available!')
        self.global_step = 1
        self.log = SummaryWriter(expdir)

        self.args = args
        self.config = config
        self.rconfig = config['runner']
        self.preprocessor = preprocessor.to(self.device)
        self.upstream_model = upstream.to(self.device)
        self.downstream_model = downstream.to(self.device)
        self.grad_clip = self.rconfig['gradient_clipping']
        self.expdir = expdir
        self.metrics = [eval(f'{m}_eval') for m in self.rconfig['eval_metrics']]
        self.ascending = torch.arange(MAX_POSITIONS_LEN).to(device=self.device)
        self.eps = eps

        criterion_config = config['objective'][args.objective] if args.objective in config['objective'] else {}
        self.criterion = eval(f'{self.args.objective}(**criterion_config)').to(device=self.device)

        assert self.metrics is not None
        assert self.criterion is not None

        self.ctx = mp.get_context("spawn")
        self.manager = self.ctx.Manager()
        self.active_samples = self.manager.list(range(self.config['runner']['active_num']))

    def set_model(self):
        if self.args.fine_tune:
            self.upstream_model.train()
            param_optimizer = list(self.upstream_model.named_parameters()) + list(self.downstream_model.named_parameters())
            self.optimizer = get_optimizer(params=param_optimizer,
                                           lr=float(self.rconfig['learning_rate']), 
                                           warmup_proportion=float(self.rconfig['warmup_proportion']),
                                           training_steps=int(self.rconfig['total_step']))
        else:
            if self.args.random_label:
                self.upstream_model.train()
            else:
                self.upstream_model.eval()

            # self.optimizer = Adam(self.downstream_model.parameters(), lr=float(self.rconfig['learning_rate']), betas=(0.9, 0.999))
            self.optimizer = get_optimizer(params=list(self.downstream_model.named_parameters()),
                                            lr=float(self.rconfig['learning_rate']), 
                                            warmup_proportion=float(self.rconfig['warmup_proportion']),
                                            training_steps=int(self.rconfig['total_step']))

        self.downstream_model.train()
        if self.args.resume is not None:
            self.load_model(self.args.resume)

    def load_model(self, ckptpth):
        ckpt = torch.load(ckptpth)
        if self.args.fine_tune:
            self.upstream_model.load_state_dict(ckpt['Upstream'])
        self.downstream_model.load_state_dict(ckpt['Downstream'])
        self.optimizer.load_state_dict(ckpt['Optimizer'])
        self.global_step = ckpt['Global_step']

    def save_model(self, save_type=None):
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

        def check_ckpt_num(directory):
            ckpts = glob.glob(f'{directory}/states-*.ckpt')
            if len(ckpts) >= self.rconfig['max_keep']:
                ckpts = sorted(ckpts, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                for ckpt in ckpts[:len(ckpts) - self.rconfig['max_keep']]:
                    os.remove(ckpt)

        save_dir = self.expdir if save_type is None else f'{self.expdir}/{save_type}'
        os.makedirs(save_dir, exist_ok=True)
        check_ckpt_num(save_dir)
        model_path = f'{save_dir}/states-{self.global_step}.ckpt'
        torch.save(all_states, model_path)

    def _get_length_masks(self, lengths):
        # lengths: (batch_size, ) in cuda
        ascending = self.ascending[:lengths.max().item()].unsqueeze(0).expand(len(lengths), -1)
        length_masks = (ascending < lengths.unsqueeze(-1)).long()
        return length_masks

    def _active_sample(self):
        active_samples = [sample for sample in self.active_samples if type(sample) is torch.Tensor]

        # kill existing active sampler
        if hasattr(self, 'child'):
            self.child.terminate()

        # create new active sampler based on current model
        handshake = self.ctx.Queue()
        self.active_samples = self.manager.list(range(self.config['runner']['active_num']))
        self.child = self.ctx.Process(
            target=find_active_samples,
            args=(handshake, self.active_samples, self.args, self.config, copy.deepcopy(self.downstream_model).cpu(), copy.deepcopy(self.preprocessor).cpu())
        )
        self.child.start()
        handshake.get()

        return active_samples

    def train(self, trainloader, subtrainloader=None, devloader=None, testloader=None):
        total_steps = self.rconfig['total_step']
        pbar = tqdm(total=total_steps, dynamic_ncols=True)
        pbar.n = self.global_step - 1

        variables = locals()
        eval_splits = self.rconfig['eval_splits']
        eval_metrics = self.rconfig['eval_metrics']
        eval_settings = [(split_name, eval(f'{split_name}loader', None, variables), torch.zeros(len(self.metrics)))
                          for split_name in eval_splits]
        # eval_settings: [(split_name, split_loader, split_current_best_metrics), ...]
        
        def eval_and_log(media=False):
            for split_name, split_loader, metrics_best in eval_settings:
                if split_loader is None:
                    continue
                print(f'[Runner] - Evaluating on {split_name} set')
                loss, scores, *eval_wavs = self.evaluate(split_loader)
                self.log.add_scalar(f'{split_name}_loss', loss.item(), self.global_step)
                for score, metric_name in zip(scores, eval_metrics):
                    self.log.add_scalar(f'{split_name}_{metric_name}', score.item(), self.global_step)

                if (scores > metrics_best).sum() > 0:
                    metrics_best.data = torch.max(scores, metrics_best).data
                    self.save_model(split_name)

                if media:
                    for idx, wavs in enumerate(zip(*eval_wavs)):
                        for tag, wav in zip(['noisy', 'clean', 'enhanced'], wavs):
                            marker = f'{split_name}-{tag}-{idx}'
                            
                            # log audio
                            self.log.add_audio(f'{marker}.wav', wav.reshape(-1, 1), global_step=self.global_step,
                                            sample_rate=self.preprocessor._sample_rate)

                            # log spectrogram
                            feat = {'feat_type': 'linear', 'log': True}
                            linear = self.preprocessor(wav.reshape(1, 1, -1).to(self.device), [feat])[0]
                            fig = plot_spectrogram(linear)
                            self.log.add_figure(f'{marker}.png', fig, global_step=self.global_step)

        if self.args.eval_init:
            eval_and_log()

        if self.args.active_sampling:
            active_samples = self._active_sample()
        
        # start training
        logging_temp = partial(logging, logger=self.log, preprocessor=copy.deepcopy(self.preprocessor).cpu())
        loss_sum = 0
        while self.global_step <= total_steps:
            for lengths, wavs, channel3_lengths, channel3_wavs in trainloader:
                # wavs: (batch_size, channel, max_len)
                eval_loggers = []
                try:
                    if self.global_step > total_steps:
                        break
                    
                    lengths = lengths.to(device=self.device)
                    wavs = wavs.to(device=self.device)
                    feats_for_upstream, feats_for_downstream, linear_inp, phase_inp, linear_tar, phase_tar = self.preprocessor(wavs)
                    if self.args.upstream == 'transformer':
                        feats_for_upstream = wavs.transpose(1, 2)
                    features = self.upstream_model(feats_for_upstream)
                    # all features are already in CUDA and in shape: (batch_size, max_time, feat_dim)
                    # For reconstruction waveform from linear spectrogram ((power=2)) and phase, use the following:
                    # wav = self.preprocessor.istft(linear, phase)

                    if self.args.fine_tune:
                        features = self.upstream_model(feats_for_upstream)
                    else:
                        with torch.no_grad():
                            features = self.upstream_model(feats_for_upstream)
                    # features: (batch_size, max_time, feat_dim)

                    if self.args.pseudo_clean or self.args.pseudo_noise:
                        with torch.no_grad():
                            linear_predicted, _ = self.upstream_model.SpecHead(features)
                            if self.args.pseudo_clean:
                                linear_tar = linear_predicted
                            else:
                                # add logging for debug
                                eval_loggers.append(partial(logging_temp, tag='original_noisy', data=wavs[0, 0, :], mode='audio'))
                                eval_loggers.append(partial(logging_temp, tag='original_clean', data=wavs[0, 1, :], mode='audio'))
                                eval_loggers.append(partial(logging_temp, tag='database_clean', data=channel3_wavs[0], mode='audio'))

                                channel3_lengths = channel3_lengths.to(device=self.device)
                                channel3_wavs = channel3_wavs.to(device=self.device)
                                wav_noise = self.preprocessor.istft(linear_predicted, phase_inp)
                                wav_noisy, scaled_noise = OnlineDataset.add_noise(channel3_wavs, wav_noise, trainloader.dataset.snrs)

                                lengths = channel3_lengths
                                length_masks = self._get_length_masks(channel3_lengths)
                                wav_noisy, scaled_noise = wav_noisy * length_masks, scaled_noise * length_masks
                                wavs = torch.stack([wav_noisy, channel3_wavs], dim=1)
                                _, feats_for_downstream, linear_inp, phase_inp, linear_tar, phase_tar = self.preprocessor(wavs)

                                # add logging for debug
                                eval_loggers.append(partial(logging_temp, tag='pseudo_noise', data=wav_noise[0], mode='audio'))
                                eval_loggers.append(partial(logging_temp, tag='pseudo_noisy', data=wav_noisy[0], mode='audio'))

                    stft_lengths = lengths // self.preprocessor._win_args['hop_length'] + 1
                    stft_length_masks = self._get_length_masks(stft_lengths)
                    # stft_length_masks: (batch_size, max_time)

                    down_inp = features
                    if self.args.from_waveform:
                        # nn_transformer will take care of feature extraction
                        down_inp = wavs.transpose(1, 2)
                    elif self.args.from_rawfeature:
                        down_inp = feats_for_downstream

                    predicted, model_results = self.downstream_model(features=down_inp, linears=linear_inp)

                    loss, objective_results = self.criterion(**remove_self(locals()), **model_results)
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
                    if self.global_step % int(self.rconfig['log_step']) == 0:
                        loss_avg = loss_sum / self.rconfig['log_step']
                        self.log.add_scalar('loss', loss_avg, self.global_step)
                        self.log.add_scalar('gradient norm', grad_norm, self.global_step)
                        pbar.set_description('Loss %.5f' % (loss_avg))
                        loss_sum = 0
                        
                        for results in [model_results, objective_results]:
                            if 'logger' in results:
                                results['logger'](self.log, self.global_step)

                    # evaluate and save the best
                    if self.global_step % int(self.rconfig['eval_step']) == 0:
                        self.save_model()

                        # log wav and images
                        media = self.global_step % int(self.rconfig['media_step']) == 0
                        eval_and_log(media)
                        if media:
                            for logger in eval_loggers:
                                logger(step=self.global_step)

                    if self.args.active_sampling and self.global_step % int(self.rconfig['active_step']) == 0:
                        active_samples = self._active_sample()

                    del model_results
                    del objective_results
                    torch.cuda.empty_cache()

                except RuntimeError as e:
                    if not 'CUDA out of memory' in str(e): raise
                    print('[Runner] - CUDA out of memory at step: ', self.global_step)
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()

                pbar.update(1)
                self.global_step += 1
            
            self.child.terminate()

        pbar.close()
        self.log.close()

    def evaluate(self, dataloader):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        torch.cuda.empty_cache()
        self.upstream_model.eval()
        self.downstream_model.eval()
        
        data_num = len(dataloader)
        sampled_wav_num = int(self.rconfig['eval_log_wavs_num'])
        sample_interval = int(data_num / sampled_wav_num)
        sample_indices = list(range(0, data_num, sample_interval))[:sampled_wav_num]
        noisy_wavs, clean_wavs, enhanced_wavs = [], [], []

        loss_sum = 0
        oom_counter = 0
        scores_sum = torch.zeros(len(self.metrics))
        for indice, (lengths, wavs, _, _) in enumerate(tqdm(dataloader, desc="Iteration", dynamic_ncols=True)):
            with torch.no_grad():
                try:
                    lengths = lengths.to(device=self.device)
                    wavs = wavs.to(device=self.device)
                    wav_inp = wavs[:, self.preprocessor.channel_inp, :]
                    wav_tar = wavs[:, self.preprocessor.channel_tar, :]
                    # wav: (batch_size, time)

                    feats_for_upstream, feats_for_downstream, linear_inp, phase_inp, linear_tar, phase_tar = self.preprocessor(wavs)
                    if self.args.upstream == 'transformer':
                        feats_for_upstream = wavs.transpose(1, 2)
                    features = self.upstream_model(feats_for_upstream)
                    # features: (batch_size, max_time, feat_dim)

                    stft_lengths = lengths // self.preprocessor._win_args['hop_length'] + 1
                    stft_length_masks = self._get_length_masks(stft_lengths)
                    assert stft_length_masks.size(-1) == features.size(-2)
                    # stft_length_masks: (batch_size, max_time)

                    down_inp = features
                    if self.args.from_waveform:
                        # nn_transformer will take care of feature extraction
                        down_inp = wavs.transpose(1, 2)
                    elif self.args.from_rawfeature:
                        down_inp = feats_for_downstream

                    predicted, model_results = self.downstream_model(features=down_inp, linears=linear_inp)
                    wav_predicted = self.preprocessor.istft(predicted, phase_inp)
                    wav_predicted = torch.cat([wav_predicted, wav_predicted.new_zeros(wav_predicted.size(0), max(lengths) - wav_predicted.size(1))], dim=1)
                    length_masks = self._get_length_masks(lengths)
                
                    loss, _ = self.criterion(**remove_self(locals()), **model_results)
                    loss_sum += loss

                    wav_predicted = masked_normalize_decibel(wav_predicted, wav_tar, length_masks)

                    if indice in sample_indices:
                        noisy_wavs.append(wav_inp[0].detach().cpu())
                        clean_wavs.append(wav_tar[0].detach().cpu())
                        enhanced_wavs.append(wav_predicted[0].detach().cpu())

                    if self.args.no_metric:
                        continue

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