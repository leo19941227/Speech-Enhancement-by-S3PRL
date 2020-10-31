import os
import time
import glob
import math
import copy
import random
import multiprocessing as mp
from functools import partial
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
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
from sampler import *

OOM_RETRY_LIMIT = 10
MAX_POSITIONS_LEN = 16000 * 50
LOG_WAV_NUM = 3


def logging(logger, step, tag, data, mode='scalar', preprocessor=None):
    if type(data) is torch.Tensor:
        data = data.detach().cpu().reshape(-1)

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
    def __init__(self, args, config, preprocessor, upstream, upstream2, downstream, expdir, eps=1e-6):
        self.device = torch.device('cuda') if (args.gpu and torch.cuda.is_available()) else torch.device('cpu')
        if torch.cuda.is_available(): print('[Runner] - CUDA is available!')
        self.global_step = 1
        self.log = SummaryWriter(expdir)
        self.logging = partial(logging, logger=self.log, preprocessor=copy.deepcopy(preprocessor).cpu())

        self.args = args
        self.config = config
        self.rconfig = config['runner']
        self.preprocessor = preprocessor.to(self.device)

        self.upstream_model = upstream.to(self.device)
        self.upstream_model2 = upstream2.to(self.device)
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
        self.parent_msg = self.ctx.Queue()
        self.child_msg = self.ctx.Queue()
        self.sampler_buffers = self.manager.dict()
        self.scoring_tmp = partial(scoring, self.args, self.config, self.preprocessor,
                                   self.downstream_model, self.criterion, self.ascending)

        self.pseudo_clean = None
        self.pseudo_noise = None


    def set_model(self):
        self.upstream_model.eval()
        if self.args.dropout is not None:
            self.upstream_model.train()

        self.upstream_model2.eval()
        if self.args.dropout2 is not None:
            self.upstream_model2.train()

        if self.args.optim == 'BertAdam':
            self.optimizer = get_optimizer(params=list(self.downstream_model.named_parameters()),
                                        lr=float(self.rconfig['learning_rate']), 
                                        warmup_proportion=float(self.rconfig['warmup_proportion']),
                                        training_steps=int(self.rconfig['total_step']))
        elif self.args.optim == 'Adam':
            self.optimizer = Adam(self.downstream_model.parameters(), lr=float(self.rconfig['learning_rate']), betas=(0.9, 0.999))

        self.downstream_model.train()
        if self.args.resume is not None:
            self.load_model(self.args.resume)


    def load_model(self, ckptpth):
        ckpt = torch.load(ckptpth)
        self.downstream_model.load_state_dict(ckpt['Downstream'])
        self.optimizer.load_state_dict(ckpt['Optimizer'])
        self.global_step = ckpt['Global_step']


    def save_model(self, save_type=None):
        all_states = {
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


    def get_dataset(self, mode='train'):
        split = 'train' if mode in ['subtrain', 'query'] else mode

        ds_type = eval(f'self.args.{split}set')
        ds_conf = self.config[f'{ds_type}_{split}']
        
        if type(ds_conf.get('pseudo_modes')) is list:
            if self.pseudo_clean is None or self.pseudo_noise is None:
                self._build_pseudo_wavs()
        
        dataset = eval(ds_type)(
            **ds_conf,
            pseudo_clean=self.pseudo_clean,
            pseudo_noise=self.pseudo_noise,
        )

        if mode == 'subtrain':
            dataset = dataset.get_subset(n_file=100)
        
        if mode == 'query':
            dataset.pseudo_modes = [3]

        print(f'[Dataset] - {mode} dataset is created.')
        return dataset

    
    def get_dataloader(self, dataset, train=True, bsz=None):
        if bsz is None:
            bsz = self.config['dataloader']['batch_size'] if train else self.config['dataloader']['eval_batch_size']

        return DataLoader(
            dataset,
            batch_size=bsz,
            shuffle=train,
            num_workers=self.args.n_jobs,
            collate_fn=dataset.collate_fn
        )


    def _get_length_masks(self, lengths):
        # lengths: (batch_size, ) in cuda
        ascending = self.ascending[:lengths.max().item()].unsqueeze(0).expand(len(lengths), -1)
        length_masks = (ascending < lengths.unsqueeze(-1)).long()
        return length_masks


    def _start_sampler(self):
        # create new active sampler based on current model
        self.child = self.ctx.Process(
            target=sampler_driver,
            args=(
                self.parent_msg,
                self.child_msg,
                self.sampler_buffers,
                self.args, self.config,
                copy.deepcopy(self.preprocessor).cpu(),
                copy.deepcopy(self.downstream_model).cpu(),
                copy.deepcopy(self.criterion).cpu(),
                self.pseudo_clean,
                self.pseudo_noise,
            )
        )
        self.child.start()
        message = self.parent_msg.get()
        print(f'[Runner] - get message {message}')


    def _kill_sampler(self):
        if hasattr(self, 'child'):
            self.child.terminate()
            self.child.join()
            delattr(self, 'child')


    def _collect_samples(self):
        print('[Runner] - Notify children to prepare samples.')
        self.child_msg.put('collect')
        message = self.parent_msg.get()
        print(f'[Runner] - get message {message}')

        print('[Runner] - Start collecting samples...')
        samples = {}
        for key in list(self.sampler_buffers.keys()):
            samples[key] = copy.deepcopy(self.sampler_buffers[key])
            print(f'[Runner] - key {key} gets {len(samples[key])} samples.')
            self.sampler_buffers.pop(key)
        return samples


    def _decode_wav(self, linear_tar, phase_inp, lengths, target_level=-25):
        wav = self.preprocessor.istft(linear_tar, phase_inp)
        wav = torch.cat([wav, wav.new_zeros(wav.size(0), max(lengths) - wav.size(1))], dim=1)
        wav = masked_normalize_decibel(wav, target_level, self._get_length_masks(lengths))
        return wav


    def _pseudo_clean(self, wavs, *args):
        with torch.no_grad():
            features = self.upstream_model(wavs.transpose(1, 2))
            linear_predicted, _ = self.upstream_model.SpecHead(features)
        return self._decode_wav(linear_predicted, *args)


    def _pseudo_noise(self, wavs, *args):
        with torch.no_grad():
            features = self.upstream_model2(wavs.transpose(1, 2))
            linear_predicted, _ = self.upstream_model2.SpecHead(features)
        return self._decode_wav(linear_predicted, *args)


    def _build_pseudo_wavs(self):
        recordset = self.get_dataset('record')
        recordloader = self.get_dataloader(recordset, train=False, bsz=recordset.__len__())
        lengths, wavs = next(iter(recordloader))
        self.logging(step=1, tag='record/noisy', data=wavs[:, 0, :], mode='audio')
        self.logging(step=1, tag='record/clean', data=wavs[:, 1, :], mode='audio')
        self.logging(step=1, tag='record/noise', data=wavs[:, 2, :], mode='audio')

        wavs = wavs.to(device=self.device)
        lengths = lengths.to(device=self.device)
        feats_for_upstream, feats_for_downstream, linear_inp, phase_inp, linear_tar, phase_tar = self.preprocessor(wavs)

        pseudo_clean = self._pseudo_clean(wavs, phase_inp, lengths).detach().cpu()
        self.logging(step=1, tag='record/pseudo_clean', data=pseudo_clean, mode='audio')
        self.pseudo_clean = [clean[:length] for clean, length in zip(pseudo_clean, lengths)]

        pseudo_noise = self._pseudo_noise(wavs, phase_inp, lengths).detach().cpu()
        self.logging(step=1, tag='record/pseudo_noise', data=pseudo_noise, mode='audio')
        self.pseudo_noise = [noise[:length] for noise, length in zip(pseudo_noise, lengths)]


    def train(self):
        total_steps = self.rconfig['total_step']
        pbar = tqdm(total=total_steps, dynamic_ncols=True)
        pbar.n = self.global_step - 1

        eval_settings = []
        eval_splits = self.rconfig['eval_splits']
        eval_metrics = self.rconfig['eval_metrics']
        for split_name in eval_splits:
            split_dataset = self.get_dataset(split_name)
            split_dataloader = self.get_dataloader(split_dataset)
            eval_settings.append((
                split_name,
                split_dataloader,
                torch.zeros(len(self.metrics)),
            ))
        
        def eval_and_log(log_media=False):
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

                if log_media:
                    for idx, wavs in enumerate(zip(*eval_wavs)):
                        for tag, wav in zip(['noisy', 'clean', 'enhanced'], wavs):
                            self.logging(step=self.global_step, tag=f'{split_name}-{tag}-{idx}', data=wav, mode='audio')

        if self.args.eval_init:
            eval_and_log()

        trainset = self.get_dataset('train')
        if self.args.sync_sampler:
            queryset = self.get_dataset('query')
            queryloader = self.get_dataloader(queryset, bsz=self.config['runner']['active_query_num'])
            queryloader_iter = iter(queryloader)
            trainloader = self.get_dataloader(trainset, bsz=self.config['dataloader']['active_batch_size'])
        else:
            trainloader = self.get_dataloader(trainset)

        # start training
        loss_sum = 0
        active_samples = defaultdict(lambda: defaultdict(list))
        while self.global_step <= total_steps:
            for batch in trainloader:
                if len(batch) == 2:
                    lengths, wavs = batch
                elif len(batch) == 3:
                    lengths, wavs, cases = batch
                else:
                    raise NotImplementedError

                train_loggers = []
                try:
                    if self.global_step > total_steps:
                        break

                    if self.args.sampler_device is not None:
                        if not hasattr(self, 'child') or not self.child.is_alive():
                            self._start_sampler()

                        if self.global_step % int(self.rconfig['sampler_collect_step']) == 0:
                            samples = self._collect_samples()
                            for key in samples.keys():
                                active_samples[self.global_step][key] += samples[key]

                    if self.args.sync_sampler:
                        try:
                            query_lengths, query_wavs, _  = next(queryloader_iter)
                        except:
                            queryloader_iter = iter(queryloader)
                            query_lengths, query_wavs, _  = next(queryloader_iter)

                        query_scores = self.scoring_tmp(query_lengths.to(self.device), query_wavs.to(self.device), mean=True)
                        train_scores = self.scoring_tmp(lengths.to(self.device), wavs.to(self.device))

                        # matching
                        match_scores = matching(query_scores, train_scores)
                        is_match = thresholding(match_scores).nonzero().view(-1)

                        wavs = wavs.detach().cpu()
                        match_scores = match_scores.detach().cpu()
                        for idx in is_match:
                            active_samples[self.global_step][cases[idx].item()].append({
                                'wavs': wavs[idx, :, :lengths[idx].cpu()].transpose(-1, -2).contiguous(),
                                'match_score': match_scores[idx],
                            })

                    if self.args.active_sampling:
                        prev_step = self.global_step - self.rconfig['active_refresh_step']
                        if prev_step > 1:
                            active_samples.pop(prev_step, None)

                        merged_samples = defaultdict(list)
                        for step_samples in active_samples.values():
                            for key, value in step_samples.items():
                                merged_samples[key] += value

                        pairs = torch.LongTensor([[i, w] for i, w in enumerate(self.rconfig['active_buffer_weights']) if len(merged_samples[i]) > 0])
                        if len(pairs.view(-1)) > 0:
                            keys = pairs[:, 0].tolist()
                            weights = pairs[:, 1].tolist()
                            types = random.choices(keys, weights, k=self.config['dataloader']['batch_size'])
                            wavs = [random.choice(merged_samples[t])['wavs'] for t in types]
                            lengths, wavs = trainloader.dataset.collate_fn(wavs)

                    wavs = wavs.to(device=self.device)
                    lengths = lengths.to(device=self.device)
                    feats_for_upstream, feats_for_downstream, linear_inp, phase_inp, linear_tar, phase_tar = self.preprocessor(wavs)

                    train_loggers.append(partial(self.logging, tag='noisy', data=wavs[:, 0, :], mode='audio'))
                    train_loggers.append(partial(self.logging, tag='clean', data=wavs[:, 1, :], mode='audio'))
                    train_loggers.append(partial(self.logging, tag='noise', data=wavs[:, 2, :], mode='audio'))

                    if self.args.pseudo_clean:
                        pseudo_clean = self._pseudo_clean(wavs, phase_inp, lengths)
                        train_loggers.append(partial(self.logging, tag='pseudo_clean', data=pseudo_clean, mode='audio'))

                    if self.args.pseudo_noise:
                        pseudo_noise = self._pseudo_noise(wavs, phase_inp, lengths)
                        train_loggers.append(partial(self.logging, tag='pseudo_noise', data=pseudo_noise, mode='audio'))

                    if self.args.from_waveform:
                        # nn_transformer will take care of feature extraction
                        down_inp = wavs.transpose(1, 2)
                    elif self.args.from_rawfeature:
                        down_inp = feats_for_downstream

                    predicted, model_results = self.downstream_model(features=down_inp, linears=linear_inp)

                    stft_lengths = lengths // self.preprocessor._win_args['hop_length'] + 1
                    stft_length_masks = self._get_length_masks(stft_lengths)

                    loss, objective_results = self.criterion(**remove_self(locals()), **model_results)
                    loss.backward()
                    loss_sum += loss.item()

                    # gradient clipping
                    down_paras = list(self.downstream_model.parameters())
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
                    
                    log_media = self.global_step % int(self.rconfig['media_step']) == 0
                    if log_media:
                        for logger in train_loggers:
                            logger(step=self.global_step)

                    if self.args.active_sampling and self.global_step % int(self.rconfig['sampler_refresh_step']) == 0:
                        self._kill_sampler()

                    # evaluate and save the best
                    if self.global_step % int(self.rconfig['eval_step']) == 0:
                        self.save_model()
                        eval_and_log(log_media)

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

        if hasattr(self, 'child') and self.child.is_alive():
            self.child.terminate()
            self.child.join()

        pbar.close()
        self.log.close()


    def evaluate(self, dataloader=None):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        torch.cuda.empty_cache()
        self.upstream_model.eval()
        self.upstream_model2.eval()
        self.downstream_model.eval()

        if dataloader is None:
            testset = self.get_dataset('test')
            dataloader = self.get_dataloader(testset, train=False)

        data_num = len(dataloader)
        sample_interval = int(data_num / LOG_WAV_NUM)
        sample_indices = list(range(0, data_num, sample_interval))[:LOG_WAV_NUM]
        noisy_wavs, clean_wavs, enhanced_wavs = [], [], []

        loss_sum = 0
        oom_counter = 0
        scores_sum = torch.zeros(len(self.metrics))
        for indice, batch in enumerate(tqdm(dataloader, desc="Iteration", dynamic_ncols=True)):
            with torch.no_grad():
                if len(batch) == 2:
                    lengths, wavs = batch
                elif len(batch) == 3:
                    lengths, wavs, cases = batch
                else:
                    raise NotImplementedError

                try:
                    wavs = wavs.to(device=self.device)
                    lengths = lengths.to(device=self.device)
                    feats_for_upstream, feats_for_downstream, linear_inp, phase_inp, linear_tar, phase_tar = self.preprocessor(wavs)

                    wav_inp = wavs[:, self.preprocessor.channel_inp, :]
                    wav_tar = wavs[:, self.preprocessor.channel_tar, :]

                    if self.args.from_waveform:
                        # nn_transformer will take care of feature extraction
                        down_inp = wavs.transpose(1, 2)
                    elif self.args.from_rawfeature:
                        down_inp = feats_for_downstream

                    predicted, model_results = self.downstream_model(features=down_inp, linears=linear_inp)
                    wav_predicted = self._decode_wav(predicted, phase_inp, lengths, wav_tar)
                
                    stft_lengths = lengths // self.preprocessor._win_args['hop_length'] + 1
                    stft_length_masks = self._get_length_masks(stft_lengths)

                    loss, _ = self.criterion(**remove_self(locals()), **model_results)
                    loss_sum += loss

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

        print(f'[Runner evaluate]: loss {loss_avg}, scores {scores_avg}')
        return loss_avg, scores_avg, noisy_wavs, clean_wavs, enhanced_wavs


    def test_gradient(self):
        from ipdb import set_trace

        self._build_pseudo_wavs()

        query_set = eval(self.args.trainset)(
            **self.config[f'{self.args.trainset}_train'],
            pseudo_modes=[3],
            pseudo_clean=self.pseudo_clean,
            pseudo_noise=self.pseudo_noise,
        )
        query_loader = iter(DataLoader(
            query_set, batch_size=self.config['dataloader']['batch_size'], shuffle=True,
            num_workers=self.args.n_jobs, collate_fn=query_set.collate_fn
        ))

        train_set = eval(self.args.trainset)(
            **self.config[f'{self.args.trainset}_train'],
            pseudo_modes=list(range(ACTIVE_BUFFER_NUM)),
            pseudo_clean=self.pseudo_clean,
            pseudo_noise=self.pseudo_noise,
        )
        train_loader = iter(DataLoader(
            train_set, batch_size=self.config['dataloader']['batch_size'], shuffle=True,
            num_workers=self.args.n_jobs, collate_fn=train_set.collate_fn
        ))

        similarities = defaultdict(list)
        for i in tqdm(range(self.args.n_iterate), dynamic_ncols=True):
            query_lengths, query_wavs, _ = next(query_loader)
            train_lengths, train_wavs, cases = next(train_loader)

            if query_wavs.shape == train_wavs.shape and torch.allclose(query_wavs, train_wavs):
                print('Skip when qeury_wavs == train_wavs')
                continue

            query_score = self.scoring_tmp(query_lengths.to(self.device), query_wavs.to(self.device)).mean(dim=0, keepdim=True)
            train_score = self.scoring_tmp(train_lengths.to(self.device), train_wavs.to(self.device))

            query_score_norm = query_score / (query_score.pow(2).sum(dim=-1, keepdim=True).pow(0.5) + self.eps)
            train_score_norm = train_score / (train_score.pow(2).sum(dim=-1, keepdim=True).pow(0.5) + self.eps)

            similarity = (query_score_norm * train_score_norm).sum(dim=-1).view(-1)
            for sim, case in zip(similarity, cases):
                similarities[case.item()].append(sim.item())

        plt.figure()
        sims = [similarities[i] for i in range(4)]
        plt.boxplot(sims)
        plt.savefig(f'{self.expdir}/sim_box.png')
