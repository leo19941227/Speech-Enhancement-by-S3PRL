import os
import math
import torch
import random
from tqdm import tqdm
from torch.optim import Adam
from tensorboardX import SummaryWriter
from downstream.solver import get_optimizer
from evaluation import pesq_eval, stoi_eval, estoi_eval
from objective import Stoi, Estoi, SI_SDR

OOM_RETRY_LIMIT = 10
METRIC_NUM = 3

class Runner():
    ''' Handler for complete training and evaluation progress of downstream models '''
    def __init__(self, args, runner_config, dataloader, preprocessor, upstream, downstream, expdir):
        self.device = torch.device('cuda') if (args.gpu and torch.cuda.is_available()) else torch.device('cpu')
        if torch.cuda.is_available(): print('[Runner] - CUDA is available!')
        self.model_kept = []
        self.global_step = 1
        self.log = SummaryWriter(expdir)

        self.args = args
        self.config = runner_config
        self.dataloader = dataloader
        self.preprocessor = preprocessor
        self.upstream_model = upstream.to(self.device)
        self.downstream_model = downstream.to(self.device)
        self.grad_clip = self.config['gradient_clipping']
        self.expdir = expdir
        self.criterion = None

        if self.config['loss'] == 'si_sdr':
            self.criterion = SI_SDR()

        elif self.config['loss'] == 'stoi':
            self.criterion = Stoi(self.device)

        elif self.config['loss'] == 'estoi':
            self.criterion = Estoi(self.device)

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


    def train(self):
        total_steps = int(self.config['epochs'] * len(self.dataloader['train']))
        pbar = tqdm(total=total_steps)
        
        loss_sum = 0
        metrics_best = {'dev': torch.zeros(METRIC_NUM), 'test': torch.zeros(METRIC_NUM)}
        while self.global_step <= total_steps:
            for wavs in self.dataloader['train']:
                # wavs: (batch_size, channel, max_len)
                try:
                    if self.global_step > total_steps:
                        break
                    
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

                    label_mask = (features.sum(dim=-1) != 0).long()
                    # label_mask: (batch_size, seq_len), LongTensor
                    # Since zero padding, some timestamps of features are not valid
                    # For each timestamps, we mark 1 on valid timestamps, and 0 otherwise
                    # This is useful for frame-wise loss computation

                    predicted = self.downstream_model(features)
                    loss = self.criterion(src = predicted, tar = linear_tar)
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
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # log
                    if self.global_step % int(self.config['log_step']) == 0:
                        loss_avg = loss_sum / self.config['log_step']
                        self.log.add_scalar('loss', loss_avg, self.global_step)
                        self.log.add_scalar('gradient norm', grad_norm, self.global_step)
                        pbar.set_description('Loss %.5f' % (loss_avg))
                        loses = 0

                    # evaluate and save the best
                    if self.global_step % int(self.config['eval_step']) == 0:
                        print(f'[Runner] - Evaluating on development set')
                        def evaluate(split):
                            loss, metrics = self.evaluate(split=split)
                            self.log.add_scalar(f'{split}_loss', loss.item(), self.global_step)
                            self.log.add_scalar(f'{split}_stoi', metrics[0].item(), self.global_step)
                            self.log.add_scalar(f'{split}_estoi', metrics[1].item(), self.global_step)
                            self.log.add_scalar(f'{split}_pesq', metrics[2].item(), self.global_step)

                            if (metrics > metrics_best[split]).sum() > 0:
                                metrics_best[split] = torch.max(metrics, metrics_best[split])
                                print('[Runner] - Saving new best model')
                                self.save_model(save_best=f'best_{split}')

                        if self.dataloader['dev'] is not None: evaluate('dev')
                        if self.dataloader['test'] is not None: evaluate('test')

                except RuntimeError as e:
                    if not 'CUDA out of memory' in str(e): raise
                    print('[Runner] - CUDA out of memory at step: ', self.global_step)
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()

                pbar.update(1)
                self.global_step += 1

        if self.dataloader['dev'] is not None: evaluate('dev')
        if self.dataloader['test'] is not None: evaluate('test')

        pbar.close()
        self.log.close()


    def evaluate(self, split):
        torch.cuda.empty_cache()
        self.upstream_model.eval()
        self.downstream_model.eval()
        
        loss_sum = 0
        oom_counter = 0
        metrics_sum = torch.zeros(METRIC_NUM)
        for wavs in tqdm(self.dataloader[split], desc="Iteration"):
            wavs = torch.randn(2, 2, 160000)
            with torch.no_grad():
                try:
                    wavs = wavs.to(device=self.device)
                    wav_inp = wavs[:, self.preprocessor.channel_inp, :]
                    wav_tar = wavs[:, self.preprocessor.channel_tar, :]

                    feats_for_upstream, linear_inp, phase_inp, linear_tar, phase_tar = self.preprocessor(wavs)
                    features = self.upstream_model(feats_for_upstream)
                    label_mask = (features.sum(dim=-1) != 0).long()
                    
                    predicted = self.downstream_model(features)

                    
                    metrics = torch.zeros(METRIC_NUM)
                    # here should be a istft process
                    # for i in range(predicted.shape[0]):
                    #     metrics[0] = stoi_eval(src = predicted[i],  tar = linear_tar[i])
                    #     metrics[1] = estoi_eval(src = predicted[i],  tar = linear_tar[i])
                    #     metrics[2] = pesq_eval(src = predicted[i],  tar = linear_tar[i])

                    loss = self.criterion(src = predicted,  tar = linear_tar)
                    
                    loss_sum += loss
                    metrics_sum += metrics

                except RuntimeError as e:
                    if not 'CUDA out of memory' in str(e): raise
                    if oom_counter >= OOM_RETRY_LIMIT: 
                        oom_counter = 0
                        break
                    oom_counter += 1
                    print(f'[Runner] - CUDA out of memory during testing {split} set, aborting after ' + str(10 - oom_counter) + ' more tries')
                    torch.cuda.empty_cache()
        
        n_sample = len(self.dataloader[split])
        loss_avg = loss_sum / n_sample
        metrics_avg = metrics_sum / n_sample
        print(f'[Runner] - {split} result: loss {loss_avg}, metrics {metrics_avg}')
        
        self.upstream_model.train()
        self.downstream_model.train()
        torch.cuda.empty_cache()

        return loss_avg, metrics_avg