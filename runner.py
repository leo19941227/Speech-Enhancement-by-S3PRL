import os
import math
import torch
import random
from tqdm import tqdm
from torch.optim import Adam
from tensorboardX import SummaryWriter
from downstream.solver import get_optimizer

OOM_RETRY_LIMIT = 10

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
        best_metrics = {'dev': 0, 'test': 0}
        
        loses = 0
        metrics = 0
        while self.global_step <= total_steps:
            for wavs in self.dataloader['train']:
                # wavs: (batch_size, channel, max_len)
                try:
                    if self.global_step > total_steps:
                        break
                    
                    wavs = wavs.to(device=self.device)
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

                    loss, metric = self.downstream_model(features, linear_tar, label_mask)
                    loss.backward()
                    loses += loss.item()
                    metrics += metric

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
                        los = loses / self.config['log_step']
                        self.log.add_scalar('loss', los, self.global_step)
                        self.log.add_scalar('metric', metrics / self.config['log_step'], self.global_step)
                        self.log.add_scalar('gradient norm', grad_norm, self.global_step)
                        pbar.set_description('Loss %.5f' % (los))
                        loses = 0

                    # evaluate and save the best
                    if self.global_step % int(self.config['eval_step']) == 0:
                        print(f'[Runner] - Evaluating on development set')
                        def evaluate(split):
                            eval_loss, eval_metric = self.evaluate(split=split)
                            self.log.add_scalar(f'{split}_loss', eval_loss, self.global_step)
                            self.log.add_scalar(f'{split}_metric', eval_metric, self.global_step)
                            if eval_metric > best_metrics[split]:
                                best_metrics[split] = eval_metric
                                print('[Runner] - Saving new best model')
                                self.save_model(save_best='best_dev')
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
        
        loses = 0
        metrics = 0
        oom_counter = 0
        for wavs in tqdm(self.dataloader[split], desc="Iteration"):
            wavs = torch.randn(2, 2, 160000)
            with torch.no_grad():
                try:
                    wavs = wavs.to(device=self.device)
                    feats_for_upstream, linear_inp, phase_inp, linear_tar, phase_tar = self.preprocessor(wavs)
                    features = self.upstream_model(feats_for_upstream)
                    label_mask = (features.sum(dim=-1) != 0).long()
                    loss, metric = self.downstream_model(features, linear_tar, label_mask)
                    loses += loss.item()
                    metrics += metric

                except RuntimeError as e:
                    if not 'CUDA out of memory' in str(e): raise
                    if oom_counter >= OOM_RETRY_LIMIT: 
                        oom_counter = 0
                        break
                    oom_counter += 1
                    print(f'[Runner] - CUDA out of memory during testing {split} set, aborting after ' + str(10 - oom_counter) + ' more tries')
                    torch.cuda.empty_cache()
        
        n_sample = len(self.dataloader[split])
        average_loss = loses / n_sample
        average_metric = metrics / n_sample
        print(f'[Runner] - {split} result: loss {average_loss}, metric {average_metric}')
        
        self.upstream_model.train()
        self.downstream_model.train()
        torch.cuda.empty_cache()

        return average_loss, average_metric