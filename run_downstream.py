import os
import copy
import glob
import random
import argparse
from shutil import copyfile
from functools import partial
from importlib import import_module

import yaml
import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# packages in S3PRL
from utility.preprocessor import OnlinePreprocessor
from transformer.nn_transformer import TRANSFORMER
from downstream.model import dummy_upstream

from dataset import *
from runner import *
from model import *
from utils import *


def get_downstream_args():
    parser = argparse.ArgumentParser(description='Argument Parser for Downstream Tasks of the S3PLR project.')
    parser.add_argument('--resume', help='Specify the downstream checkpoint path for continual training')

    parser.add_argument('--name', help='Name of current experiment.')
    parser.add_argument('--n_jobs', default=12, type=int)
    parser.add_argument('--dev_num', default=500, type=int)

    # upstream settings
    parser.add_argument('--upstream', choices=['transformer', 'baseline'], default='transformer', help='Specify the teacher model for distillation', required=False)
    parser.add_argument('--ckpt', default='', help='Path to upstream pre-trained checkpoint, required if using other than baseline', required=False)
    parser.add_argument('--dropout', type=float)

    # upstream2 settings
    parser.add_argument('--upstream2', choices=['transformer', 'baseline'], default='transformer', required=False)
    parser.add_argument('--ckpt2', default='', help='Path to upstream pre-trained checkpoint, required if using other than baseline', required=False)
    parser.add_argument('--dropout2', type=float)

    # how to use upstreams
    parser.add_argument('--pseudo_clean', action='store_true')
    parser.add_argument('--pseudo_noise', action='store_true')

    # Options
    parser.add_argument('--downstream', default='LSTM', required=False)
    parser.add_argument('--dckpt', default='', help='Path to upstream pre-trained checkpoint, required if using other than baseline', required=False)
    parser.add_argument('--objective', default='L1', required=False)
    parser.add_argument('--from_waveform', action='store_true')
    parser.add_argument('--from_rawfeature', action='store_true')

    # optimizer
    parser.add_argument('--optim', default='BertAdam')

    parser.add_argument('--config', default='config/vcb.yaml', help='Path to downstream experiment config.', required=False)
    parser.add_argument('--expdir', default='result', help='Path to store experiment result, if empty then default is used.', required=False)
    parser.add_argument('--seed', default=1337, type=int, help='Random seed for reproducable results.', required=False)
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--eval_init', action='store_true')
    parser.add_argument('--no_metric', action='store_true')
    parser.add_argument('--save_best', action='store_true')

    parser.add_argument('--active_sampling', action='store_true')
    parser.add_argument('--record_num', default=5, type=int)
    parser.add_argument('--sampler_device', type=int)
    parser.add_argument('--active_layerid', type=int)
    parser.add_argument('--n_iterate', type=int)
    parser.add_argument('--sync_sampler', action='store_true')

    parser.add_argument('--train_speech')
    parser.add_argument('--train_noise')
    parser.add_argument('--test_speech')
    parser.add_argument('--test_noise')

    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_gradient', action='store_true')

    # parse
    args = parser.parse_args()
    if args.resume is None:
        setattr(args, 'gpu', not args.cpu)
        config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
        for overwrite in ['train_speech', 'train_noise', 'test_speech', 'test_noise']:
            filestrs = eval(f'args.{overwrite}')
            if filestrs is None: continue
            dataset_type, data_type = overwrite.split('_')
            config[f'OnlineDataset_{dataset_type}'][data_type]['filestrs'] = filestrs
    else:
        if os.path.isdir(args.resume):
            ckpts = glob.glob(f'{args.resume}/*.ckpt')
            assert len(ckpts) > 0
            ckpts = sorted(ckpts, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
            resume_ckpt = ckpts[-1]
        else:
            resume_ckpt = args.resume

        ckpt = torch.load(resume_ckpt, map_location='cpu')
        args = update_args(args, ckpt['Settings']['Paras'])
        config = ckpt['Settings']['Config']
        setattr(args, 'resume', resume_ckpt)
    
    if args.wandb:
        wandb = import_module('wandb')
        if args.resume is None:
            wandb.init(name=args.name, sync_tensorboard=True)
            setattr(args, 'wandbid', wandb.run.id)
            wandb.config.update({
                'args': vars(args),
                'config': config
            })
        else:
            wandb.init(name=args.name, resume=args.wandbid, sync_tensorboard=True)
    
    return args, config


def get_preprocessor(args, config):
    # load the same preprocessor as pretraining stage
    # can also be used for extracting baseline feature (for fair comparison)
    if args.ckpt != '':
        pretrain_config = torch.load(args.ckpt, map_location='cpu')['Settings']['Config']
    else:
        pretrain_config = yaml.load(open('config/pretrain_sample.yaml', 'r'), Loader=yaml.FullLoader)

    if args.upstream == 'transformer':
        upstream_feat = pretrain_config['online']['input']
    else:
        upstream_feat = config['preprocessor']['baseline']

    if args.dckpt != '':
        downstream_config = torch.load(args.dckpt, map_location='cpu')['Settings']['Config']
        if 'online' in downstream_config:
            downstream_feat = downstream_config['online']['input']
        else:
            downstream_feat = downstream_config['preprocessor']['baseline']
    else:
        downstream_feat = config['preprocessor']['baseline']

    channel_inp = config['preprocessor']['input_channel']
    channel_tar = config['preprocessor']['target_channel']
    upstream_feat['channel'] = channel_inp
    downstream_feat['channel'] = channel_inp
    
    feat_list = [
        upstream_feat,
        downstream_feat,
        OnlinePreprocessor.get_feat_config('linear', channel_inp),
        OnlinePreprocessor.get_feat_config('phase', channel_inp),
        OnlinePreprocessor.get_feat_config('linear', channel_tar),
        OnlinePreprocessor.get_feat_config('phase', channel_tar),
    ]

    preprocessor = OnlinePreprocessor(**pretrain_config['online'], feat_list=feat_list)
    setattr(preprocessor, 'channel_inp', channel_inp)
    setattr(preprocessor, 'channel_tar', channel_tar)
    
    upstream_feat, downstream_feat, inp_linear, inp_phase, tar_linear, tar_phase = preprocessor()
    return preprocessor, upstream_feat.size(-1), downstream_feat.size(-1), tar_linear.size(-1)


def get_upstream_model(input_dim, upstream, ckpt, dropout):
    print('[run_downstream] - getting upstream model:', upstream)
    if upstream == 'transformer':
        options = {'ckpt_file'     : ckpt,
                   'load_pretrain' : 'True',
                   'no_grad'       : 'False',
                   'dropout'       : 'default' if dropout is None else dropout,
                   'spec_aug'      : 'False',
                   'spec_aug_prev' : 'True',
                   'weighted_sum'  : 'False',
                   'select_layer'  : -1,
                   'permute_input' : 'False',
        }
        # get input_dim for specific checkpoint
        pretrain_config = torch.load(ckpt, map_location='cpu')['Settings']['Config']
        preprocessor = OnlinePreprocessor(**pretrain_config['online'])
        inp_feat, tar_feat = preprocessor(feat_list=[pretrain_config['online']['input'], pretrain_config['online']['target']])
        upstream_model = TRANSFORMER(options, inp_feat.size(-1))
        setattr(upstream_model, 'SpecHead', SpecHead(tar_feat.size(-1), ckpt))

    elif upstream == 'baseline':
        upstream_model = dummy_upstream(input_dim)

    assert(hasattr(upstream_model, 'forward'))
    assert(hasattr(upstream_model, 'out_dim'))
    return upstream_model


def get_downstream_model(args, input_dim, output_dim, config):
    if args.dckpt == '':
        model_config = config['model'][args.downstream] if args.downstream in config['model'] else {}
    else:
        dckpt = torch.load(args.dckpt, map_location='cpu')
        model_config = {}
        if args.downstream != 'Mockingjay':
            dconfig = dckpt['Settings']['Config']
            if 'small_model' in dconfig:
                model_config = dconfig['small_model']['model']
            else:
                model_config = dconfig['model'][dckpt['Settings']['Paras'].downstream]

    configs = vars(args)
    configs.update(model_config)
    model = eval(args.downstream)(input_size=input_dim, output_size=output_dim, **configs)

    if args.dckpt != '' and args.downstream != 'Mockingjay':
        if 'SmallModel' in dckpt:
            state_dict = {'.'.join(key.split('.')[1:]): value for key, value in dckpt['SmallModel'].items()}
        else:
            state_dict = dckpt['Downstream']
        model.load_state_dict(state_dict)
    return model


########
# MAIN #
########
def main():
    
    # get config and arguments
    args, config = get_downstream_args()
    
    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    # make experiment directory
    expdir = os.path.join(f'{args.expdir}/{args.name}')
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    # get preprocessor
    preprocessor, upstream_feat_dim, downstream_feat_dim, tar_linear_dim = get_preprocessor(args, config)

    # get upstream model
    upstream_model = get_upstream_model(upstream_feat_dim, args.upstream, args.ckpt, args.dropout)
    upstream_model2 = get_upstream_model(upstream_feat_dim, args.upstream2, args.ckpt2, args.dropout2)

    # get downstream model
    downstream_inpdim = downstream_feat_dim if (args.from_rawfeature or args.from_waveform) else upstream_model.out_dim
    downstream_model = get_downstream_model(args, downstream_inpdim, tar_linear_dim, config)

    runner = Runner(args=args,
                    config=config,
                    preprocessor=preprocessor,
                    upstream=upstream_model,
                    upstream2=upstream_model2,
                    downstream=downstream_model,
                    expdir=expdir)
    runner.set_model()

    if args.test:
        runner.evaluate()
    elif args.test_gradient:
        runner.test_gradient()
    else:
        runner.train()


if __name__ == '__main__':
    main()
