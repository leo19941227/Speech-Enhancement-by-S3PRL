import os
import yaml
import copy
import torch
import torchaudio
import random
from functools import partial
import argparse
import numpy as np
from shutil import copyfile
from torch.utils.data import DataLoader
from dataloader import OnlinePreprocessor
from transformer.nn_transformer import TRANSFORMER
from downstream.model import dummy_upstream
from runner import Runner
from model import LSTM, Linear
from dataset import PseudoDataset


def get_downstream_args():
    parser = argparse.ArgumentParser(description='Argument Parser for Downstream Tasks of the S3PLR project.')
    parser.add_argument('--name', required=True, type=str, help='Name of current experiment.')
    parser.add_argument('--dataset', required=True, choices=['dns'])

    # upstream settings
    parser.add_argument('--upstream', choices=['transformer', 'baseline'], default='baseline', help='Whether to use upstream models for speech representation or fine-tune.', required=False)
    parser.add_argument('--ckpt', default='', type=str, help='Path to upstream pre-trained checkpoint, required if using other than baseline', required=False)
    parser.add_argument('--fine_tune', action='store_true', help='Whether to fine tune the transformer model with downstream task.', required=False)
    parser.add_argument('--weighted_sum', action='store_true', help='Whether to use weighted sum on the transformer model with downstream task.', required=False)

    # Options
    parser.add_argument('--downstream', choices=['LSTM', 'Linear'], default='LSTM', help='Whether to use upstream models for speech representation or fine-tune.', required=False)
    parser.add_argument('--config', default='config/downstream.yaml', type=str, help='Path to downstream experiment config.', required=False)
    parser.add_argument('--expdir', default='result', type=str, help='Path to store experiment result, if empty then default is used.', required=False)
    parser.add_argument('--seed', default=1337, type=int, help='Random seed for reproducable results.', required=False)
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')

    # parse
    args = parser.parse_args()
    setattr(args, 'gpu', not args.cpu)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
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
    elif args.upstream == 'baseline':
        upstream_feat = config['preprocessor']['baseline']
    
    channel_inp = config['preprocessor']['input_channel']
    channel_tar = config['preprocessor']['target_channel']
    upstream_feat['channel'] = channel_inp
    
    feat_list = [
        upstream_feat,
        OnlinePreprocessor.get_feat_config('linear', channel_inp),
        OnlinePreprocessor.get_feat_config('phase', channel_inp),
        OnlinePreprocessor.get_feat_config('linear', channel_tar),
        OnlinePreprocessor.get_feat_config('phase', channel_tar),
    ]

    device = 'cpu' if args.cpu else 'cuda'
    preprocessor = OnlinePreprocessor(**pretrain_config, feat_list=feat_list).to(device=device)
    setattr(preprocessor, 'channel_inp', channel_inp)
    setattr(preprocessor, 'channel_tar', channel_tar)
    
    upstream_feat, inp_linear, inp_phase, tar_linear, tar_phase = preprocessor()
    return preprocessor, upstream_feat.size(-1), tar_linear.size(-1)


def get_upstream_model(args, input_dim):
    print('[run_downstream] - getting upstream model:', args.upstream)
    if args.upstream == 'transformer':
        options = {'ckpt_file'     : args.ckpt,
                   'load_pretrain' : 'True',
                   'no_grad'       : 'True' if not args.fine_tune else 'False',
                   'dropout'       : 'default',
                   'spec_aug'      : 'False',
                   'spec_aug_prev' : 'True',
                   'weighted_sum'  : 'True' if args.weighted_sum else 'False',
                   'select_layer'  : -1,
        }
        upstream_model = TRANSFORMER(options, input_dim)
        upstream_model.permute_input = False
    elif args.upstream == 'baseline':
        upstream_model = dummy_upstream(input_dim)

    assert(hasattr(upstream_model, 'forward'))
    assert(hasattr(upstream_model, 'out_dim'))
    return upstream_model


def get_dataloader(args, dataloader_config):
    dataset = PseudoDataset()
    dataloader = DataLoader(dataset, batch_size=dataloader_config['batch_size'])
    train_loader = dataloader
    dev_loader = copy.deepcopy(dataloader)
    test_loader = copy.deepcopy(dataloader)
    return train_loader, dev_loader, test_loader


def get_downstream_model(args, input_dim, output_dim, config):
    device = 'cpu' if args.cpu else 'cuda'
    if args.downstream == 'LSTM':
        model = LSTM(input_dim, output_dim).to(device=device) 
    elif args.downstream == 'Linear':
        model = Linear(input_dim, output_dim).to(device=device)    
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

    # make experiment directory
    expdir = os.path.join(f'result/{args.name}')
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    copyfile(args.config, os.path.join(expdir, args.config.split('/')[-1]))

    # get preprocessor
    preprocessor, upstream_feat_dim, tar_linear_dim = get_preprocessor(args, config)

    # get upstream model
    upstream_model = get_upstream_model(args, upstream_feat_dim)

    # get dataloaders
    train_loader, dev_loader, test_loader = get_dataloader(args, config['dataloader'])

    # get downstream model
    downstream_model = get_downstream_model(args, upstream_model.out_dim, tar_linear_dim, config)

    # train
    runner = Runner(args=args,
                    runner_config=config['runner'],
                    dataloader= {'train':train_loader, 'dev':dev_loader, 'test':test_loader},
                    preprocessor=preprocessor,
                    upstream=upstream_model,
                    downstream=downstream_model,
                    expdir=expdir)
    runner.set_model()
    runner.train()


if __name__ == '__main__':
    main()