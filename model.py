import torch
import torch.nn as nn
from utility.preprocessor import OnlinePreprocessor
from transformer.nn_transformer import TRANSFORMER
from transformer.model import TransformerConfig, TransformerSpecPredictionHead


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, activation='ReLU', **kwargs):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.act = eval(f'nn.{activation}()')

    def forward(self, features, **kwargs):
        predicted = self.linear(features)
        predicted = self.act(predicted)
        return predicted, {}


class LinearResidual(nn.Module):
    def __init__(self, input_size=201, output_size=201, activation='Sigmoid', cmvn=True, eps=1e-6, **kwargs):
        super(LinearResidual, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.act = eval(f'nn.{activation}()')
        self.cmvn = cmvn
        self.eps = eps

    def forward(self, features, linears, **kwargs):
        if self.cmvn:
            features = (features - features.mean(dim=1, keepdim=True)) / (features.std(dim=1, keepdim=True) + self.eps)
        offset = self.linear(features)
        offset = self.act(offset)
        predicted = linears * offset
        return predicted, {'offset': offset}


class LSTM(nn.Module):
    def __init__(self, input_size=201, output_size=201, hidden_size=201, num_layers=3, bidirectional=False,
                 activation='ReLU', **kwargs):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.scaling_layer = nn.Sequential(
            nn.Linear(max(1, int(bidirectional) * 2) * hidden_size, output_size), eval(f'nn.{activation}()'))
        self.init_weights()
        self.bidirectional = bidirectional

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'scaling_layer.0.weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, features, **kwargs):
        predicted, _ = self.lstm(features)
        predicted = self.scaling_layer(predicted)
        return predicted, {}


class Residual(nn.Module):
    def __init__(self, input_size=201, output_size=201, hidden_size=201, num_layers=3, bidirectional=False,
                 activation='Sigmoid', cmvn=False, eps=1e-6, **kwargs):
        super(Residual, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.scaling_layer = nn.Sequential(
            nn.Linear(max(1, int(bidirectional) * 2) * hidden_size, output_size), eval(f'nn.{activation}()'))
        self.init_weights()
        self.bidirectional = bidirectional
        self.cmvn = cmvn
        self.eps = eps

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'scaling_layer.0.weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, features, linears, **kwargs):
        offset, _ = self.lstm(features)
        if self.cmvn:
            offset = (offset - offset.mean(dim=1, keepdim=True)) / (offset.std(dim=1, keepdim=True) + self.eps)
        offset = self.scaling_layer(offset)
        predicted = linears * offset
        return predicted, {'offset': offset}


class SpecHead(nn.Module):
    def __init__(self, output_dim, ckpt, activation='ReLU', random_init=False, eps=1e-6, **kwargs):
        super(SpecHead, self).__init__()
        assert ckpt != ''
        ckpt = torch.load(ckpt, map_location='cpu')
        trans_config = TransformerConfig(ckpt['Settings']['Config'])
        trans_spechead = TransformerSpecPredictionHead(trans_config, output_dim)
        trans_spechead.load_state_dict(ckpt['SpecHead'])
        
        assert trans_spechead.output.out_features == output_dim

        self.spechead = trans_spechead
        self.eps = eps

        target_config = ckpt['Settings']['Config']['online']['target']
        self.log = False if 'log' not in target_config else target_config['log']
        self.act = eval(f'nn.{activation}()')

        if random_init:
            for param in self.parameters():
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param.data)
                else:
                    nn.init.constant_(param.data, 0)

    def forward(self, features, **kwargs):
        predicted, _ = self.spechead(features)
        if self.log:
            predicted = predicted.exp()
        predicted = self.act(predicted)
        return predicted, {}


class Mockingjay(nn.Module):
    def __init__(self, dckpt, activation='ReLU', eps=1e-6, **kwargs):
        super(Mockingjay, self).__init__()
        options = {'ckpt_file'     : dckpt,
                   'load_pretrain' : 'True',
                   'no_grad'       : 'False',
                   'dropout'       : 'default',
                   'spec_aug'      : 'False',
                   'spec_aug_prev' : 'True',
                   'weighted_sum'  : 'False',
                   'select_layer'  : -1,
                   'permute_input' : 'False',
        }
        # get input_dim for specific checkpoint
        ckpt = torch.load(dckpt, map_location='cpu')
        pretrain_config = ckpt['Settings']['Config']
        preprocessor = OnlinePreprocessor(**pretrain_config['online'])
        inp_feat, tar_feat = preprocessor(feat_list=[pretrain_config['online']['input'], pretrain_config['online']['target']])

        # TRANSFORMER will automatically load parameters
        self.mockingjay = TRANSFORMER(options, inp_feat.size(-1))

        trans_config = TransformerConfig(ckpt['Settings']['Config'])
        trans_spechead = TransformerSpecPredictionHead(trans_config, tar_feat.size(-1))
        trans_spechead.load_state_dict(ckpt['SpecHead'])
        assert trans_spechead.output.out_features == tar_feat.size(-1)

        self.spechead = trans_spechead
        self.eps = eps

        target_config = ckpt['Settings']['Config']['online']['target']
        self.log = False if 'log' not in target_config else target_config['log']
        self.act = eval(f'nn.{activation}()')

    def forward(self, features, **kwargs):
        features = self.mockingjay(features)
        predicted, _ = self.spechead(features)
        if self.log:
            predicted = predicted.exp()
        predicted = self.act(predicted)
        return predicted, {}
