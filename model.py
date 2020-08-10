import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features, **kwargs):
        predicted = self.linear(features)
        return predicted


class LSTM(nn.Module):
    def __init__(self, input_size=201, output_size=201, hidden_size=201, num_layers=3, bidirectional=False, **kwargs):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.scaling_layer = nn.Sequential(
            nn.Linear(hidden_size, output_size), nn.ReLU())
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
        return predicted


class Residual(nn.Module):
    def __init__(self, input_size=201, output_size=201, hidden_size=201, num_layers=3, bidirectional=False,
                 cmvn=False, eps=1e-6, **kwargs):
        super(Residual, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.scaling_layer = nn.Sequential(
            nn.Linear(hidden_size, output_size), nn.ReLU())
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
        return predicted
