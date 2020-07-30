import torch
import torch.nn as nn


class PseudoDownstream(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PseudoDownstream, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, features):
        predicted = self.linear(features)
        return predicted


class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        pass
    
    def forward(self, features, linear_tar, phase_inp):
        pass