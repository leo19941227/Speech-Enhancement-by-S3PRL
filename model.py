import torch
import torch.nn as nn


class PseudoDownstream(nn.Module):
    def __init__(self, input_dim):
        super(PseudoDownstream, self).__init__()
        self.linear = nn.Linear(input_dim, 10)
    
    def forward(self, features, linear_tar, phase_inp):
        tmp = self.linear(features)
        return tmp.sum(), 0