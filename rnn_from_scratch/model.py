import torch
from torch import nn

import numpy as np

class RNNModel(nn.Module):
    def __init__(self, input_feature_num, hidden_feature_num=200, n_layers=1, out_dim=1):
        super(RNNModel, self).__init__()
        
        self.input_feature_num = input_feature_num
        self.hidden_feature_num = hidden_feature_num
        self.n_layers = n_layers
        self.out_dim = out_dim
        
        self.rnn = nn.RNN(input_feature_num, hidden_feature_num, n_layers)
        self.fc = nn.Linear(hidden_feature_num, out_dim)
        
    def forward(self, x):
        batch_size = x.size(1)
        hidden = self.init_hidden(batch_size)
        hidden = hidden.to(device)
        out, hidden = self.rnn(x, hidden)
        hidden = hidden.contiguous().view(-1, self.hidden_feature_num)
        out = self.fc(hidden)
        return out
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_feature_num)
        return hidden