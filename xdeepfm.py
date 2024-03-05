import torch
from torch.nn import Embedding, Linear, Sequential, ReLU, Sigmoid, Parameter, Conv1d, BatchNorm1d, Dropout
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class CINConfig:
    layers: int = 3
    hidden: int = 100

@dataclass
class DNNConfig:
    layers: int = 3
    hidden: int = 100

@dataclass
class NetConfig:
    fields: List[int] # [# of unique]

    m: int

    dnn: DNNConfig
    cin: CINConfig

    lr: int = 0.001
    l2: int = 0.0001
    dropout: int = 0 # Only for DNN
    embed_dim: int = 10 # per field for all m fields

    batch_size: int = 2048

class DirectLinear(torch.nn.Module):
    def __init__(self, fields):
        super(DirectLinear, self).__init__()

        self.embedding = Embedding(sum(fields), 1).cuda()
        self.offsets = torch.tensor((0, *np.cumsum(fields)[:-1])).cuda()
        self.bias = Parameter(torch.zeros(1))
    
    def forward(self, x):
        x_offsets = x + self.offsets
        out = torch.sum(self.embedding(x_offsets).flatten(1), dim=1) + self.bias
        return out.unsqueeze(1)
    

class FlatEmbedding(torch.nn.Module):
    def __init__(self, fields, dim):
        super(FlatEmbedding, self).__init__()

        self.embedding = Embedding(sum(fields), dim).cuda()
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        self.offsets = torch.tensor((0, *np.cumsum(fields)[:-1])).cuda()
    
    def forward(self, x):
        x_offsets = x + self.offsets
        return self.embedding(x_offsets).flatten(1, 2)

class DNN(torch.nn.Module):
    def __init__(self, config):
        super(DNN, self).__init__()

        self.dnn = Sequential(
            Linear(config.m * config.embed_dim, config.dnn.hidden),
            BatchNorm1d(config.dnn.hidden),
            ReLU(),
            Dropout(config.dropout),
            Linear(config.dnn.hidden, config.dnn.hidden),
            BatchNorm1d(config.dnn.hidden),
            ReLU(),
            Dropout(config.dropout),
            Linear(config.dnn.hidden, config.dnn.hidden),
            BatchNorm1d(config.dnn.hidden),
            ReLU(),
            Dropout(config.dropout),
            Linear(config.dnn.hidden, 1),
        )

    def forward(self, e):
        return self.dnn(e)

class CIN(torch.nn.Module):
    def __init__(self, config: NetConfig):
        super(CIN, self).__init__()

        self.config = config
        self.convs = torch.nn.ModuleList([
            Conv1d(config.m * config.m, config.cin.hidden, 1),
            Conv1d(config.cin.hidden * config.m, config.cin.hidden, 1),
            Conv1d(config.cin.hidden * config.m, config.cin.hidden, 1)
        ])
        self.relu = ReLU()
        self.fc = Linear(config.cin.hidden * config.cin.layers, 1)

    def forward(self, e):
        # e: (B, m, d)
        e = e.unflatten(1, (self.config.m, self.config.embed_dim))
        # transform it to (B, m, 1, d)
        B = self.config.batch_size
        H_k_size = self.config.cin.hidden # for now H_k will be all the same

        X_0 = e.unsqueeze(2)
        H = e

        T = self.config.cin.layers

        p_plus = torch.zeros((T, B, H_k_size)).cuda()

        for i in range(T):
            outer = X_0 * H.unsqueeze(1) # (B, m, H_k or m, d)
            outer = outer.flatten(1, 2) # (B, m * H_k, d)
            feature_map = self.relu(self.convs[i](outer))
            H = feature_map
            p_i = feature_map.sum(axis = 2)
            p_plus[i] = p_i

        p_plus = p_plus.transpose(0, 1).flatten(1)
        return self.fc(p_plus)

class xDeepFM(torch.nn.Module):
    def __init__(self, config: NetConfig):
        """
        Implementing the xDeepFM architecture consisting of three main parts:
        - Linear layer on raw input
        - CIN to learn high-order bounded feature interactions
        - DNN to learn bit-wise and implicitly
            - 3 layers
            - 100 units
            - RELU

        Embedding layer at the beginning
        """
        super(xDeepFM, self).__init__()

        self.lin = DirectLinear(config.fields).cuda()
        self.embed = FlatEmbedding(config.fields, config.embed_dim).cuda()
        self.dnn = DNN(config).cuda()
        self.cin = CIN(config).cuda()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        e = self.embed(x)
        cin = self.cin(e)
        dnn = self.dnn(e)
        lin = self.lin(x)
        output = (dnn + cin + lin).squeeze(1)
        output = self.sigmoid(output)
        return output