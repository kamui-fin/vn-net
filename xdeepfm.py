import torch
from torch.nn import Embedding, Linear, Sequential, ReLU, Sigmoid, Parameter, Conv1d
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
    dropout: int = 0.5 # Only for DNN
    embed_dim: int = 10 # per field for all m fields

    batch_size: int = 2048

class DirectLinear(torch.nn.Module):
    def __init__(self, fields):
        super(DirectLinear, self).__init__()

        self.embedding = Embedding(sum(fields), 1).cuda()
        self.offsets = torch.tensor((0, *np.cumsum(fields)[:-1])).cuda()
    
    def forward(self, x):
        x_offsets = x + self.offsets
        return self.embedding(x_offsets).flatten(1)
    

class FlatEmbedding(torch.nn.Module):
    def __init__(self, fields, dim):
        super(FlatEmbedding, self).__init__()

        self.embedding = Embedding(sum(fields), dim).cuda()
        self.offsets = torch.tensor((0, *np.cumsum(fields)[:-1])).cuda()
    
    def forward(self, x):
        x_offsets = x + self.offsets
        return self.embedding(x_offsets).flatten(1, 2)

class DNN(torch.nn.Module):
    def __init__(self, config):
        super(DNN, self).__init__()

        self.dnn = Sequential(
            Linear(config.m * config.embed_dim, config.dnn.hidden),
            ReLU(),
            Linear(config.dnn.hidden, config.dnn.hidden),
            ReLU(),
            Linear(config.dnn.hidden, config.dnn.hidden),
            ReLU()
        )

    def forward(self, e):
        return self.dnn(e)

class CIN(torch.nn.Module):
    def __init__(self, config: NetConfig):
        super(CIN, self).__init__()

        self.config = config
        # Input channels: H_k * m
        # Output channels: H_k
        self.first_conv = Conv1d(config.m * config.m, config.cin.hidden, 1)
        self.conv = Conv1d(config.cin.hidden * config.m, config.cin.hidden, 1)
        self.relu = ReLU()

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
            conv = self.conv if i > 0 else self.first_conv
            outer = X_0 * H.unsqueeze(1) # (B, m, H_k or m, d)
            outer = outer.flatten(1, 2) # (B, m * H_k, d)
            feature_map = self.relu(conv(outer))
            p_i = feature_map.sum(axis = 2)
            p_plus[i] = p_i
            H = feature_map

        p_plus = p_plus.transpose(0, 1).flatten(1)
        return p_plus


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

        # output unit
        self.W_dnn = torch.zeros((config.dnn.hidden, 1)).cuda()
        self.W_cin = torch.zeros((config.cin.layers * config.cin.hidden, 1)).cuda()
        self.W_lin = torch.zeros((config.m, 1)).cuda()
        self.output_bias = Parameter(torch.zeros(1)).cuda()

        torch.nn.init.xavier_uniform_(self.W_dnn)
        torch.nn.init.xavier_uniform_(self.W_cin)
        torch.nn.init.xavier_uniform_(self.W_lin)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        e = self.embed(x)
        cin = self.cin(e)
        dnn = self.dnn(e)
        lin = self.lin(x)
        output = (dnn @ self.W_dnn) + (cin @ self.W_cin) + (lin @ self.W_lin) + self.output_bias
        output = self.sigmoid(output)
        return output