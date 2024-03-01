import torch
from torch.nn import Embedding, Linear, Sequential, ReLU, Sigmoid, Parameter
from dataclasses import dataclass
from typing import List

@dataclass
class CINConfig:
    layers: int = 3
    hidden: int = 100

@dataclass
class DNNConfig:
    layers: int = 3
    hidden: int = 100

@dataclass
class Field:
    key: str # debug
    num_unique: int

@dataclass
class NetConfig:
    fields: List[Field]

    m: int

    dnn: DNNConfig
    cin: CINConfig

    lr: int = 0.001
    l2: int = 0.0001
    dropout: int = 0.5 # Only for DNN
    embed_dim: int = 10 # per field for all m fields

class ConcatEmbedding:
    """
    Utility class to convert raw examples to a concatenated embedding vector

    Ex input: [1, 200] (user_id, movie_id)
       output: [ ...d...  ....d... ]

    Class must know which positions correspond to which fields
    """

    # FIXME: account for batch dimension
    def __init__(self, fields: List[Field], dim):
        self.m = len(fields)
        self.dim = dim
        self.pos_to_field = {}
        self.embedding_layers = {}
        self.onehot_size = 0

        for pos, field in enumerate(fields):
            self.pos_to_field[pos] = field
            self.embedding_layers[pos] = Embedding(field.num_unique, dim) #.cuda()
            self.onehot_size += field.num_unique

    def raw_to_onehot(self, batch):
        onehot = []
        start_stop = []
        i = 0
        for pos in range(self.m):
            feature = batch[:, pos]
            num_classes = self.pos_to_field[pos].num_unique
            enc = torch.nn.functional.one_hot(feature, num_classes=num_classes)
            onehot.append(enc)
            start_stop.append((i, num_classes - 1))
            i += num_classes
        onehot = torch.cat(onehot, dim=1).float()
        return (onehot, start_stop)
    
    def raw_to_embed(self, batch):
        all_embedded = []
        for pos in range(self.m):
            feature = batch[:, pos]
            embedding = self.embedding_layers[pos](feature)
            all_embedded.append(embedding)
        all_embedded = torch.stack(all_embedded, dim=1)
        return all_embedded


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

        self.embedding_layer = ConcatEmbedding(config.fields, config.embed_dim)

        self.dnn = Sequential(
            Linear(config.m * config.embed_dim, config.dnn.hidden),
            ReLU(),
            Linear(config.dnn.hidden, config.dnn.hidden),
            ReLU(),
            Linear(config.dnn.hidden, config.dnn.hidden),
            ReLU()
        )

        self.cin_weight_0 = torch.zeros((config.cin.hidden, config.m, config.embed_dim))
        self.cin_weights = torch.zeros((config.cin.layers - 1, config.cin.hidden, config.cin.hidden, config.embed_dim))

        # output unit
        self.W_linear = torch.zeros((self.embedding_layer.onehot_size, 1))
        self.W_dnn = torch.zeros((config.dnn.hidden, 1))
        self.W_cin = torch.zeros((config.cin.layers * config.cin.hidden, 1))
        self.output_bias = Parameter(torch.zeros(1))
        self.output_relu = ReLU()        

        torch.nn.init.xavier_uniform_(self.cin_weight_0)
        torch.nn.init.xavier_uniform_(self.cin_weights)
        torch.nn.init.xavier_uniform_(self.W_linear)
        torch.nn.init.xavier_uniform_(self.W_dnn)
        torch.nn.init.xavier_uniform_(self.W_cin)

        self.sigmoid = Sigmoid()
        self.config = config

    def forward(self, x):
        batch_size = 12

        m, d = self.config.m, self.config.embed_dim
        T, H_k = self.config.cin.layers, self.config.cin.hidden

        X_0 = self.embedding_layer.raw_to_embed(x)
        X_k_prev = X_0
        p_plus = torch.zeros((T, batch_size, H_k))

        for k in range(T):
            X_k = torch.zeros((batch_size, H_k, d))
            for h in range(H_k):
                # Eq. 6
                for i in range(H_k if k != 0 else m):
                    for j in range(m):
                        hadamard = X_k_prev[:, i] *  X_0[:, j]
                        if k == 0:
                            out = self.cin_weight_0[h, i, j] * hadamard
                        else:
                            out = self.cin_weights[k - 1, h, i, j] * hadamard
                        X_k[:, h] = out
            p_k = X_k.sum(axis=2)
            p_plus[k] = p_k
            X_k_prev = X_k

        p_plus = p_plus.transpose(0, 1)
        p_plus = p_plus.flatten(1)

        x_dnn = self.dnn(X_0.flatten(1))
        a, pos_a = self.embedding_layer.raw_to_onehot(x)
        output = (x_dnn @ self.W_dnn) + (p_plus @ self.W_cin) + (a @ self.W_linear) + self.output_bias
        output = self.sigmoid(output)
        return output