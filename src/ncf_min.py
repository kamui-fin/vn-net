import torch
import torch.nn as nn
import os
from pathlib import Path

DATA_DIR = Path("./data")

# Benchmark
class MatrixFactor(nn.Module):
    def __init__(self, num_users, num_items, n_factors=16):
        super().__init__()
        self.user_factor = nn.Embedding(num_users + 1, n_factors)
        self.item_factor = nn.Embedding(num_items + 1, n_factors)
        self.sigmoid = nn.Sigmoid()

        self.pt_file = DATA_DIR / f"vnmf_{self.name}.pt"
        self.load_if_exists()

    def forward(self, u, i):
        user_emb = self.user_factor(u)
        item_emb = self.item_factor(i)

        pred = torch.sum(user_emb * item_emb, dim=1)
        pred = self.sigmoid(pred)
        return pred

    def load_if_exists(self):
        if os.path.isfile(self.pt_file):
            checkpoint = torch.load(self.pt_file)
            self.load_state_dict(checkpoint["model_state_dict"])