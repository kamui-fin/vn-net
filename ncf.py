from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os

def get_data(df, batch_size = 1024):
    X_user = df["user_id"].to_numpy()
    X_vn = df["vn_id"].to_numpy()
    y = df["vote"].to_numpy()

    num_users = np.max(X_user).item() + 1
    num_books = np.max(X_vn).item() + 1

    (
        train_X_user,
        test_X_user,
        train_X_book,
        test_X_book,
        train_y,
        test_y,
    ) = train_test_split(X_user, X_vn, y, test_size=0.2)

    train_dl = DataLoader(
        TensorDataset(
            torch.from_numpy(train_X_user),
            torch.from_numpy(train_X_book),
            torch.from_numpy(train_y).float(),
        ),
        batch_size=batch_size,
    )
    test_dl = DataLoader(
        TensorDataset(
            torch.from_numpy(test_X_user),
            torch.from_numpy(test_X_book),
            torch.from_numpy(test_y).float(),
        ),
        batch_size=batch_size,
    )


    return train_dl, test_dl, num_users, num_books

def evaluate_net(net, test_dl, loss_fn):
    net.eval()

    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for users, books, ratings in tqdm(test_dl, desc="Evaluating"):
            users, books, ratings = users.cuda(), books.cuda(), ratings.cuda()
            predicted_ratings = net(users, books)

            loss = loss_fn(predicted_ratings, ratings)
            total_loss += loss.item()
            predictions.extend(predicted_ratings.cpu().numpy())
            targets.extend(ratings.cpu().numpy())

    avg_loss = total_loss / len(test_dl)
    print(f"Average Loss: {avg_loss:.4f}")
    return avg_loss


def fit(net, train_dl, test_dl, loss_fn, epochs=10, lr=0.001):
    optim = torch.optim.Adam(net.parameters(), lr)

    net.train()
    net.cuda()

    J = []
    for e in tqdm(range(epochs)):
        for batch_u, batch_i, batch_y in tqdm(train_dl):
            batch_u, batch_i, batch_y = batch_u.cuda(), batch_i.cuda(), batch_y.cuda()
            optim.zero_grad()
            out = net(batch_u, batch_i)
            loss = loss_fn(out, batch_y)
            loss.backward()
            optim.step()
            J.append(loss)

        print(f"Epoch {e + 1}/{epochs}: Loss {loss:.2f}")
        evaluate_net(net, test_dl, loss_fn)

        torch.save(
            {
                "epoch": e + 1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "loss": loss,
            },
            f"vndb_{net.name}.pt",
        )

    return J

class MatrixFactor(nn.Module):
    def __init__(self, name, num_users, num_books, n_factors=16):
        super().__init__()
        self.user_factor = nn.Embedding(num_users + 1, n_factors)
        self.book_factor = nn.Embedding(num_books + 1, n_factors)
        self.sigmoid = nn.Sigmoid()

        self.name = name
        self.load_if_exists()

    def forward(self, u, i):
        user_emb = self.user_factor(u)
        book_emb = self.book_factor(i)

        pred = torch.sum(user_emb * book_emb, dim=1)
        pred = self.sigmoid(pred)
        return pred

    def load_if_exists(self):
        if os.path.isfile(f"vndb_{self.name}.pt"):
            checkpoint = torch.load(f"vndb_{self.name}.pt")
            self.load_state_dict(checkpoint["model_state_dict"])
