from tqdm import tqdm 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import src.xdeepfm as xdeepfm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.xdeepfm import Field

def load_df():
    # Movie-lens specifically for now
    df = pd.read_csv("./data/ml-100k/u.data", header=None, delim_whitespace=True, encoding='utf-8')
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    return df

def df_post_process(df, batch_size = 2048):
    X = df[['user_id', 'item_id']].to_numpy()
    y = df['rating'].to_numpy()
    y = (y >= 4).astype(int)

    num_users = np.max(df['user_id']).item() + 1
    num_items = np.max(df['item_id']).item() + 1
    fields = [num_users, num_items]

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5)
    train_dl = DataLoader(
        TensorDataset(
            torch.tensor(train_X),
            torch.tensor(train_y).float(),
        ),
        batch_size=batch_size,
        drop_last=True
    )
    test_dl = DataLoader(
        TensorDataset(
            torch.tensor(test_X),
            torch.tensor(test_y).float(),
        ),
        batch_size=batch_size,
        drop_last=True
    )

    return fields, train_dl, test_dl

def evaluate_net(net, loss_fn, test_dl):
    net.cuda()
    net.eval()
    total_loss = 0.0
    predictions = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_dl, desc="Evaluating"):
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            batch_y = batch_y.flatten()
            predicted_ratings = net(batch_x)
            loss = loss_fn(predicted_ratings, batch_y)
            total_loss += loss.item()
            predictions.extend(predicted_ratings.cpu().numpy())
    avg_loss = total_loss / len(test_dl)
    print(f"Average Loss: {avg_loss:.4f}")
    return avg_loss

def fit(net, loss_fn, train_dl, test_dl, epochs=10, lr=0.001):
    net.train()
    optim = torch.optim.Adam(net.parameters(), lr)
    J = []  
    for e in tqdm(range(epochs)):
        for batch_x, batch_y in tqdm(train_dl):
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            optim.zero_grad()
            batch_y = batch_y.flatten()
            out = net(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            optim.step()
            J.append(loss)
        evaluate_net(net, loss_fn, test_dl)
        net.train()
        print(f"Epoch {e + 1}/{epochs}: Loss {loss:.2f}")
        torch.save(
            {
                "epoch": e + 1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "loss": loss,
            },
            net.pt_file,
        )
    return J