"""
train.py

Trains the LSTM autoencoder on "normal" windows only, then saves the model.

Usage:
    python src/train.py
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import LSTMAutoencoder
from data_loader import load_channel, make_windows

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def train(channel_id: str, window_size: int = 100, epochs: int = 20,
          batch_size: int = 32, lr: float = 1e-3) -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    train_arr = load_channel(channel_id, split="train")
    windows = make_windows(train_arr, window_size=window_size)
    x = torch.tensor(windows, dtype=torch.float32)

    dataset = TensorDataset(x, x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMAutoencoder(n_features=x.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = loss_fn(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch + 1}/{epochs} — loss: {total_loss / len(loader):.6f}")

    torch.save(model.state_dict(), MODEL_DIR / f"{channel_id}_autoencoder.pt")
    print(f"Saved model to {MODEL_DIR / f'{channel_id}_autoencoder.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="A-1")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train(args.channel, epochs=args.epochs)
