"""Train the MLP surrogate (p0, G) -> (p2, p3) on dataset_1d_p2p3.csv.

Usage (from the ml/ folder):
    python src/train.py
    python src/train.py --n_hidden 32 --n_layers 3 --n_epochs 3000

Saves in --out_dir (default "checkpoints/"):
    model.pt       weights of the best model (lowest val_loss)
    scalers.json   input/output normalization + hyperparameters
                   (everything needed to use the model afterwards)
    history.json   train/validation loss curves, to diagnose overfitting
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import load_and_prepare
from model import SurrogateMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    csv_path: str = "../data/dataset_1d_p2p3.csv",
    out_dir: str = "checkpoints",
    n_hidden: int = 64,
    n_layers: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 3000,
    batch_size: int = 32,
    val_frac: float = 0.2,
    seed: int = 42,
    patience: int = 300,
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    train_ds, val_ds, x_scaler, y_scaler, df, train_idx, val_idx = load_and_prepare(
        csv_path, val_frac=val_frac, seed=seed
    )

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    model = SurrogateMLP(n_hidden=n_hidden, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        train_loss = running / len(train_ds)

        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss = loss_fn(model(xb), yb).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement in {patience} epochs)")
                break

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            print(
                f"epoch {epoch:5d}  train_loss={train_loss:.5f}  "
                f"val_loss={val_loss:.5f}  best={best_val:.5f}"
            )

    model.load_state_dict(best_state)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "scalers.json"), "w") as f:
        json.dump(
            {
                "x_scaler": x_scaler.state_dict(),
                "y_scaler": y_scaler.state_dict(),
                "features": ["p0_Pa", "G_kPa"],
                "targets_log": ["p2_Pa", "p3_Pa"],
                "n_hidden": n_hidden,
                "n_layers": n_layers,
                "seed": seed,
                "val_frac": val_frac,
            },
            f,
            indent=2,
        )
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\nBest val_loss (MSE in log-standardized space): {best_val:.5f}")
    print(f"Model saved in {out_dir}/")
    return model, x_scaler, y_scaler, history, df, train_idx, val_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="../data/dataset_1d_p2p3.csv")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--n_hidden", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=300)
    args = parser.parse_args()

    train(**vars(args))
