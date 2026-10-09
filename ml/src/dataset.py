"""Loading of dataset_1d_p2p3.csv, log transform of the targets,
standardization and train/validation split.

Everything needed to reproduce the preprocessing exactly in evaluate.py (or
in any new notebook) lives here, so that the normalization logic is not
duplicated in several places.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURES = ["p0_Pa", "G_kPa"]
TARGETS_RAW = ["p2_Pa", "p3_Pa"]


class Scaler:
    """Standardization (mean 0, standard deviation 1). It is fitted ONLY on
    the training set (fit), and that same transformation is then applied to
    the validation set and to any new data, so that no information from the
    validation set leaks into training.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Scaler":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_

    def state_dict(self) -> dict:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict) -> "Scaler":
        s = cls()
        s.mean_ = np.array(state["mean"])
        s.std_ = np.array(state["std"])
        return s


class HarmonicsDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_and_prepare(csv_path: str, val_frac: float = 0.2, seed: int = 42):
    """Read the CSV, apply the log to p2/p3, split train/validation (random,
    with a fixed seed) and fit the Scalers ONLY on the training set.

    Returns: train_ds, val_ds, x_scaler, y_scaler, df (the full dataframe,
    useful for evaluate.py), train_idx, val_idx (indices into df).
    """
    df = pd.read_csv(csv_path)

    X_raw = df[FEATURES].to_numpy(dtype=np.float64)
    Y_log = np.log(df[TARGETS_RAW].to_numpy(dtype=np.float64))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_val = int(round(len(df) * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_scaler = Scaler().fit(X_raw[train_idx])
    y_scaler = Scaler().fit(Y_log[train_idx])

    X_train = x_scaler.transform(X_raw[train_idx])
    X_val = x_scaler.transform(X_raw[val_idx])
    Y_train = y_scaler.transform(Y_log[train_idx])
    Y_val = y_scaler.transform(Y_log[val_idx])

    train_ds = HarmonicsDataset(X_train, Y_train)
    val_ds = HarmonicsDataset(X_val, Y_val)

    return train_ds, val_ds, x_scaler, y_scaler, df, train_idx, val_idx


def predict_physical(model, x_scaler: Scaler, y_scaler: Scaler, p0_Pa, G_kPa):
    """Given a trained model and its scalers, predict p2 and p3 in Pa
    (undoing the standardization and the log) for arrays of p0_Pa and G_kPa.
    Used both in evaluate.py and to use the trained surrogate directly.
    """
    p0_Pa = np.atleast_1d(np.asarray(p0_Pa, dtype=np.float64))
    G_kPa = np.atleast_1d(np.asarray(G_kPa, dtype=np.float64))
    X = np.column_stack([p0_Pa, G_kPa])
    X_norm = x_scaler.transform(X)

    model.eval()
    with torch.no_grad():
        Y_norm = model(torch.tensor(X_norm, dtype=torch.float32)).numpy()

    Y_log = y_scaler.inverse_transform(Y_norm)
    Y = np.exp(Y_log)
    return Y[:, 0], Y[:, 1]  # p2_Pa, p3_Pa
