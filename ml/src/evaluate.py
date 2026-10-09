"""Evaluate the trained surrogate: metrics in physical units (Pa, not in the
log-standardized space used for training) and check plots (p2, p3 versus
p0, one curve per G) -- dots = simulation, line = network prediction.

Usage (from the ml/ folder):
    python src/evaluate.py
    python src/evaluate.py --checkpoint_dir checkpoints --csv_path ../data/dataset_1d_p2p3.csv
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import FEATURES, TARGETS_RAW, Scaler, load_and_prepare, predict_physical
from model import SurrogateMLP

# One colour per medium (G): blue, purple, orange, green, ...
COLORS = ["#0072B2", "#7E2F8E", "#D95319", "#2E8B57", "#CC79A7", "#8A6D1E"]


def load_model(checkpoint_dir: str):
    with open(os.path.join(checkpoint_dir, "scalers.json")) as f:
        meta = json.load(f)
    model = SurrogateMLP(n_hidden=meta["n_hidden"], n_layers=meta["n_layers"])
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))
    model.eval()
    x_scaler = Scaler.from_state_dict(meta["x_scaler"])
    y_scaler = Scaler.from_state_dict(meta["y_scaler"])
    return model, x_scaler, y_scaler, meta


def relative_errors(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Mean, median and maximum relative error, and R^2 -- all in physical
    scale (Pa), not in log. The relative error is what matters here, since
    p2 and p3 span several orders of magnitude."""
    rel_err = np.abs(y_pred - y_true) / np.abs(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {
        "mape_%": float(np.mean(rel_err) * 100),
        "median_error_%": float(np.median(rel_err) * 100),
        "max_error_%": float(np.max(rel_err) * 100),
        "r2": float(r2),
    }


def evaluate(checkpoint_dir: str = "checkpoints", csv_path: str = "../data/dataset_1d_p2p3.csv"):
    model, x_scaler, y_scaler, meta = load_model(checkpoint_dir)

    # Rebuild EXACTLY the same train/validation split that train.py used
    # (same seed), to measure the error only on the validation set -- the
    # training set says nothing about the ability to generalize.
    _, _, _, _, df, train_idx, val_idx = load_and_prepare(
        csv_path, val_frac=meta["val_frac"], seed=meta["seed"]
    )

    df_val = df.iloc[val_idx]
    p2_pred, p3_pred = predict_physical(
        model, x_scaler, y_scaler, df_val["p0_Pa"].to_numpy(), df_val["G_kPa"].to_numpy()
    )

    err_p2 = relative_errors(df_val["p2_Pa"].to_numpy(), p2_pred)
    err_p3 = relative_errors(df_val["p3_Pa"].to_numpy(), p3_pred)

    print(f"Validation ({len(df_val)} points, not seen during training):\n")
    print(f"  p2:  MAPE={err_p2['mape_%']:.2f}%  median={err_p2['median_error_%']:.2f}%  "
          f"max={err_p2['max_error_%']:.2f}%  R2={err_p2['r2']:.4f}")
    print(f"  p3:  MAPE={err_p3['mape_%']:.2f}%  median={err_p3['median_error_%']:.2f}%  "
          f"max={err_p3['max_error_%']:.2f}%  R2={err_p3['r2']:.4f}")

    with open(os.path.join(checkpoint_dir, "eval_metrics.json"), "w") as f:
        json.dump({"p2": err_p2, "p3": err_p3, "n_val": len(df_val)}, f, indent=2)

    _plot_curves(model, x_scaler, y_scaler, df, df_val, checkpoint_dir)

    return err_p2, err_p3


def _plot_curves(model, x_scaler, y_scaler, df, df_val, out_dir):
    """One figure: p2 and p3 versus p0, one curve per G, with the network
    prediction superimposed on ALL simulated points (training + validation).
    Validation points are marked with a ring, to see whether the network
    interpolates well where it did not see data."""
    G_values = sorted(df["G_kPa"].unique())
    # if there are many G values, show a readable subset instead of a pile of curves
    if len(G_values) > 6:
        idx_pick = np.linspace(0, len(G_values) - 1, 6).round().astype(int)
        G_plot = [G_values[i] for i in sorted(set(idx_pick))]
    else:
        G_plot = G_values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    val_set = set(df_val.index)

    for i, G in enumerate(G_plot):
        color = COLORS[i % len(COLORS)]
        sub = df[df["G_kPa"] == G].sort_values("p0_Pa")

        p0_line = np.linspace(sub["p0_Pa"].min(), sub["p0_Pa"].max(), 200)
        p2_line, p3_line = predict_physical(model, x_scaler, y_scaler, p0_line, np.full_like(p0_line, G))

        is_val = sub.index.isin(val_set)

        axes[0].plot(p0_line / 1e3, p2_line, "-", color=color, linewidth=1.5, label=f"G={G:g} kPa")
        axes[0].scatter(sub["p0_Pa"][~is_val] / 1e3, sub["p2_Pa"][~is_val], color=color, s=18, marker="o")
        axes[0].scatter(sub["p0_Pa"][is_val] / 1e3, sub["p2_Pa"][is_val], facecolors="none",
                         edgecolors=color, s=40, marker="o", linewidths=1.3)

        axes[1].plot(p0_line / 1e3, p3_line, "-", color=color, linewidth=1.5, label=f"G={G:g} kPa")
        axes[1].scatter(sub["p0_Pa"][~is_val] / 1e3, sub["p3_Pa"][~is_val], color=color, s=18, marker="o")
        axes[1].scatter(sub["p0_Pa"][is_val] / 1e3, sub["p3_Pa"][is_val], facecolors="none",
                         edgecolors=color, s=40, marker="o", linewidths=1.3)

    axes[0].set_xlabel("Source amplitude $p_0$ (kPa)")
    axes[0].set_ylabel("Second harmonic amplitude $p_2$ (Pa)")
    axes[1].set_xlabel("Source amplitude $p_0$ (kPa)")
    axes[1].set_ylabel("Third harmonic amplitude $p_3$ (Pa)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Filled dots = training · Ringed dots = validation (unseen) · Line = MLP")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "prediction_vs_simulation.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved in {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--csv_path", default="../data/dataset_1d_p2p3.csv")
    args = parser.parse_args()
    evaluate(args.checkpoint_dir, args.csv_path)
