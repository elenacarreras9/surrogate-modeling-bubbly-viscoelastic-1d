"""Quick command-line prediction with the trained MLP, without opening an
interactive Python session.

Usage (from the ml/ folder, with the environment activated):

    python src/predict.py --G 33
        -> prints the full curve p2(p0), p3(p0) for G = 33 kPa, from p0_min
           up to the highest p0 that the network actually saw in training
           for that G.

    python src/predict.py --G 33 --p0 25000
        -> prints p2, p3 for that specific point (p0, G).

G does not have to be one of the values used in training (0, 5, 10, ...,
100 kPa) -- the model interpolates.

Where the p0 limit comes from: it is read directly from
../data/dataset_1d_p2p3.csv (the same CSV produced by generate_dataset_1d.m
and used by train.py/evaluate.py). For each G, the highest p0 in that column
is by definition the highest p0 the network saw in training for that G (it
already includes the safety_margin used when generating it). No other file
is needed.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from dataset import predict_physical
from evaluate import load_model


def load_p0_max_from_dataset(path: str):
    """Read dataset_1d_p2p3.csv and return (sorted G_kPa, p0_max) -- the
    highest p0 present in the CSV for each G, which is the real limit seen by
    the network during training."""
    max_p0_per_G = defaultdict(float)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            G = float(row["G_kPa"])
            p0 = float(row["p0_Pa"])
            if p0 > max_p0_per_G[G]:
                max_p0_per_G[G] = p0
    if not max_p0_per_G:
        raise ValueError(f"{path} is empty or does not have the expected format.")
    G_vals = np.array(sorted(max_p0_per_G.keys()))
    p0_max_vals = np.array([max_p0_per_G[g] for g in G_vals])
    return G_vals, p0_max_vals


def interpolated_p0_max(G_kPa: float, G_vals: np.ndarray, p0_max_vals: np.ndarray) -> float:
    if G_kPa < G_vals.min() or G_kPa > G_vals.max():
        print(f"WARNING: G={G_kPa:g} kPa is outside the range of the dataset "
              f"({G_vals.min():g}-{G_vals.max():g} kPa) -- p0_max here is an "
              f"extrapolation, with fewer guarantees.")
    return float(np.interp(G_kPa, G_vals, p0_max_vals))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--G", type=float, required=True, help="Shear modulus G, in kPa")
    parser.add_argument("--p0", type=float, default=None,
                         help="Source amplitude p0, in Pa (optional; if omitted, "
                              "the full curve is printed)")
    parser.add_argument("--n", type=int, default=10,
                         help="Number of curve points if --p0 is not given (default 10)")
    parser.add_argument("--p0_min", type=float, default=500,
                         help="Minimum p0 of the curve, in Pa (default 500)")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--dataset_file", default="../data/dataset_1d_p2p3.csv",
                         help="The same CSV used by train.py/evaluate.py (default "
                              "../data/dataset_1d_p2p3.csv)")
    args = parser.parse_args()

    model, x_scaler, y_scaler, meta = load_model(args.checkpoint_dir)
    G_vals, p0_max_vals = load_p0_max_from_dataset(args.dataset_file)
    p0_max = interpolated_p0_max(args.G, G_vals, p0_max_vals)

    if args.p0 is not None:
        if args.p0 > p0_max:
            print(f"WARNING: p0={args.p0:g} Pa exceeds the highest p0 the network saw in "
                  f"training for G={args.G:g} kPa ({p0_max:.0f} Pa) -- the "
                  f"prediction is an extrapolation, with no physical guarantee.")
        p2, p3 = predict_physical(model, x_scaler, y_scaler, p0_Pa=args.p0, G_kPa=args.G)
        # predict_physical always returns arrays (so that arrays of p0/G can be
        # passed as well); for a single point we take the only element with [0]
        # instead of float(), which fails in recent numpy for arrays with ndim>0.
        print(f"G={args.G:g} kPa, p0={args.p0:g} Pa  ->  p2={p2[0]:.4g} Pa, p3={p3[0]:.4g} Pa")
    else:
        p0_array = np.linspace(args.p0_min, p0_max, args.n)
        p2_array, p3_array = predict_physical(
            model, x_scaler, y_scaler, p0_array, np.full_like(p0_array, args.G))
        print(f"G={args.G:g} kPa  (highest p0 seen in training, interpolated = {p0_max:.0f} Pa)\n")
        print(f"{'p0 (Pa)':>10}  {'p2 (Pa)':>12}  {'p3 (Pa)':>12}")
        for p0_i, p2_i, p3_i in zip(p0_array, p2_array, p3_array):
            print(f"{p0_i:10.0f}  {p2_i:12.4g}  {p3_i:12.4g}")


if __name__ == "__main__":
    main()
