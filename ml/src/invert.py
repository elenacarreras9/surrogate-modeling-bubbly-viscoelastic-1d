"""Inverse problem: fix G or p0, give a target for p2 and/or p3, and search
for the value of the other variable that best reproduces it, using the
already trained MLP (no retraining needed).

Usage (from the ml/ folder, with the environment activated): give EXACTLY one
of --G or --p0 (the one you fix) and AT LEAST one of --p2 or --p3 (the target).

    python src/invert.py --G 33 --p2 5000
        -> fixes G = 33 kPa, searches the p0 that gives p2 = 5000 Pa.

    python src/invert.py --p0 25000 --p2 5000 --p3 450
        -> fixes p0 = 25000 Pa, searches the G that best reproduces
           p2 = 5000 Pa AND p3 = 450 Pa at the same time (relative
           least-squares fit, since the two targets generally cannot be
           met exactly: there are 2 targets and only 1 unknown in this mode).

The search is done within the range that the network actually saw in
training (same mechanism as predict.py, reading
../data/dataset_1d_p2p3.csv); if the best solution found falls right at the
edge of that range, a warning is printed: the requested target may not be
reachable inside the valid zone.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import minimize_scalar

from dataset import predict_physical
from evaluate import load_model
from predict import load_p0_max_from_dataset, interpolated_p0_max


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--G", type=float, default=None, help="Fixed G, in kPa -- if given, p0 is searched")
    parser.add_argument("--p0", type=float, default=None, help="Fixed p0, in Pa -- if given, G is searched")
    parser.add_argument("--p2", type=float, default=None, help="Target p2, in Pa")
    parser.add_argument("--p3", type=float, default=None, help="Target p3, in Pa")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--dataset_file", default="../data/dataset_1d_p2p3.csv")
    args = parser.parse_args()

    if (args.G is None) == (args.p0 is None):
        parser.error("Give EXACTLY one of --G or --p0 (the one to fix); the other one is searched.")
    if args.p2 is None and args.p3 is None:
        parser.error("Give at least one of --p2 or --p3 as the target.")

    model, x_scaler, y_scaler, meta = load_model(args.checkpoint_dir)
    G_vals, p0_max_vals = load_p0_max_from_dataset(args.dataset_file)

    if args.G is not None:
        free = "p0"
        lo, hi = 500.0, interpolated_p0_max(args.G, G_vals, p0_max_vals)

        def forward(x):
            p2, p3 = predict_physical(model, x_scaler, y_scaler, p0_Pa=x, G_kPa=args.G)
            return float(p2[0]), float(p3[0])
    else:
        free = "G"
        lo, hi = float(G_vals.min()), float(G_vals.max())

        def forward(x):
            p2, p3 = predict_physical(model, x_scaler, y_scaler, p0_Pa=args.p0, G_kPa=x)
            return float(p2[0]), float(p3[0])

    def error(x):
        p2_pred, p3_pred = forward(x)
        e = 0.0
        if args.p2 is not None:
            e += ((p2_pred - args.p2) / args.p2) ** 2
        if args.p3 is not None:
            e += ((p3_pred - args.p3) / args.p3) ** 2
        return e

    res = minimize_scalar(error, bounds=(lo, hi), method="bounded",
                           options={"xatol": 1e-3 * (hi - lo)})
    x_sol = res.x
    p2_sol, p3_sol = forward(x_sol)

    at_edge = min(abs(x_sol - lo), abs(x_sol - hi)) < 1e-3 * (hi - lo)

    print(f"Fixing {'G=%.4g kPa' % args.G if args.G is not None else 'p0=%.4g Pa' % args.p0}, "
          f"searching {free} in [{lo:.4g}, {hi:.4g}]...\n")
    print(f"{free} found = {x_sol:.4g}")
    print(f"  -> p2 = {p2_sol:.4g} Pa" + (f"  (target {args.p2:.4g}, error {100*(p2_sol-args.p2)/args.p2:+.2f}%)" if args.p2 is not None else ""))
    print(f"  -> p3 = {p3_sol:.4g} Pa" + (f"  (target {args.p3:.4g}, error {100*(p3_sol-args.p3)/args.p3:+.2f}%)" if args.p3 is not None else ""))

    if at_edge:
        print(f"\nWARNING: the solution falls right at the edge of the valid range of {free} "
              f"({lo:.4g} to {hi:.4g}) -- the requested target may not be reachable "
              f"inside the zone the network saw in training. Check the error above.")


if __name__ == "__main__":
    main()
