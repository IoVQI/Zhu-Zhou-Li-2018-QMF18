#!/usr/bin/env python3
"""Compute mass-weighted tidal deformability Lambda_tilde for a binary system.

Formula: Flanagan & Hinderer (2008), PRD 77, 021502 — Eq.(9)
         Lambda_tilde = 16/13 * sum(weight_i * Lambda_i) / (M1+M2)^5

Usage:
    python compute_tilde_lambda.py output/2026-07-01_4L_scan/qmf18_L40
    python compute_tilde_lambda.py                              # auto-detect L=40
"""
import os, sys, glob as globmod
import numpy as np
from scipy.interpolate import PchipInterpolator

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Paper values (Zhu+2018 Table 4): M_chirp = 1.188 M_sun, q = 0.7-1
M_CHIRP_DEFAULT = 1.188
Q_RANGE = (0.7, 1.0)
PAPER_LAMBDA_TILDE = (381.4, 388.4)


def compute_tilde_lambda(mr_csv_path, M_chirp=M_CHIRP_DEFAULT, n_q=7):
    """Compute Lambda_tilde(q) from an M-R-Lambda CSV.

    Parameters
    ----------
    mr_csv_path : str
        Path to mr_curve.csv (columns: rho_c_fm3, M_Msun, R_km, Lambda, k2, C).
    M_chirp : float
        Chirp mass in M_sun.
    n_q : int
        Number of mass ratios to sample in [Q_RANGE[0], Q_RANGE[1]].

    Returns
    -------
    list of (q, M1, M2, Lambda1, Lambda2, Lambda_tilde)
    """
    # Load M-R-Lambda data
    data = np.loadtxt(mr_csv_path, delimiter=',', skiprows=1)
    M_arr   = data[:, 1]   # M_Msun
    Lamb_arr = data[:, 3]  # Lambda

    # Build interpolator: Lambda(M), monotonic decreasing → PCHIP
    Lambda_of_M = PchipInterpolator(M_arr, Lamb_arr)

    # Mass ratio range
    q_vals = np.linspace(Q_RANGE[0], Q_RANGE[1], n_q)

    results = []
    for q in q_vals:
        # M1 >= M2, q = M2/M1
        f_q = (1 + q) ** 0.2 / q ** 0.6
        M1 = M_chirp * f_q
        M2 = q * M1

        # Clamp to available M range
        M_min, M_max = M_arr[0], M_arr[-1]
        if M1 < M_min or M1 > M_max or M2 < M_min or M2 > M_max:
            continue

        L1 = float(Lambda_of_M(M1))
        L2 = float(Lambda_of_M(M2))

        # Flanagan & Hinderer (2008) Eq.(9)
        numerator = (M1 + 12 * M2) * M1**4 * L1 + (M2 + 12 * M1) * M2**4 * L2
        denominator = (M1 + M2) ** 5
        Lt = (16.0 / 13.0) * numerator / denominator

        results.append((q, M1, M2, L1, L2, Lt))

    return results


def main():
    # Find mr_curve.csv
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target and os.path.isfile(target):
        mr_path = target
    elif target and os.path.isdir(target):
        mr_path = os.path.join(target, 'mr_curve.csv')
    else:
        # Auto-detect L=40 in latest run
        output_root = os.path.join(_THIS_DIR, 'output')
        runs = sorted(
            [d for d in os.listdir(output_root)
             if os.path.isdir(os.path.join(output_root, d)) and d not in ('archive',)],
            reverse=True,
        )
        if not runs:
            print("ERROR: no runs found", file=sys.stderr); sys.exit(1)
        mr_path = os.path.join(output_root, runs[0], 'qmf18_L40', 'mr_curve.csv')

    if not os.path.exists(mr_path):
        print(f"ERROR: {mr_path} not found", file=sys.stderr); sys.exit(1)

    print(f"Data: {mr_path}")
    print(f"M_chirp = {M_CHIRP_DEFAULT} M_sun, q = {Q_RANGE[0]}-{Q_RANGE[1]}")
    print()

    results = compute_tilde_lambda(mr_path)

    # Table
    print(f"{'q':>6} {'M1':>8} {'M2':>8} {'Lambda1':>9} {'Lambda2':>9} {'Lambda_tilde':>12}")
    print("-" * 56)
    for q, M1, M2, L1, L2, Lt in results:
        print(f"{q:6.3f} {M1:8.4f} {M2:8.4f} {L1:9.1f} {L2:9.1f} {Lt:12.1f}")

    # Summary
    min_Lt = min(r[5] for r in results)
    max_Lt = max(r[5] for r in results)
    print(f"\nLambda_tilde range: {min_Lt:.1f} — {max_Lt:.1f}")
    print(f"Paper Table 4:      {PAPER_LAMBDA_TILDE[0]:.1f} — {PAPER_LAMBDA_TILDE[1]:.1f}")
    delta = (max_Lt - PAPER_LAMBDA_TILDE[1]) / PAPER_LAMBDA_TILDE[1] * 100
    print(f"Deviation from paper: {delta:+.1f}%")


if __name__ == '__main__':
    main()
