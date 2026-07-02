#!/usr/bin/env python3
"""Step 2: TOV integration + tidal deformability for a pre-generated QMF18 EOS.

Reads eos.csv from a qmf18_L{} directory, runs the pure-Python TOV solver,
and writes mr_curve.csv + updates benchmarks.json with TOV/tidal results.

Usage:
    python compute_tov.py output/qmf18_L40           # single directory
    python compute_tov.py output/qmf18_L20 output/qmf18_L40 ...  # multiple
    python compute_tov.py output/qmf18_L*             # glob (shell-expanded)
"""

import os, sys, time, json, glob as globmod
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from tov import TovSolver as CppTovSolver


def solve_tov_tidal(eos_path, dr=0.001, rs=1e-5, max_steps=200000, n_rho_c=200):
    """Run TOV + tidal computation on an EOS table.

    Args:
        eos_path: Path to eos.csv file (3 columns: nB, eps, P).
        dr, rs, max_steps: TOV integration parameters.
        n_rho_c: Number of central density grid points (default 200, use 500 for fine).

    Returns:
        dict with mr_curve arrays, benchmarks, and metadata.
    """
    # Read EOS
    data = np.loadtxt(eos_path, delimiter=',', skiprows=1)
    nB = data[:, 0]
    eps = data[:, 1]
    P   = data[:, 2]

    # Central density grid
    rho_c_grid = np.logspace(np.log10(0.15), np.log10(3.5), n_rho_c)

    # Run TOV + tidal via Python solver
    solver = CppTovSolver(dr=dr, rs=rs, max_steps=max_steps)
    result = solver.solve_tidal_batch(nB, P, eps, rho_c_grid)

    ok = result['success']
    if ok.sum() == 0:
        raise RuntimeError("TOV solver: zero successful integrations")

    # Filter to stable branch (ascending M, up to M_max)
    M_all = result['M_Msun'][ok]
    R_all = result['R_km'][ok]
    L_all = result['Lambda'][ok]
    k2_all = result['k2'][ok]
    C_all = result['C'][ok]
    rho_c_all = rho_c_grid[ok]

    idx_peak = np.argmax(M_all)
    mask = np.concatenate([[True], np.diff(M_all[:idx_peak + 1]) > 1e-6])
    M_stable = M_all[:idx_peak + 1][mask]
    R_stable = R_all[:idx_peak + 1][mask]
    L_stable = L_all[:idx_peak + 1][mask]
    k2_stable = k2_all[:idx_peak + 1][mask]
    C_stable = C_all[:idx_peak + 1][mask]
    rho_stable = rho_c_all[:idx_peak + 1][mask]

    # Extract benchmarks
    M_max = float(M_all[idx_peak])
    R_at_Mmax = float(R_all[idx_peak])

    # Interpolate at M = 1.4 M_sun
    if M_stable[0] <= 1.4 <= M_stable[-1]:
        R14 = float(np.interp(1.4, M_stable, R_stable))
        L14 = float(np.interp(1.4, M_stable, L_stable))
        k2_14 = float(np.interp(1.4, M_stable, k2_stable))
        C14 = float(np.interp(1.4, M_stable, C_stable))
    else:
        R14 = L14 = k2_14 = C14 = None

    return {
        'mr_curve': {
            'rho_c_fm3': rho_stable.tolist(),
            'M_Msun': M_stable.tolist(),
            'R_km': R_stable.tolist(),
            'Lambda': L_stable.tolist(),
            'k2': k2_stable.tolist(),
            'C': C_stable.tolist(),
        },
        'benchmarks': {
            'M_TOV_Msun': M_max,
            'R_at_Mmax_km': R_at_Mmax,
            'R14_km': R14,
            'Lambda14': L14,
            'k2_14': k2_14,
            'C14': C14,
            'MR14': C14,  # M(km)/R(km) = compactness
        },
        'n_stars': int(ok.sum()),
        'dr_km': dr,
    }


def save_results(tov_data, output_dir):
    """Save MR curve CSV and update benchmarks.json."""
    os.makedirs(output_dir, exist_ok=True)

    # Save MR curve
    mr = tov_data['mr_curve']
    mr_path = os.path.join(output_dir, 'mr_curve.csv')
    header = 'rho_c_fm3,M_Msun,R_km,Lambda,k2,C'
    np.savetxt(mr_path,
               np.column_stack([mr['rho_c_fm3'], mr['M_Msun'], mr['R_km'],
                                mr['Lambda'], mr['k2'], mr['C']]),
               delimiter=',', header=header, comments='')
    n_pts = len(mr['M_Msun'])
    print(f"  MR curve    -> {mr_path}  ({n_pts} stable stars)")

    # Update benchmarks.json
    bench_path = os.path.join(output_dir, 'benchmarks.json')
    if os.path.exists(bench_path):
        with open(bench_path, 'r') as f:
            benchmarks = json.load(f)
    else:
        benchmarks = {}

    benchmarks['tov'] = {
        'solver': 'Pure Python RK4 (tov.py)',
        'dr_km': tov_data['dr_km'],
        'n_stars': tov_data['n_stars'],
        **tov_data['benchmarks'],
    }

    with open(bench_path, 'w') as f:
        json.dump(benchmarks, f, indent=2)
    print(f"  Benchmarks  -> {bench_path}")

    # Print summary
    b = tov_data['benchmarks']
    print(f"  M_TOV = {b['M_TOV_Msun']:.4f} M_sun")
    if b['R14_km'] is not None:
        print(f"  R(1.4) = {b['R14_km']:.2f} km,  "
              f"Lambda(1.4) = {b['Lambda14']:.0f},  "
              f"M/R = {b['C14']:.4f}")


# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='QMF18 TOV + Tidal (Step 2)')
    ap.add_argument('dirs', nargs='+',
                    help='Output directories (e.g. output/qmf18_L40) or glob patterns')
    ap.add_argument('--dr', type=float, default=0.001,
                    help='TOV step size [km]')
    ap.add_argument('--max-steps', type=int, default=200000,
                    help='Max TOV integration steps')
    ap.add_argument('--n-rho-c', type=int, default=200,
                    help='Number of central density grid points (default 200)')
    args = ap.parse_args()

    # Expand globs
    dirs = []
    for d in args.dirs:
        if '*' in d or '?' in d:
            dirs.extend(globmod.glob(d))
        else:
            dirs.append(d)

    for output_dir in dirs:
        eos_path = os.path.join(output_dir, 'eos.csv')
        if not os.path.exists(eos_path):
            print(f'SKIP {output_dir}: no eos.csv found')
            continue

        L_str = os.path.basename(output_dir).replace('qmf18_L', '')
        print(f'\n{"="*50}')
        print(f'QMF18 TOV + Tidal — L={L_str} MeV')
        print(f'{"="*50}')

        t0 = time.time()
        tov_data = solve_tov_tidal(eos_path, dr=args.dr, max_steps=args.max_steps,
                                   n_rho_c=args.n_rho_c)
        elapsed = time.time() - t0

        save_results(tov_data, output_dir)
        print(f'  Completed in {elapsed:.1f}s')
