# -*- coding: utf-8 -*-
"""QMF18 EOS Reproduction — Main entry point.

Chain Template 11, W7→W8: Generates QMF18 EOS, M-R curve, and tidal Λ.
"""

import os
import sys
import time
import json
import numpy as np

# Ensure package-relative imports work
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from constants import (
    COUPLING, COUPLING_ALL, BENCHMARKS, hbar_c, Msun_km, km_to_Msun,
    mevfm3_to_km2, Msun_to_km,
)
from quark import compute_M_star, compute_dM_star_dsigma, build_M_star_table
from eos import generate_eos_table
from crust import build_unified_eos, load_bps_crust, load_nv_crust


def build_qmf18_eos(L=40, rho_B_min=0.01, rho_B_max=1.30, n_points=130,
                    verbose=True):
    """Generate QMF18 EOS table (β-equilibrium NS matter).

    Args:
        L: Symmetry energy slope [MeV] — selects coupling set.
        rho_B_min, rho_B_max: Density range [fm⁻³].
        n_points: Number of density grid points.
        verbose: Print progress.

    Returns:
        dict: {rho_B, Y_p, eps, P, ...} arrays + unified crust+core EOS.
    """
    coupling = COUPLING_ALL[L]

    t0 = time.time()

    # 1. Build M_N*(σ) lookup table
    if verbose:
        print("Building M_N*(σ) table ...")
    mstar_table = build_M_star_table(coupling['g_sigma_q'], alpha=0.0)

    from scipy.interpolate import PchipInterpolator
    M_star_interp = PchipInterpolator(mstar_table['sigma'], mstar_table['M_star'])
    dM_interp = PchipInterpolator(mstar_table['sigma'], mstar_table['dM_dsigma'])

    if verbose:
        M0 = M_star_interp(0.0)
        print(f"  M_N*(σ=0) = {M0:.1f} MeV (expect 939)")
        print(f"  Table built in {time.time()-t0:.1f}s")

    # 2. Generate EOS
    if verbose:
        print(f"\nGenerating QMF18 EOS (L={L} MeV, {n_points} density points) ...")

    rho_B_array = np.linspace(rho_B_min, rho_B_max, n_points)
    results = generate_eos_table(rho_B_array, coupling, M_star_interp, dM_interp,
                                 verbose=verbose)

    converged = [r for r in results if r['converged']]
    if verbose:
        print(f"  Converged: {len(converged)}/{len(results)} points")
        print(f"  EOS generation: {time.time()-t0:.1f}s")

    # 3. Build unified EOS with crust
    if verbose:
        print("\nBuilding unified crust+core EOS ...")
    unified = build_unified_eos(results)

    if verbose:
        print(f"  Unified EOS: {len(unified['nB'])} points")
        print(f"  nB range: {unified['nB'][0]:.4e} - {unified['nB'][-1]:.3f} fm⁻³")

    return {
        'core_results': results,
        'unified': unified,
        'coupling': coupling,
        'L': L,
        'mstar_table': mstar_table,
    }


def compute_benchmarks(eos_data):
    """Compare generated EOS with QMF18 Table 3 and Table 4 benchmarks.

    Returns:
        dict of benchmark comparisons.
    """
    unified = eos_data['unified']
    core_conv = [r for r in eos_data['core_results'] if r['converged']]

    benchmarks = {}

    # Check saturation properties for SNM (Y_p=0.5, no β-equilibrium)
    # This would need a separate SNM computation — skip for now

    # Check M_TOV if TOV solver is available
    # (delegated to separate tov run)

    return benchmarks


def print_summary(eos_data):
    """Print summary of generated EOS."""
    unified = eos_data['unified']
    coupling = eos_data['coupling']

    print("\n" + "=" * 60)
    print(f"QMF18 EOS Summary (L={eos_data['L']} MeV)")
    print("=" * 60)

    nB, eps, P = unified['nB'], unified['eps'], unified['P']

    print(f"\n  Unified EOS: {len(nB)} points")
    print(f"  nB range:    {nB[0]:.4e} – {nB[-1]:.3f} fm⁻³")
    print(f"  eps range:   {eps[0]:.4e} – {eps[-1]:.2f} MeV/fm³")
    print(f"  P range:     {P[0]:.4e} – {P[-1]:.2f} MeV/fm³")

    # Find saturation point (P ≈ 0)
    # The crust-core transition is at nB ≈ 0.08 fm⁻³
    idx_cc = np.argmin(np.abs(nB - 0.08))
    print(f"\n  At core-crust transition (nB≈0.08 fm⁻³):")
    print(f"    eps = {eps[idx_cc]:.2f} MeV/fm³")
    print(f"    P   = {P[idx_cc]:.4f} MeV/fm³")

    idx_016 = np.argmin(np.abs(nB - 0.16))
    if idx_016 < len(nB):
        print(f"\n  At saturation density (nB=0.16 fm⁻³):")
        print(f"    eps = {eps[idx_016]:.2f} MeV/fm³")
        print(f"    P   = {P[idx_016]:.4f} MeV/fm³")

    # Print selected EOS points for comparison with Table 3
    print(f"\n  Selected EOS table (cf. Table 3):")
    print(f"  {'nB [fm⁻³]':>12s}  {'eps [MeV/fm³]':>15s}  {'P [MeV/fm³]':>15s}  {'eps [g/cm³]':>15s}")
    target_nB = [0.082, 0.100, 0.160, 0.220, 0.280, 0.340, 0.400, 0.460,
                 0.520, 0.580, 0.640, 0.700, 0.800, 0.900, 1.000, 1.200, 1.300]
    for tn in target_nB:
        idx = np.argmin(np.abs(nB - tn))
        eps_cgs = eps[idx] * 1.6022e33 / 9e20  # MeV/fm³ → g/cm³ (approx)
        print(f"  {nB[idx]:12.6f}  {eps[idx]:15.6e}  {P[idx]:15.6e}  {eps_cgs:15.4e}")

    print(f"\n  Coupling parameters:")
    for k, v in coupling.items():
        print(f"    {k}: {v}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='QMF18 EOS Reproduction')
    ap.add_argument('-L', type=float, default=40.0,
                    help='Symmetry energy slope [MeV] (default: 40)')
    ap.add_argument('--n-points', type=int, default=130,
                    help='Number of density grid points')
    ap.add_argument('--rho-min', type=float, default=0.01,
                    help='Minimum density [fm⁻³]')
    ap.add_argument('--rho-max', type=float, default=1.30,
                    help='Maximum density [fm⁻³]')
    ap.add_argument('--output', type=str, default=None,
                    help='Output directory for tables')
    args = ap.parse_args()

    eos_data = build_qmf18_eos(
        L=args.L,
        rho_B_min=args.rho_min,
        rho_B_max=args.rho_max,
        n_points=args.n_points,
    )
    print_summary(eos_data)

    if args.output:
        os.makedirs(args.output, exist_ok=True)

        # Save EOS table
        unified = eos_data['unified']
        table = np.column_stack([unified['nB'], unified['eps'], unified['P']])
        header = "nB [fm^-3]  eps [MeV/fm^3]  P [MeV/fm^3]"
        path = os.path.join(args.output, f'qmf18_eos_L{int(args.L)}.dat')
        np.savetxt(path, table, header=header)
        print(f"\nEOS table saved to {path}")

        # Save benchmarks
        bench = compute_benchmarks(eos_data)
        path = os.path.join(args.output, f'qmf18_benchmarks_L{int(args.L)}.json')
        with open(path, 'w') as f:
            json.dump(bench, f, indent=2, default=str)
        print(f"Benchmarks saved to {path}")
