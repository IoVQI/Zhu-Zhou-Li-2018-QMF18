#!/usr/bin/env python3
"""Step 1: Generate QMF18 EOS table for a given L value.

Pipeline: quark M*(sigma) -> meson field equations -> beta equilibrium -> crust EOS.

Usage:
    python generate_eos.py -L 40                # single L
    python generate_eos.py -L 20,40,60,80       # all four L values
    python generate_eos.py -L 40 --n-points 128 --no-snm  # custom resolution, skip SNM

Output per L (in output/qmf18_L{}/):
    eos.csv        — unified crust+core EOS table
    benchmarks.json — metadata + SNM saturation + placeholder for TOV results
"""

import os, sys, time, json, argparse
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from scipy.interpolate import PchipInterpolator
from constants import (
    COUPLING_ALL, m_n, hbar_c,
    a_fm3, a_MeV3, V0, m_q, rho0_fm3, EA_sat, K_sat, Ms_ratio_sat,
)
from quark_barik_full import build_M_star_table
from eos import generate_eos_table
from crust import build_unified_eos
from fields import solve_meson_fields, compute_energy_pressure


def compute_snm_saturation(coupling, M_star_interp, dM_interp):
    """Find SNM (Y_p=0.5) saturation point and properties.

    Returns dict with rho0, EA, K, Mstar_ratio, P_sat.
    """
    rho_vals = np.linspace(0.08, 0.30, 200)
    results = []
    for rho_B in rho_vals:
        fields = solve_meson_fields(rho_B, 0.5, coupling, M_star_interp, dM_interp)
        if not fields['converged']:
            continue
        eps, P = compute_energy_pressure(rho_B, 0.5, fields, coupling)
        EA = eps / rho_B - m_n
        results.append({
            'rho': rho_B, 'EA': EA, 'P': P,
            'M_star': fields['M_star'], 'sigma': fields['sigma'],
        })

    if not results:
        return None

    rhos = np.array([r['rho'] for r in results])
    EAs  = np.array([r['EA']  for r in results])
    Ps   = np.array([r['P']   for r in results])

    idx = np.argmin(EAs)
    dpdrho = np.gradient(Ps, rhos)

    return {
        'rho0_fm3':     float(rhos[idx]),
        'EA_MeV':       float(EAs[idx]),
        'K_MeV':        float(9.0 * dpdrho[idx]),
        'Mstar_ratio':  float(results[idx]['M_star'] / m_n),
        'P_sat_MeVfm3': float(Ps[idx]),
        'sigma_sat_MeV': float(results[idx]['sigma']),
    }


def generate_eos(L, n_points=96, rho_min=0.01, rho_max=1.30, do_snm=True):
    """Generate QMF18 EOS for a given symmetry energy slope L.

    Returns dict with 'unified' EOS arrays, 'coupling', 'snm' (if computed).
    """
    coupling = COUPLING_ALL[L]
    g_sigma_q = coupling['g_sigma_q']

    # Build M*(sigma) table
    mstar_table = build_M_star_table(g_sigma_q, n_points=200)
    M_f = PchipInterpolator(mstar_table['sigma'], mstar_table['M_star'])
    dM_f = PchipInterpolator(mstar_table['sigma'], mstar_table['dM_dsigma'])

    # Generate beta-equilibrium EOS
    rho_B_array = np.linspace(rho_min, rho_max, n_points)
    results = generate_eos_table(rho_B_array, coupling, M_f, dM_f, verbose=False)

    converged = [r for r in results if r['converged']]
    if len(converged) < n_points:
        print(f"  WARNING: only {len(converged)}/{n_points} points converged")

    # Build unified crust+core EOS
    unified = build_unified_eos(results)

    # SNM saturation
    snm = compute_snm_saturation(coupling, M_f, dM_f) if do_snm else None

    return {
        'L': L,
        'coupling': coupling,
        'unified': unified,
        'snm': snm,
        'n_converged': len(converged),
        'n_total': n_points,
    }


def save_output(eos_data, output_dir):
    """Save EOS data to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    L = eos_data['L']
    unified = eos_data['unified']

    # Save EOS table as CSV
    csv_path = os.path.join(output_dir, 'eos.csv')
    header = 'nB_fm3,eps_MeVfm3,P_MeVfm3'
    np.savetxt(csv_path,
               np.column_stack([unified['nB'], unified['eps'], unified['P']]),
               delimiter=',', header=header, comments='')
    print(f"  EOS table -> {csv_path}  ({len(unified['nB'])} rows)")

    # Save benchmarks JSON (metadata + SNM, TOV placeholder for step 2)
    bench_path = os.path.join(output_dir, 'benchmarks.json')
    benchmarks = {
        'paper': 'Zhu, Zhou & Li (2018), ApJ 862, 98',
        'arxiv': '1802.05510',
        'model': 'QMF18',
        'L_MeV': L,
        'code_version': 'quark_barik_full.py (Zhu & Li 2018 Eqs.5-9)',
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'eos_points': len(unified['nB']),
        'nB_min_fm3': float(unified['nB'][0]),
        'nB_max_fm3': float(unified['nB'][-1]),
        'coupling': {k: v for k, v in eos_data['coupling'].items()
                     if not k.startswith('g_omega_N') and not k.startswith('g_rho_N')},
    }

    if eos_data['snm']:
        snm = eos_data['snm']
        benchmarks['snm_saturation'] = snm
        benchmarks['snm_targets'] = {
            'rho0_fm3': rho0_fm3, 'EA_MeV': EA_sat,
            'K_MeV': K_sat, 'Mstar_ratio': Ms_ratio_sat,
        }
        # Compute deltas
        benchmarks['snm_deltas'] = {
            'rho0_pct': round((snm['rho0_fm3'] - rho0_fm3) / rho0_fm3 * 100, 2),
            'EA_diff_MeV': round(snm['EA_MeV'] - EA_sat, 2),
            'K_pct': round((snm['K_MeV'] - K_sat) / K_sat * 100, 1),
            'Mstar_ratio_diff': round(snm['Mstar_ratio'] - Ms_ratio_sat, 3),
        }

    # TOV placeholder
    benchmarks['tov'] = None  # filled by compute_tov.py

    with open(bench_path, 'w') as f:
        json.dump(benchmarks, f, indent=2)
    print(f"  Benchmarks  -> {bench_path}")


# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='QMF18 EOS Generation (Step 1)')
    ap.add_argument('-L', type=str, default='40',
                    help='Symmetry energy slope(s): single (40) or comma-separated (20,40,60,80)')
    ap.add_argument('--n-points', type=int, default=96,
                    help='EOS density grid resolution')
    ap.add_argument('--rho-min', type=float, default=0.01,
                    help='Minimum density [fm^-3]')
    ap.add_argument('--rho-max', type=float, default=1.30,
                    help='Maximum density [fm^-3]')
    ap.add_argument('--no-snm', action='store_true',
                    help='Skip SNM saturation computation')
    ap.add_argument('--output-root', type=str, default=None,
                    help='Output root directory (default: ./output)')
    args = ap.parse_args()

    L_values = [float(x.strip()) for x in args.L.split(',')]
    output_root = args.output_root or os.path.join(_THIS_DIR, 'output')

    for L in L_values:
        print(f'\n{"="*50}')
        print(f'QMF18 EOS Generation — L={L:.0f} MeV')
        print(f'{"="*50}')

        t0 = time.time()
        eos_data = generate_eos(L, n_points=args.n_points,
                               rho_min=args.rho_min, rho_max=args.rho_max,
                               do_snm=not args.no_snm)
        elapsed = time.time() - t0

        output_dir = os.path.join(output_root, f'qmf18_L{L:.0f}')
        save_output(eos_data, output_dir)

        print(f'  Completed in {elapsed:.1f}s')
        if eos_data['snm']:
            s = eos_data['snm']
            print(f'  SNM: rho0={s["rho0_fm3"]:.3f}, E/A={s["EA_MeV"]:.2f}, '
                  f'K={s["K_MeV"]:.0f}, M*/M={s["Mstar_ratio"]:.3f}')
