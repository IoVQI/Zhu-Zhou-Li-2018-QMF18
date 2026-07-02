"""Crust EOS loading — BPS (1971) outer + N&V (1973) inner crust.

Data from data/: BPS 78-point P(ε) + N&V 11-point P(ε).
"""

import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_THIS_DIR, '..', '..', 'data')


def load_bps_crust():
    """Load BPS (1971) outer crust EOS.

    Returns:
        dict: {nB, eps, P} arrays in natural units [fm⁻³, MeV/fm³, MeV/fm³].
    """
    path = os.path.join(_DATA_DIR, 'bps_1971_outer_crust.dat')
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.readlines()
    # Skip comment lines starting with #
    data_lines = [l for l in raw if not l.startswith('#')]
    data = np.loadtxt(data_lines)
    return {
        'nB': data[:, 0],     # fm⁻³
        'eps': data[:, 1],    # MeV/fm³
        'P': data[:, 2],      # MeV/fm³
        'mu_B': data[:, 3],   # MeV
    }


def load_nv_crust():
    """Load N&V (1973) inner crust EOS (thermodynamically converted).

    Returns:
        dict: {nB, eps, P} arrays in natural units [fm⁻³, MeV/fm³, MeV/fm³].
    """
    path = os.path.join(_DATA_DIR, 'Negele_Vautherin_1973_inner_crust.dat')
    data = np.loadtxt(path, encoding='utf-8')
    return {
        'nB': data[:, 0],     # fm⁻³
        'eps': data[:, 1],    # MeV/fm³
        'P': data[:, 2],      # MeV/fm³
    }


def build_unified_eos(core_results, rho_crust_core=0.08):
    """Build unified crust + core EOS table.

    Appends BPS (outer) and N&V (inner) crust tables below the core-crust
    transition density, and QMF18 core EOS above it.

    Args:
        core_results: List of dicts from eos.generate_eos_table().
        rho_crust_core: Transition density [fm⁻³] (QMF18 default: 0.08).

    Returns:
        dict: {nB, eps, P} unified arrays sorted by nB.
    """
    bps = load_bps_crust()
    nv = load_nv_crust()

    # Extract core EOS points above transition
    core_converged = [r for r in core_results if r['converged']]
    core_nB = np.array([r['rho_B'] for r in core_converged])
    core_eps = np.array([r['eps'] for r in core_converged])
    core_P = np.array([r['P'] for r in core_converged])

    mask_core = core_nB >= rho_crust_core

    # Build unified table
    nB_parts = []
    eps_parts = []
    P_parts = []

    # BPS outer crust: all points (up to ~0.001 fm⁻³)
    # But some BPS points extend beyond 0.001 → truncate
    mask_bps = bps['nB'] < 0.001
    nB_parts.append(bps['nB'][mask_bps])
    eps_parts.append(bps['eps'][mask_bps])
    P_parts.append(bps['P'][mask_bps])

    # N&V inner crust: all 11 points (0.0003 - 0.079 fm⁻³)
    # Filter to points not overlapping with BPS
    mask_nv = nv['nB'] >= 0.001
    nB_parts.append(nv['nB'][mask_nv])
    eps_parts.append(nv['eps'][mask_nv])
    P_parts.append(nv['P'][mask_nv])

    # QMF18 core
    nB_parts.append(core_nB[mask_core])
    eps_parts.append(core_eps[mask_core])
    P_parts.append(core_P[mask_core])

    nB_all = np.concatenate(nB_parts)
    eps_all = np.concatenate(eps_parts)
    P_all = np.concatenate(P_parts)

    # Sort by nB
    order = np.argsort(nB_all)
    return {
        'nB': nB_all[order],
        'eps': eps_all[order],
        'P': P_all[order],
    }
