#!/usr/bin/env python3
"""Plot QMF18 EOS: EOS properties, M-R, Lambda-M for all L values.

Output: 3 PNG files saved to the run directory.

Usage:
    python plot_eos.py output/2026-07-01_4L_scan
    python plot_eos.py                         # auto-detect latest run
"""

import os, sys, glob as globmod, csv
import numpy as np
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from constants import COUPLING_ALL, m_n
from quark_barik_full import build_M_star_table
from fields import solve_meson_fields, compute_energy_pressure

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.fontsize': 8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
})

L_COLORS  = {20: '#2166ac', 40: '#b2182b', 60: '#4daf4a', 80: '#ff7f00'}
L_LS      = {20: '--',      40: '-',       60: '-.',      80: ':'}


# ══════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════

def find_run_dir(explicit_dir=None):
    if explicit_dir and os.path.isdir(explicit_dir):
        return explicit_dir
    output_root = os.path.join(_THIS_DIR, 'output')
    candidates = []
    for entry in os.listdir(output_root):
        path = os.path.join(output_root, entry)
        if not os.path.isdir(path) or entry in ('archive', '.git'):
            continue
        if globmod.glob(os.path.join(path, 'qmf18_L*')):
            candidates.append((entry, path))
    if not candidates:
        raise FileNotFoundError(f"No run dirs with qmf18_L* under {output_root}")
    candidates.sort(reverse=True)
    print(f"Run: {candidates[0][0]}")
    return candidates[0][1]


def load_mr(run_dir):
    """Load M-R + tidal data for all L values."""
    data = {}
    for L_val in [20, 40, 60, 80]:
        mr_path = os.path.join(run_dir, f'qmf18_L{L_val}', 'mr_curve.csv')
        if not os.path.exists(mr_path):
            print(f"  WARNING: missing mr_curve.csv for L={L_val}")
            continue
        mr = np.loadtxt(mr_path, delimiter=',', skiprows=1)
        data[L_val] = {
            'rho_c': mr[:, 0], 'M': mr[:, 1], 'R': mr[:, 2],
            'Lambda': mr[:, 3], 'k2': mr[:, 4], 'C': mr[:, 5],
        }
    return data


def compute_eos_properties(rho_grid, Ls):
    """Compute P(nB) and E_sym(nB) for all L values.

    Returns dict[L] -> {nB, P, eps, E_sym}.
    """
    mstar_table = build_M_star_table(3.8620366, n_points=200)
    M_f  = PchipInterpolator(mstar_table['sigma'], mstar_table['M_star'])
    dM_f = PchipInterpolator(mstar_table['sigma'], mstar_table['dM_dsigma'])

    result = {}
    for L_val in Ls:
        coupling = COUPLING_ALL[L_val]
        P_arr, eps_arr, E_sym_arr = [], [], []
        valid_rho = []

        for rho_B in rho_grid:
            # SNM (Y_p = 0.5)
            f_snm = solve_meson_fields(rho_B, 0.5, coupling, M_f, dM_f)
            if not f_snm['converged']:
                continue
            eps_snm, P_snm = compute_energy_pressure(rho_B, 0.5, f_snm, coupling)
            EA_snm = eps_snm / rho_B - m_n

            # PNM (Y_p = 0)
            f_pnm = solve_meson_fields(rho_B, 0.0, coupling, M_f, dM_f)
            if not f_pnm['converged']:
                continue
            eps_pnm, P_pnm = compute_energy_pressure(rho_B, 0.0, f_pnm, coupling)
            EA_pnm = eps_pnm / rho_B - m_n

            E_sym = EA_pnm - EA_snm

            valid_rho.append(rho_B)
            P_arr.append(P_snm)
            eps_arr.append(eps_snm)
            E_sym_arr.append(E_sym)

        result[L_val] = {
            'nB':   np.array(valid_rho),
            'P':    np.array(P_arr),
            'eps':  np.array(eps_arr),
            'E_sym': np.array(E_sym_arr),
        }
    return result


# ══════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════

def plot_eos_properties(eos_data, save_dir):
    """Figure 1: P(nB) + E_sym(nB)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    Ls = sorted(eos_data.keys())
    for L_val in Ls:
        d = eos_data[L_val]
        ax1.semilogy(d['nB'], d['P'], color=L_COLORS[L_val],
                     linestyle=L_LS[L_val], linewidth=1.2,
                     label=f'L = {L_val} MeV')
        ax2.semilogy(d['nB'], d['E_sym'], color=L_COLORS[L_val],
                     linestyle=L_LS[L_val], linewidth=1.2,
                     label=f'L = {L_val} MeV')

    ax1.set_xlabel(r'$n_B$ [fm$^{-3}$]')
    ax1.set_ylabel(r'$P$ [MeV / fm$^3$]')
    ax1.set_xlim(0.1, 0.8)
    ax1.set_ylim(1, 300)
    ax1.set_title('Pressure (SNM)')
    ax1.legend()

    ax2.set_xlabel(r'$n_B$ [fm$^{-3}$]')
    ax2.set_ylabel(r'$E_{\rm sym}$ [MeV]')
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(1, 100)
    ax2.set_title('Symmetry Energy')
    ax2.legend()

    fig.suptitle('QMF18 — EOS Properties', fontsize=13, y=1.02)
    fig.tight_layout()

    path = os.path.join(save_dir, 'eos_properties.png')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    print(f'  [1/3] {path}')
    plt.close(fig)


def plot_mr(mr_data, save_dir):
    """Figure 2: M-R curve (R: 10-16 km, M: 0-2.5 M_sun)."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    Ls = sorted(mr_data.keys())
    # Arrow offsets: length+angle tuned per L to keep arrows clear of curves
    arrow_opts = {
        20: dict(xytext=(30, 20), textcoords='offset points'),
        40: dict(xytext=(40, 15), textcoords='offset points'),
        60: dict(xytext=(45, 10), textcoords='offset points'),
        80: dict(xytext=(25, 20), textcoords='offset points'),
    }
    for L_val in Ls:
        d = mr_data[L_val]
        ax.plot(d['R'], d['M'], color=L_COLORS[L_val],
                linestyle=L_LS[L_val], linewidth=1.4,
                label=f'L = {L_val} MeV')
        # Interpolate to exact M = 1.4
        R14 = float(np.interp(1.4, d['M'], d['R']))
        L14 = float(np.interp(1.4, d['M'], d['Lambda']))
        ax.scatter(R14, 1.4, color=L_COLORS[L_val], s=30, zorder=5,
                   edgecolors='white', linewidths=0.5)
        # Arrow pointing to intersection
        kw = arrow_opts.get(L_val, dict(xytext=(30, 20), textcoords='offset points'))
        ax.annotate(r'$\Lambda(1.4)\!=\!\;$' + f'{L14:.0f}',
                    xy=(R14, 1.4), fontsize=7, color=L_COLORS[L_val],
                    arrowprops=dict(arrowstyle='->', color=L_COLORS[L_val],
                                    lw=0.8, connectionstyle='arc3,rad=0.1'),
                    **kw)

    ax.axhline(2.0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.text(15.8, 2.02, r'$2\,M_\odot$', fontsize=8, color='gray', ha='right')
    ax.axhline(1.4, color='gray', linestyle='--', alpha=0.4, linewidth=0.7)
    ax.text(15.8, 1.42, r'$1.4\,M_\odot$', fontsize=8, color='gray', ha='right')

    ax.set_xlim(10, 16)
    ax.set_ylim(0, 2.5)
    ax.set_xlabel(r'$R$ [km]')
    ax.set_ylabel(r'$M$ [$M_\odot$]')
    ax.set_title('QMF18 — Mass–Radius Relation')
    ax.legend(loc='lower right')

    fig.tight_layout()

    path = os.path.join(save_dir, 'mr_curve.png')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    print(f'  [2/3] {path}')
    plt.close(fig)


def plot_lambda(mr_data, save_dir):
    """Figure 3: Lambda vs M (left) + Lambda vs M/R (right), shared y-axis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6),
                                    sharey=True, gridspec_kw={'wspace': 0.06})

    Ls = sorted(mr_data.keys())
    for L_val in Ls:
        d = mr_data[L_val]
        # Left: Lambda vs M
        ax1.plot(d['M'], d['Lambda'], color=L_COLORS[L_val],
                 linestyle=L_LS[L_val], linewidth=1.4,
                 label=f'L = {L_val} MeV')
        idx_14 = np.argmin(np.abs(d['M'] - 1.4))
        ax1.scatter(d['M'][idx_14], d['Lambda'][idx_14],
                    color=L_COLORS[L_val], s=30, zorder=5, edgecolors='white',
                    linewidths=0.5)
        # Right: Lambda vs M/R (compactness)
        ax2.plot(d['C'], d['Lambda'], color=L_COLORS[L_val],
                 linestyle=L_LS[L_val], linewidth=1.4)
        C14 = float(np.interp(1.4, d['M'], d['C']))
        ax2.scatter(C14, d['Lambda'][idx_14],
                    color=L_COLORS[L_val], s=30, zorder=5, edgecolors='white',
                    linewidths=0.5)

    # --- Left panel annotations ---
    ax1.axhline(800, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax1.text(0.05, 820, r'$\Lambda(1.4)=800$  (GW170817)', fontsize=8, color='gray')
    ax1.axvline(1.4, color='gray', linestyle='--', alpha=0.4, linewidth=0.7)
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 3000)
    ax1.set_xlabel(r'$M$ [$M_\odot$]')
    ax1.set_ylabel(r'$\Lambda$')
    ax1.set_title('QMF18 — Tidal Deformability')
    ax1.legend(loc='upper right')

    # --- Right panel annotations ---
    ax2.axhline(800, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax2.set_xlim(0, 0.25)
    ax2.set_xlabel(r'$M/R$ [$M_\odot$ km$^{-1}$]')
    ax2.set_title('QMF18 — Compactness')

    path = os.path.join(save_dir, 'lambda_m.png')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    print(f'  [3/5] {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  Figure 4: k2 vs M (tidal Love number)
# ══════════════════════════════════════════════════════════════════════════

def plot_k2(mr_data, save_dir):
    """Figure 4: k2 vs M — standalone, x: 0-2.5 M_sun, y: 0-0.2.

    Applies Savitzky-Golay smoothing to suppress Love ODE numerical noise,
    especially for L=20 (soft EOS). Raw data shown as faint background.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    Ls = sorted(mr_data.keys())
    for L_val in Ls:
        d = mr_data[L_val]
        M, k2_raw = d['M'], d['k2']
        # Smooth: window = ~7% of data points, odd, poly order 3
        w = max(5, 2 * (len(M) // 28) + 1)
        k2_sm = savgol_filter(k2_raw, w, 3)
        # Raw data as faint line
        ax.plot(M, k2_raw, color=L_COLORS[L_val], alpha=0.15, linewidth=0.6)
        # Smoothed as main line
        ax.plot(M, k2_sm, color=L_COLORS[L_val],
                linestyle=L_LS[L_val], linewidth=1.4,
                label=f'L = {L_val} MeV')
        idx_14 = np.argmin(np.abs(M - 1.4))
        ax.scatter(M[idx_14], k2_sm[idx_14],
                   color=L_COLORS[L_val], s=30, zorder=5,
                   edgecolors='white', linewidths=0.5)

    ax.axvline(1.4, color='gray', linestyle='--', alpha=0.4, linewidth=0.7)
    ax.text(1.42, 0.195, r'$1.4\,M_\odot$', fontsize=8, color='gray')

    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 0.2)
    ax.set_xlabel(r'$M$ [$M_\odot$]')
    ax.set_ylabel(r'$k_2$')
    ax.set_title('QMF18 — Tidal Love Number')
    ax.legend(loc='upper right')

    fig.tight_layout()

    path = os.path.join(save_dir, 'k2_vs_m.png')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    print(f'  [4/5] {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  Figure 5: EOS comparison with paper Table 3
# ══════════════════════════════════════════════════════════════════════════

# Paper Table 3: (epsilon [g/cm³], P [erg/cm³], rho_N [fm⁻³])
TABLE3_PAPER = [
    (0.13855e+15, 0.79586e+33, 0.082),
    (0.14365e+15, 0.85234e+33, 0.085),
    (0.15216e+15, 0.95144e+33, 0.090),
    (0.16920e+15, 0.11706e+34, 0.100),
    (0.18626e+15, 0.14226e+34, 0.110),
    (0.20336e+15, 0.17145e+34, 0.120),
    (0.22047e+15, 0.20433e+34, 0.130),
    (0.27203e+15, 0.33950e+34, 0.160),
    (0.32393e+15, 0.55426e+34, 0.190),
    (0.37631e+15, 0.87679e+34, 0.220),
    (0.42926e+15, 0.13315e+35, 0.250),
    (0.48293e+15, 0.19385e+35, 0.280),
    (0.53741e+15, 0.27149e+35, 0.310),
    (0.59282e+15, 0.36752e+35, 0.340),
    (0.64927e+15, 0.48329e+35, 0.370),
    (0.70686e+15, 0.62008e+35, 0.400),
    (0.76568e+15, 0.77912e+35, 0.430),
    (0.82583e+15, 0.96151e+35, 0.460),
    (0.88738e+15, 0.11682e+36, 0.490),
    (0.95043e+15, 0.13999e+36, 0.520),
    (0.10150e+16, 0.16569e+36, 0.550),
    (0.10813e+16, 0.19389e+36, 0.580),
    (0.11492e+16, 0.22449e+36, 0.610),
    (0.12189e+16, 0.25733e+36, 0.640),
    (0.12904e+16, 0.29223e+36, 0.670),
    (0.13636e+16, 0.32903e+36, 0.700),
    (0.14896e+16, 0.39423e+36, 0.750),
    (0.16207e+16, 0.46399e+36, 0.800),
    (0.17568e+16, 0.53809e+36, 0.850),
    (0.18978e+16, 0.61645e+36, 0.900),
    (0.20438e+16, 0.69900e+36, 0.950),
    (0.21948e+16, 0.78573e+36, 1.000),
    (0.25116e+16, 0.97160e+36, 1.100),
    (0.28480e+16, 0.11739e+37, 1.200),
    (0.32039e+16, 0.13926e+37, 1.300),
]

# Unit conversion: paper Table 3 cgs → MeV/fm³
#   Mass density:   1 MeV/fm³ / c² = 1.7827e12 g/cm³  →  g/cm³ * (1/1.7827e12) = MeV/fm³
#   Pressure:       1 MeV/fm³ = 1.6022e33 erg/cm³     →  erg/cm³ * (1/1.6022e33) = MeV/fm³
GCM3_TO_MEVFM3  = 1.0 / 1.7827e12
ERGCM3_TO_MEVFM3 = 1.0 / 1.6022e33


def load_eos_csv(run_dir, L_val):
    """Load eos.csv → (eps_MeVfm3, P_MeVfm3) arrays (already in MeV/fm³)."""
    path = os.path.join(run_dir, f'qmf18_L{L_val}', 'eos.csv')
    eps, P = [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            eps.append(float(row['eps_MeVfm3']))
            P.append(float(row['P_MeVfm3']))
    return np.array(eps), np.array(P)


def plot_eos_vs_table3(run_dir, save_dir):
    """Figure 4: Code EOS (4 L values) vs paper Table 3 — linear axes, MeV/fm³."""
    fig, ax = plt.subplots(figsize=(5, 10))

    # Plot code EOS for all L values (already in MeV/fm³)
    for L_val in [20, 40, 60, 80]:
        eps_mf, P_mf = load_eos_csv(run_dir, L_val)
        ax.plot(eps_mf, P_mf,
                color=L_COLORS[L_val], linestyle=L_LS[L_val],
                linewidth=1.4, label=f'Code L={L_val} MeV')

    # Convert paper Table 3 from cgs to MeV/fm³
    t3_eps_mf = [r[0] * GCM3_TO_MEVFM3 for r in TABLE3_PAPER]
    t3_P_mf   = [r[1] * ERGCM3_TO_MEVFM3 for r in TABLE3_PAPER]
    ax.scatter(t3_eps_mf, t3_P_mf, color='black', s=18, zorder=10,
               marker='D', edgecolors='white', linewidths=0.4,
               label='Paper Table 3 (L=40)')

    ax.set_xlabel(r'$\varepsilon$ [MeV fm$^{-3}$]')
    ax.set_ylabel(r'$P$ [MeV fm$^{-3}$]')
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 800)
    ax.set_title('QMF18 — EOS Comparison with Paper Table 3')
    ax.legend(loc='upper left', fontsize=8)

    fig.tight_layout()

    path = os.path.join(save_dir, 'eos_vs_table3.png')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    print(f'  [5/5] {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Plot QMF18 EOS (5 figures)')
    ap.add_argument('run_dir', nargs='?', default=None,
                    help='Run directory (default: auto-detect latest)')
    args = ap.parse_args()

    run_dir = find_run_dir(args.run_dir)

    # Shared density grid for EOS properties (start low to show E_sym -> 0)
    rho_grid = np.linspace(0.005, 1.10, 100)
    Ls = [20, 40, 60, 80]

    print('Computing EOS properties (P, E_sym) ...')
    eos_data = compute_eos_properties(rho_grid, Ls)

    print('Loading M-R data ...')
    mr_data = load_mr(run_dir)

    print('Plotting ...')
    plot_eos_properties(eos_data, run_dir)
    plot_mr(mr_data, run_dir)
    plot_lambda(mr_data, run_dir)
    plot_k2(mr_data, run_dir)
    plot_eos_vs_table3(run_dir, run_dir)
    print('Done.')
