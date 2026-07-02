# -*- coding: utf-8 -*-
"""Quark-level solver — M_N*(sigma) with Zhu & Li (2018) PRC 97, 035805 Eqs.(5)-(9).

Implements the full explicit correction formulas for the nucleon mass in medium:
  - Eq.(5):  Centre-of-mass correction  ε_cm
  - Eq.(6):  Pionic correction          δM_N^π (includes I_π integral)
  - Eq.(7):  Gluonic correction         (ΔE_N)_g
  - Eq.(8):  Nucleon mass               M_N* = 3ε_q* − ε_cm + δM_N^π + (ΔE_N)_g
  - Eq.(9):  Nucleon radius             ⟨r_N²⟩

Replaces the linear approximation M*(σ) = 3·ε_q*(σ) + Δ₀ + Δ₁·σ (Δ₁ = +0.509
borrowed from Barik+2013) with the full physical corrections that carry intrinsic
σ-dependence through ε_q*(σ) and m_q*(σ).  Self-consistent with any g_sigma_q —
no cross-model parameter borrowing.

Reference: Zhu & Li (2018), Phys. Rev. C 97, 035805
           https://arxiv.org/abs/1710.00432
"""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

from constants import a_MeV3, V0, m_q, m_n, hbar_c

# ── Physical constants for corrections (Zhu & Li 2018 §II.A) ──────────
M_PI    = 140.0    # π meson mass [MeV]          — Eq.(6) context
F_PI    = 93.0     # pion decay constant [MeV]    — Eq.(6) context
ALPHA_C = 0.58     # strong coupling constant      — Eq.(7) context


# ══════════════════════════════════════════════════════════════════════════
#  Eq.(3) — Confined quark ground-state energy
# ══════════════════════════════════════════════════════════════════════════

def quark_energy_eq(eps_star, m_star):
    """Eq.(3): (epsilon_q' - m_q') * sqrt(lambda_q / a) - 3 = 0."""
    eps_prime = eps_star - V0 / 2.0
    m_prime   = m_star  + V0 / 2.0
    lambda_q  = eps_star + m_star
    if lambda_q <= 0 or a_MeV3 <= 0:
        return np.inf
    return (eps_prime - m_prime) * np.sqrt(lambda_q / a_MeV3) - 3.0


def solve_quark_energy(m_star, bracket=None):
    """Solve Eq.(3) for epsilon_q* given m_q*."""
    if bracket is None:
        lo = max(m_star - V0 + 1.0, 1.0)
        hi = lo + 2000.0
        for _ in range(20):
            if quark_energy_eq(lo, m_star) * quark_energy_eq(hi, m_star) < 0:
                break
            hi *= 1.5
        else:
            return np.nan
    else:
        lo, hi = bracket
    try:
        return brentq(quark_energy_eq, lo, hi, args=(m_star,), xtol=1e-12)
    except ValueError:
        return np.nan


# ══════════════════════════════════════════════════════════════════════════
#  Intermediate variables (shared by all three corrections)
# ══════════════════════════════════════════════════════════════════════════

def _derived_vars(eps_star, m_star):
    """Compute ε_q', m_q', λ_q, r₀ from ε_q* and m_q*.

    Returns dict with keys: eps_prime, m_prime, lambda_q, r0.
    All in MeV units; r0 in MeV⁻¹.
    """
    eps_prime = eps_star - V0 / 2.0
    m_prime   = m_star  + V0 / 2.0
    lambda_q  = eps_star + m_star
    r0        = (a_MeV3 * lambda_q) ** (-0.25)
    return {
        'eps_prime': eps_prime,
        'm_prime':   m_prime,
        'lambda_q':  lambda_q,
        'r0':        r0,
    }


# ══════════════════════════════════════════════════════════════════════════
#  Eq.(5) — Centre-of-mass correction
# ══════════════════════════════════════════════════════════════════════════

def compute_eps_cm(eps_star, m_star):
    """Eq.(5): ε_cm = (77·ε_q' + 31·m_q') / [3·(3·ε_q' + m_q')² · r₀²].

    Returns centre-of-mass energy correction [MeV].  Always positive
    (reduces the effective nucleon mass).
    """
    v = _derived_vars(eps_star, m_star)
    num = 77.0 * v['eps_prime'] + 31.0 * v['m_prime']
    denom_3mp = 3.0 * v['eps_prime'] + v['m_prime']
    denom = 3.0 * denom_3mp * denom_3mp * v['r0'] * v['r0']
    return num / denom


# ══════════════════════════════════════════════════════════════════════════
#  Eq.(6) — Pionic correction
# ══════════════════════════════════════════════════════════════════════════

def _u_k(k, lambda_q, eps_prime, m_prime, r0):
    """Eq.(6) shape factor: u(k) = [1 − 3k²/(2λ_q(5ε'+7m'))] · exp(−r₀²k²/4)."""
    bracket = 1.0 - 1.5 * k * k / (lambda_q * (5.0 * eps_prime + 7.0 * m_prime))
    return bracket * np.exp(-0.25 * r0 * r0 * k * k)


def _I_pi_integrand(k, lambda_q, eps_prime, m_prime, r0):
    """Integrand for I_π: k⁴ · u²(k) / (k² + m_π²)."""
    uk = _u_k(k, lambda_q, eps_prime, m_prime, r0)
    return k**4 * uk * uk / (k * k + M_PI * M_PI)


def compute_I_pi(lambda_q, eps_prime, m_prime, r0):
    """Eq.(6): I_π = (1/πm_π²) ∫₀^∞ dk k⁴ u²(k) / (k² + m_π²).

    Gaussian decay exp(−r₀²k²/2) ensures rapid convergence.
    Uses scipy.integrate.quad (adaptive Gauss-Kronrod).
    """
    prefac = 1.0 / (np.pi * M_PI * M_PI)
    integral, _err = quad(
        _I_pi_integrand, 0, np.inf,
        args=(lambda_q, eps_prime, m_prime, r0),
        limit=200,
    )
    return prefac * integral


def compute_f_NNpi(eps_prime, m_prime):
    """Eq.(6): f_NNπ = [(25ε'+35m')/(27ε'+9m')] · [m_π/(4√π f_π)]."""
    ratio = (25.0 * eps_prime + 35.0 * m_prime) / (27.0 * eps_prime + 9.0 * m_prime)
    return ratio * M_PI / (4.0 * np.sqrt(np.pi) * F_PI)


def compute_delta_M_pi(eps_star, m_star):
    """Eq.(6): δM_N^π = −(171/25) · I_π · f_NNπ².

    Returns pionic correction [MeV].  Always negative (attractive).
    """
    v = _derived_vars(eps_star, m_star)
    I_pi   = compute_I_pi(v['lambda_q'], v['eps_prime'], v['m_prime'], v['r0'])
    f_NNpi = compute_f_NNpi(v['eps_prime'], v['m_prime'])
    return -171.0 / 25.0 * I_pi * f_NNpi * f_NNpi


# ══════════════════════════════════════════════════════════════════════════
#  Eq.(7) — Gluonic correction
# ══════════════════════════════════════════════════════════════════════════

def compute_R_uu_sq(eps_prime, m_prime):
    """Eq.(7): R_uu² = 6 / (ε_q'² − m_q'²)."""
    denom = eps_prime * eps_prime - m_prime * m_prime
    if denom <= 0:
        return np.inf
    return 6.0 / denom


def compute_delta_E_g(eps_star, m_star):
    """Eq.(7): (ΔE_N)_g = −α_c · [256/(3√π) · 1/R_uu³ · 1/(3ε_q' + m_q')²].

    Returns gluonic correction [MeV].  Always negative (attractive).
    """
    v = _derived_vars(eps_star, m_star)
    R_uu_sq = compute_R_uu_sq(v['eps_prime'], v['m_prime'])
    if R_uu_sq <= 0 or not np.isfinite(R_uu_sq):
        return 0.0
    R_uu = np.sqrt(R_uu_sq)
    denom_3mp = 3.0 * v['eps_prime'] + v['m_prime']
    bracket = 256.0 / (3.0 * np.sqrt(np.pi)) / (R_uu * R_uu * R_uu) / (denom_3mp * denom_3mp)
    return -ALPHA_C * bracket


# ══════════════════════════════════════════════════════════════════════════
#  Eq.(8) — Nucleon effective mass in medium
# ══════════════════════════════════════════════════════════════════════════

def compute_M_star(sigma, g_sigma_q):
    """Eq.(8): M_N*(σ) = 3·ε_q* − ε_cm + δM_N^π + (ΔE_N)_g.

    All three corrections carry intrinsic σ-dependence through ε_q*(σ)
    and m_q*(σ) = m_q − g_σq·σ.  No cross-model parameter borrowing.

    Args:
        sigma:      σ meson field [MeV].
        g_sigma_q:  σ-quark coupling constant.

    Returns:
        M_N* [MeV].  Guaranteed ≥ 1.0 MeV.
    """
    m_star = m_q - g_sigma_q * sigma
    if m_star <= 1.0:
        return 1.0

    eps_star = solve_quark_energy(m_star)
    if not np.isfinite(eps_star) or eps_star <= 0:
        return 1.0

    eps_cm     = compute_eps_cm(eps_star, m_star)
    delta_M_pi = compute_delta_M_pi(eps_star, m_star)
    delta_E_g  = compute_delta_E_g(eps_star, m_star)

    M = 3.0 * eps_star - eps_cm + delta_M_pi + delta_E_g
    return max(M, 1.0)


# ══════════════════════════════════════════════════════════════════════════
#  Eq.(9) — Nucleon radius
# ══════════════════════════════════════════════════════════════════════════

def compute_r_N_squared(eps_star, m_star):
    """Eq.(9): ⟨r_N²⟩ = (11ε_q' + m_q') / [(3ε_q' + m_q')(ε_q'² − m_q'²)].

    Returns mean-square radius [MeV⁻²].  Divide by (ℏc)² for fm².
    """
    v = _derived_vars(eps_star, m_star)
    num = 11.0 * v['eps_prime'] + v['m_prime']
    denom_3mp = 3.0 * v['eps_prime'] + v['m_prime']
    denom_diff = v['eps_prime'] * v['eps_prime'] - v['m_prime'] * v['m_prime']
    if denom_diff <= 0:
        return np.inf
    return num / (denom_3mp * denom_diff)


def compute_r_N_fm(eps_star, m_star):
    """Eq.(9) in physical units: r_N [fm]."""
    r2_MeVm2 = compute_r_N_squared(eps_star, m_star)
    if not np.isfinite(r2_MeVm2) or r2_MeVm2 <= 0:
        return np.nan
    return np.sqrt(r2_MeVm2) * hbar_c   # MeV⁻¹ → fm: × (MeV·fm)


# ══════════════════════════════════════════════════════════════════════════
#  Numerical derivative + Table builder (compatible with quark.py API)
# ══════════════════════════════════════════════════════════════════════════

def compute_dM_star_dsigma(sigma, g_sigma_q, dsigma=0.01):
    """Central finite-difference derivative of M_N*(sigma)."""
    M_plus  = compute_M_star(sigma + dsigma, g_sigma_q)
    M_minus = compute_M_star(sigma - dsigma, g_sigma_q)
    return (M_plus - M_minus) / (2.0 * dsigma)


def build_M_star_table(g_sigma_q, sigma_min=0.0, sigma_max=80.0, n_points=200,
                       alpha=0.0):
    """Pre-compute M_N*(σ) and dM_N*/dσ on a grid.

    Args:
        g_sigma_q: sigma-quark coupling.
        sigma_min, sigma_max: sigma range [MeV].
        n_points: Grid resolution.
        alpha: Retained for backward compatibility — should be 0.0
               (the full correction formulas auto-adapt to g_sigma_q).

    Returns:
        dict with 'sigma', 'M_star', 'dM_dsigma' arrays.
    """
    sigma_grid = np.linspace(sigma_min, sigma_max, n_points)
    M_grid  = np.array([compute_M_star(s, g_sigma_q) + alpha * s
                        for s in sigma_grid])
    dM_grid = np.array([compute_dM_star_dsigma(s, g_sigma_q)
                        for s in sigma_grid]) + alpha
    return {'sigma': sigma_grid, 'M_star': M_grid, 'dM_dsigma': dM_grid}


# ══════════════════════════════════════════════════════════════════════════
#  Self-check — verifies vacuum calibration against paper benchmarks
# ══════════════════════════════════════════════════════════════════════════

def _self_check():
    """Verify physical benchmarks at vacuum (σ = 0).

    Checks:
      BC1: M_N*(0) = 939 MeV
      BC2: r_N(0)   ≈ 0.87 fm  (Table I, a=0.534296, V0=−62.257187)
    """
    import sys

    g_test = 3.8620366   # QMF18 g_σq (any valid g_σq should give M*(0)=939)
    sigma0 = 0.0

    m_star0  = m_q - g_test * sigma0  # = m_q = 300
    eps_star0 = solve_quark_energy(m_star0)

    if not np.isfinite(eps_star0):
        print("SELF-CHECK FAILED: solve_quark_energy(300) did not converge")
        return False

    M0 = compute_M_star(sigma0, g_test)
    if abs(M0 - 939.0) > 0.5:
        print(f"SELF-CHECK FAILED: M*(0) = {M0:.2f} MeV (expected 939)")
        return False

    r_N0 = compute_r_N_fm(eps_star0, m_star0)
    if not np.isfinite(r_N0) or abs(r_N0 - 0.87) > 0.05:
        print(f"SELF-CHECK FAILED: r_N(0) = {r_N0:.4f} fm (expected 0.87)")
        return False

    # Also verify correction signs
    eps_cm0     = compute_eps_cm(eps_star0, m_star0)
    delta_M_pi0 = compute_delta_M_pi(eps_star0, m_star0)
    delta_E_g0  = compute_delta_E_g(eps_star0, m_star0)

    checks = [
        (eps_cm0 > 0,        f"ε_cm(0) = {eps_cm0:.2f} ≤ 0"),
        (delta_M_pi0 < 0,    f"δM_N^π(0) = {delta_M_pi0:.2f} ≥ 0"),
        (delta_E_g0 < 0,     f"(ΔE_N)_g(0) = {delta_E_g0:.2f} ≥ 0"),
    ]
    for ok, msg in checks:
        if not ok:
            print(f"SELF-CHECK FAILED: {msg}")
            return False

    M_reconstructed = 3.0 * eps_star0 - eps_cm0 + delta_M_pi0 + delta_E_g0
    if abs(M_reconstructed - M0) > 0.01:
        print(f"SELF-CHECK FAILED: M* decomposition mismatch: "
              f"3×{eps_star0:.2f} − {eps_cm0:.2f} + {delta_M_pi0:.2f} + {delta_E_g0:.2f} "
              f"= {M_reconstructed:.2f} ≠ {M0:.2f}")
        return False

    print(f"Self-check PASSED:")
    print(f"  M*(0)      = {M0:.1f} MeV  (expect 939)")
    print(f"  r_N(0)     = {r_N0:.3f} fm   (expect 0.87)")
    print(f"  ε_cm(0)    = {eps_cm0:.2f} MeV")
    print(f"  δM_N^π(0)  = {delta_M_pi0:.2f} MeV")
    print(f"  (ΔE_N)_g(0) = {delta_E_g0:.2f} MeV")
    return True


if __name__ == '__main__':
    _self_check()
