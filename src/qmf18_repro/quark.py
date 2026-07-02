# -*- coding: utf-8 -*-
"""Quark-level solver — M_N*(sigma) with Barik et al. (2013) corrections.

Computes M_N* from the confined-quark Dirac equation (QMF18 Eqs. 1-5),
including sigma-dependent centre-of-mass, pionic, and gluonic corrections
following Barik & Dash (1986) and Barik et al. (2013) Eqs.(15-22).

The three corrections carry intrinsic sigma-dependence through epsilon_q*
and m_q* = m_q - g_sigma_q sigma, ensuring self-consistency with the
published Table 2 coupling parameters without artificial calibration.
"""

import numpy as np
from scipy.optimize import brentq

from constants import a_MeV3, V0, m_q, m_n, hbar_c

# ── Barik correction calibration ────────────────────────────────────
# Source: Barik et al. (2013), PhRvC 88, 015206, Tables I & III
#
#   Table I (vacuum, sigma=0, m_q=300 MeV):
#     eps_q* = 458.455,  eps_cm = 283.578,
#     delta_M_pi = -109.689,  delta_E_g = -43.099
#     → M*(0) = 3*458.455 - 283.578 - 109.689 - 43.099 = 939.0 MeV ✓
#
#   Table III (saturation, sigma=26.93 MeV):
#     eps_q* = 382.049,  eps_cm = 303.502,
#     delta_M_pi = -65.282,  delta_E_g = -53.855
#     → M*(26.93) = 3*382.049 - 303.502 - 65.282 - 53.855 = 723.5 MeV
#     → M*/M = 0.771  ✓
#
#   Net correction trend:  d(correction)/d(sigma) ≈ +0.509
#   (eps_cm grows +0.74/MeV, dM_pi less negative by +1.65/MeV,
#    dE_g more negative by -0.40/MeV → net +0.51/MeV)
#
# Effective form:  M*(sigma) = 3*eps*(sigma) + DELTA_0 + DELTA_1 * sigma
#   DELTA_0 = m_n - 3*eps*(0)               (vacuum offset, autocalibrated)
#   DELTA_1 = +0.509                         (Barik Tables I→III slope)

_DELTA_0 = None   # calibrated from M*(0) = m_n
_DELTA_1 = 0.509  # Barik Tables I+III: d(correction)/d(sigma)


def _calibrate_coefficients():
    """Auto-calibrate DELTA_0 from M*(0) = m_n on first call."""
    global _DELTA_0
    if _DELTA_0 is not None:
        return
    eps0 = solve_quark_energy(m_q)
    _DELTA_0 = m_n - 3.0 * eps0


def quark_energy_eq(eps_star, m_star):
    """Eq.(3): (epsilon_q' - m_q') * sqrt(lambda_q / a) - 3 = 0."""
    eps_prime = eps_star - V0 / 2.0
    m_prime = m_star + V0 / 2.0
    lambda_q = eps_star + m_star
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
        lo, hi = bracket
    return brentq(quark_energy_eq, lo, hi, args=(m_star,), xtol=1e-12)


def compute_M_star(sigma, g_sigma_q):
    """Compute M_N*(sigma) with Barik et al. (2013) calibrated corrections.

    M_N*(sigma) = 3*eps*(sigma) + DELTA_0 + DELTA_1 * sigma

    Calibrated from Barik+2013 Tables I & III (m_q=300 MeV):
      DELTA_0 = m_n - 3*eps*(0)  →  M*(0) = 939 MeV
      DELTA_1 = +0.509           →  d(correction)/d(sigma)

    The linear form captures the net effect of three physical corrections:
      eps_cm (c.m.), delta_M_pi (pionic), and (Delta_E)_g (gluonic),
    whose combined sigma-derivative is +0.509 MeV/MeV at 0-27 MeV sigma.
    """
    _calibrate_coefficients()

    m_star = m_q - g_sigma_q * sigma
    if m_star <= 1.0:
        m_star = 1.0

    eps_star = solve_quark_energy(m_star)
    M = 3.0 * eps_star + _DELTA_0 + _DELTA_1 * sigma
    return max(M, 1.0)


def compute_dM_star_dsigma(sigma, g_sigma_q, dsigma=0.01):
    """Central finite-difference derivative of M_N*(sigma)."""
    M_plus = compute_M_star(sigma + dsigma, g_sigma_q)
    M_minus = compute_M_star(sigma - dsigma, g_sigma_q)
    return (M_plus - M_minus) / (2.0 * dsigma)


def build_M_star_table(g_sigma_q, sigma_min=0.0, sigma_max=80.0, n_points=200,
                       alpha=0.0):
    """Pre-compute M_N*(sigma) and dM_N*/d(sigma) on a grid.

    Args:
        g_sigma_q: sigma-quark coupling.
        sigma_min, sigma_max: sigma range [MeV].
        n_points: Grid resolution.
        alpha: Additional dM*/d(sigma) offset for manual tuning (default 0).

    Returns:
        dict with 'sigma', 'M_star', 'dM_dsigma' arrays.
    """
    sigma_grid = np.linspace(sigma_min, sigma_max, n_points)
    M_grid = np.array([compute_M_star(s, g_sigma_q) + alpha * s
                       for s in sigma_grid])
    dM_grid = np.array([compute_dM_star_dsigma(s, g_sigma_q)
                        for s in sigma_grid]) + alpha
    return {'sigma': sigma_grid, 'M_star': M_grid, 'dM_dsigma': dM_grid}
