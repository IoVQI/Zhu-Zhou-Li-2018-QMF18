# -*- coding: utf-8 -*-
"""Meson field equations — self-consistent solution of sigma, omega, rho (QMF18 Eqs. 8-10).

Uses scipy.optimize.fsolve for the 3-field coupled system.
"""

import numpy as np
from scipy.optimize import least_squares

from constants import m_sigma, m_omega, m_rho, hbar_c
from fermi import kF_from_density, scalar_density


def _field_residuals(x, kF_n, kF_p, coupling, M_star_fn, dM_dsigma_fn):
    """Residuals for the 3-field system: {sigma, omega, rho}_meson.

    Args:
        x: [sigma, omega, rho_meson] in MeV.
        kF_n, kF_p: Fermi momenta [MeV].
        coupling: Dict of coupling parameters.
        M_star_fn: sigma -> M_N*.
        dM_dsigma_fn: sigma -> dM_N*/dsigma.

    Returns:
        [res_sigma, res_omega, res_rho] — zeros give self-consistent solution.
    """
    sigma, omega, rho_m = x

    # Physical bound: m_q* = m_q - g_sigma_q*sigma must stay > ~5 MeV
    sigma_max = 300.0 / coupling['g_sigma_q'] - 1.5  # ~76.2
    if sigma < 0 or sigma > sigma_max:
        return [1e10 * (1 + abs(sigma)), 1e10, 1e10]

    M_star = M_star_fn(sigma)
    if M_star <= 0 or M_star < 10.0:
        return [1e10 * (1 + abs(sigma)), 1e10, 1e10]

    g_sigma_q = coupling['g_sigma_q']
    g_omega_N = coupling['g_omega_N']
    g_rho_N = coupling['g_rho_N']
    g2_MeV = coupling['g2'] * hbar_c   # fm^-1 -> MeV
    g3 = coupling['g3']
    Lambda_v = coupling['Lambda_v']

    # Scalar densities — source term ⟨ψ̄ψ⟩ = integral / π²
    rho_s_n = scalar_density(kF_n, M_star) / (np.pi ** 2)
    rho_s_p = scalar_density(kF_p, M_star) / (np.pi ** 2)
    rho_s = rho_s_n + rho_s_p

    # Vector densities
    n_n = kF_n ** 3 / (3.0 * np.pi ** 2)
    n_p = kF_p ** 3 / (3.0 * np.pi ** 2)
    rho_B_MeV3 = n_n + n_p

    dM_dsigma = dM_dsigma_fn(sigma)

    # Eq.(8): m_sigma^2 sigma + g2 sigma^2 + g3 sigma^3 + dM*/dsigma rho_s = 0
    res_sigma = (m_sigma ** 2 * sigma + g2_MeV * sigma ** 2
                 + g3 * sigma ** 3 + dM_dsigma * rho_s)

    # Eq.(9): m_omega^2 omega + Lambda_v g_omegaN^2 g_rhoN^2 omega rho^2 - g_omegaN rho_B = 0
    res_omega = (m_omega ** 2 * omega
                 + Lambda_v * g_omega_N ** 2 * g_rho_N ** 2 * omega * rho_m ** 2
                 - g_omega_N * rho_B_MeV3)

    # Eq.(10): m_rho^2 rho + Lambda_v g_rhoN^2 g_omegaN^2 rho omega^2 - g_rhoN (n_p - n_n) = 0
    res_rho = (m_rho ** 2 * rho_m
               + Lambda_v * g_rho_N ** 2 * g_omega_N ** 2 * rho_m * omega ** 2
               - g_rho_N * (n_p - n_n))

    return [res_sigma, res_omega, res_rho]


def solve_meson_fields(rho_B_fm3, Y_p, coupling, M_star_fn, dM_dsigma_fn,
                       x0=None, maxfev=400):
    """Self-consistently solve sigma, omega, rho field equations.

    Args:
        rho_B_fm3: Baryon density [fm^-3].
        Y_p: Proton fraction.
        coupling: Dict with g_sigma_q, g_omega_N, g_rho_N, g2, g3, Lambda_v.
        M_star_fn: callable sigma[MeV] -> M_N*[MeV].
        dM_dsigma_fn: callable sigma[MeV] -> dM_N*/dsigma.
        x0: Initial guess [sigma, omega, rho] in MeV. Auto if None.
        maxfev: Max fsolve evaluations.

    Returns:
        dict: {sigma, omega, rho_meson, M_star, converged, n_iter}.
    """
    rho_B_MeV3 = rho_B_fm3 * (hbar_c ** 3)
    n_n_MeV3 = (1.0 - Y_p) * rho_B_MeV3
    n_p_MeV3 = Y_p * rho_B_MeV3
    kF_n = kF_from_density(n_n_MeV3)
    kF_p = kF_from_density(n_p_MeV3)

    # Initial guess — physically motivated, clamped to bounds
    if x0 is None:
        # omega ≈ g_omega_N * rho_B / m_omega^2
        omega0 = coupling['g_omega_N'] * rho_B_MeV3 / (m_omega ** 2)
        # sigma ≈ 25 * (rho/rho0)^0.7, clamped to [10, sigma_bound-5]
        sigma0 = 25.0 * (rho_B_fm3 / 0.16) ** 0.7
        sigma_bound = 300.0 / coupling['g_sigma_q'] - 2.0
        sigma0 = max(10.0, min(sigma0, sigma_bound - 5.0))
        # rho meson from source term
        rho0 = coupling['g_rho_N'] * (n_p_MeV3 - n_n_MeV3) / (m_rho ** 2)
        x0 = [sigma0, omega0, rho0]

    # Bounds: sigma in [1, sigma_max], omega in [0, 500], rho in [-200, 200]
    # sigma_max: m_q* = m_q - g_sigma_q*sigma > 0 → sigma < m_q/g_sigma_q ≈ 77.7
    sigma_bound = 300.0 / coupling['g_sigma_q'] - 2.0  # ~75.7, allows m* > ~8 MeV
    bounds = ([1.0, 0.0, -300.0], [sigma_bound, 500.0, 300.0])

    try:
        sol = least_squares(
            lambda x: _field_residuals(x, kF_n, kF_p, coupling, M_star_fn, dM_dsigma_fn),
            x0,
            bounds=bounds,
            max_nfev=maxfev,
            xtol=1e-10,
            ftol=1e-10,
            method='trf',
        )
        sigma, omega, rho_m = sol.x
        converged = sol.success

        M_star = M_star_fn(sigma) if converged else 0.0

        return {
            'sigma': sigma,
            'omega': omega,
            'rho_meson': rho_m,
            'M_star': M_star,
            'converged': converged,
            'n_iter': sol.nfev,
            'message': sol.message if not converged else 'OK',
        }
    except Exception as exc:
        return {
            'sigma': 0.0, 'omega': 0.0, 'rho_meson': 0.0,
            'M_star': 0.0, 'converged': False, 'n_iter': -1,
            'message': str(exc),
        }


def compute_energy_pressure(rho_B_fm3, Y_p, fields, coupling):
    """Compute energy density and pressure from field solution (Eqs. 11-12).

    Args:
        rho_B_fm3: Baryon density [fm^-3].
        Y_p: Proton fraction.
        fields: Dict from solve_meson_fields().
        coupling: Coupling dict.

    Returns:
        (energy_density, pressure) in [MeV/fm^3].
    """
    from fermi import kinetic_energy_density, kinetic_pressure

    rho_B_MeV3 = rho_B_fm3 * (hbar_c ** 3)
    n_n_MeV3 = (1.0 - Y_p) * rho_B_MeV3
    n_p_MeV3 = Y_p * rho_B_MeV3
    kF_n = kF_from_density(n_n_MeV3)
    kF_p = kF_from_density(n_p_MeV3)

    sigma = fields['sigma']
    omega = fields['omega']
    rho_m = fields['rho_meson']
    M_star = fields['M_star']

    g_omega_N = coupling['g_omega_N']
    g_rho_N = coupling['g_rho_N']
    g2_MeV = coupling['g2'] * hbar_c
    g3 = coupling['g3']
    Lambda_v = coupling['Lambda_v']

    # Kinetic contributions (MeV^4)
    eps_kin = kinetic_energy_density(kF_n, kF_p, M_star) / (np.pi ** 2)
    P_kin = kinetic_pressure(kF_n, kF_p, M_star) / (3.0 * np.pi ** 2)

    # Potential terms
    U_sigma = 0.5 * m_sigma ** 2 * sigma ** 2 + (1.0/3.0) * g2_MeV * sigma ** 3 + 0.25 * g3 * sigma ** 4
    U_omega = 0.5 * m_omega ** 2 * omega ** 2
    U_rho = 0.5 * m_rho ** 2 * rho_m ** 2
    U_omega_rho = 0.5 * Lambda_v * g_rho_N ** 2 * g_omega_N ** 2 * rho_m ** 2 * omega ** 2

    # Energy density (Eq.11): omega-rho term has coefficient 3/2
    eps_MeV4 = eps_kin + U_sigma + U_omega + U_rho + 3.0 * U_omega_rho

    # Pressure (Eq.12): omega-rho term has coefficient 1/2
    P_MeV4 = P_kin - U_sigma + U_omega + U_rho + U_omega_rho

    # Convert MeV^4 -> MeV/fm^3
    eps = eps_MeV4 / (hbar_c ** 3)
    P = P_MeV4 / (hbar_c ** 3)

    return eps, P
