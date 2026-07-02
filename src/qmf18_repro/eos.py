# -*- coding: utf-8 -*-
"""EOS computation — beta-equilibrium neutron star matter (QMF18 Eq. 13).

Solves mu_n = mu_e + mu_p (beta-equilibrium) and n_p = n_e + n_mu (charge
neutrality) at each baryon density via iterative Y_p adjustment.
"""

import numpy as np

from constants import hbar_c, m_e, m_mu
from fermi import kF_from_density
from fields import solve_meson_fields, compute_energy_pressure


def chemical_potential_nucleon(kF, M_star, g_omega_N, g_rho_N, omega, rho_m, tau3):
    """Nucleon chemical potential: mu_i = sqrt(kF^2 + M*^2) + g_omega omega + g_rho tau3 rho."""
    return np.sqrt(kF ** 2 + M_star ** 2) + g_omega_N * omega + g_rho_N * tau3 * rho_m


def compute_eos_point(rho_B, coupling, M_star_fn, dM_dsigma_fn, Y_p0=0.05):
    """Compute one EOS point at given density with beta-equilibrium.

    Iterative method: adjust Y_p until charge neutrality n_p = n_e + n_mu
    under the beta-equilibrium condition mu_e = mu_n - mu_p.

    Args:
        rho_B: Baryon density [fm^-3].
        coupling: Coupling dict.
        M_star_fn, dM_dsigma_fn: M*(sigma) functions.
        Y_p0: Initial Y_p guess.

    Returns:
        dict: {rho_B, Y_p, eps, P, M_star, sigma, omega, rho_meson,
               mu_n, mu_e, converged}
    """
    rho_B_MeV3 = rho_B * (hbar_c ** 3)
    g_omega_N = coupling['g_omega_N']
    g_rho_N = coupling['g_rho_N']
    Y_p = Y_p0

    for iteration in range(40):
        fields = solve_meson_fields(rho_B, Y_p, coupling, M_star_fn, dM_dsigma_fn)
        if not fields['converged']:
            return {'rho_B': rho_B, 'Y_p': Y_p, 'eps': 0.0, 'P': 0.0,
                    'converged': False}

        M_star = fields['M_star']
        omega = fields['omega']
        rho_m = fields['rho_meson']

        n_n = (1.0 - Y_p) * rho_B_MeV3
        n_p = Y_p * rho_B_MeV3
        kF_n = kF_from_density(n_n)
        kF_p = kF_from_density(n_p)

        mu_n = chemical_potential_nucleon(kF_n, M_star, g_omega_N, g_rho_N, omega, rho_m, -0.5)
        mu_p = chemical_potential_nucleon(kF_p, M_star, g_omega_N, g_rho_N, omega, rho_m, +0.5)
        mu_e = mu_n - mu_p

        # Lepton densities at beta-equilibrium
        if mu_e > m_e:
            kF_e = np.sqrt(mu_e ** 2 - m_e ** 2)
            n_e = kF_e ** 3 / (3.0 * np.pi ** 2)
        else:
            n_e = 0.0

        if mu_e > m_mu:
            kF_mu = np.sqrt(mu_e ** 2 - m_mu ** 2)
            n_mu = kF_mu ** 3 / (3.0 * np.pi ** 2)
        else:
            n_mu = 0.0

        Y_p_new = (n_e + n_mu) / rho_B_MeV3

        if abs(Y_p_new - Y_p) < 1e-8:
            Y_p = Y_p_new
            break

        Y_p = 0.5 * Y_p + 0.5 * Y_p_new

    # Final solution
    fields = solve_meson_fields(rho_B, Y_p, coupling, M_star_fn, dM_dsigma_fn)
    if not fields['converged']:
        return {'rho_B': rho_B, 'Y_p': Y_p, 'eps': 0.0, 'P': 0.0, 'converged': False}

    eps, P = compute_energy_pressure(rho_B, Y_p, fields, coupling)

    # Diagnostics
    n_n_f = (1.0 - Y_p) * rho_B_MeV3
    n_p_f = Y_p * rho_B_MeV3
    kF_n_f = kF_from_density(n_n_f)
    kF_p_f = kF_from_density(n_p_f)
    mu_n_f = chemical_potential_nucleon(kF_n_f, fields['M_star'], g_omega_N, g_rho_N,
                                        fields['omega'], fields['rho_meson'], -0.5)
    mu_p_f = chemical_potential_nucleon(kF_p_f, fields['M_star'], g_omega_N, g_rho_N,
                                        fields['omega'], fields['rho_meson'], +0.5)

    return {
        'rho_B': rho_B, 'Y_p': Y_p, 'eps': eps, 'P': P,
        'M_star': fields['M_star'], 'sigma': fields['sigma'],
        'omega': fields['omega'], 'rho_meson': fields['rho_meson'],
        'mu_n': mu_n_f, 'mu_e': mu_n_f - mu_p_f, 'converged': True,
    }


def generate_eos_table(rho_B_array, coupling, M_star_fn, dM_dsigma_fn, verbose=True):
    """Generate EOS table over a grid of baryon densities.

    Carries Y_p from the previous density point as the initial guess,
    improving convergence at high density.
    """
    results = []
    Y_p_prev = 0.05  # starting guess at lowest density
    for rho_B in rho_B_array:
        if verbose:
            print(f"  rho_B = {rho_B:.3f} fm^-3 ...", end=" ")
        point = compute_eos_point(rho_B, coupling, M_star_fn, dM_dsigma_fn,
                                  Y_p0=Y_p_prev)
        if point['converged']:
            Y_p_prev = point['Y_p']  # carry forward
            if verbose:
                print(f"Y_p={point['Y_p']:.4f}, P={point['P']:.4f} MeV/fm^3")
        else:
            if verbose:
                print("FAILED")
        results.append(point)
    return results
