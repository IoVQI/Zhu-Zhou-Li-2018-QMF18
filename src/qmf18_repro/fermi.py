"""Fermi integrals — analytic closed forms for zero-temperature nuclear matter.

Eqs.(11-12) kinetic terms: energy density and pressure from nucleon Fermi seas.
"""

import numpy as np


def _fermi_energy_density_integral(kF, M_star):
    """∫₀^{kF} √(k² + M*²) k² dk  (analytic).

    Returns the kinetic energy density contribution for one species.
    """
    if kF <= 0 or M_star <= 0:
        return 0.0
    kF2 = kF * kF
    M2 = M_star * M_star
    Ek = np.sqrt(kF2 + M2)
    return 0.25 * (
        kF * Ek ** 3
        - 0.5 * M2 * kF * Ek
        - 0.5 * M2 * M2 * np.log((kF + Ek) / M_star)
    )


def _fermi_pressure_integral(kF, M_star):
    """∫₀^{kF} k⁴ / √(k² + M*²) dk  (analytic).

    Returns the kinetic pressure contribution for one species.
    """
    if kF <= 0 or M_star <= 0:
        return 0.0
    kF2 = kF * kF
    M2 = M_star * M_star
    Ek = np.sqrt(kF2 + M2)
    return 0.25 * (
        kF2 * kF * Ek
        - 1.5 * M2 * kF * Ek
        + 1.5 * M2 * M2 * np.log((kF + Ek) / M_star)
    )


def kinetic_energy_density(kF_n, kF_p, M_star):
    """Kinetic energy density from neutron + proton Fermi seas.

    Args:
        kF_n, kF_p: Fermi momenta [MeV].
        M_star: Effective nucleon mass [MeV].

    Returns:
        ε_kin [MeV⁴] (multiply by 1/π² in caller).
    """
    en = _fermi_energy_density_integral(kF_n, M_star)
    ep = _fermi_energy_density_integral(kF_p, M_star)
    return en + ep


def kinetic_pressure(kF_n, kF_p, M_star):
    """Kinetic pressure from neutron + proton Fermi seas.

    Args:
        kF_n, kF_p: Fermi momenta [MeV].
        M_star: Effective nucleon mass [MeV].

    Returns:
        P_kin [MeV⁴] (multiply by 1/(3π²) in caller).
    """
    pn = _fermi_pressure_integral(kF_n, M_star)
    pp = _fermi_pressure_integral(kF_p, M_star)
    return pn + pp


def scalar_density(kF, M_star):
    """Scalar density ⟨ψ̄ψ⟩ for one species.

    ∫₀^{kF} M* / √(k²+M*²) k² dk

    Args:
        kF: Fermi momentum [MeV].
        M_star: Effective nucleon mass [MeV].

    Returns:
        ρ_s [MeV³].
    """
    if kF <= 0 or M_star <= 0:
        return 0.0
    kF2 = kF * kF
    M2 = M_star * M_star
    Ek = np.sqrt(kF2 + M2)
    return 0.5 * M_star * (
        kF * Ek - M2 * np.log((kF + Ek) / M_star)
    )


def total_scalar_density(kF_n, kF_p, M_star):
    """Total scalar density for neutrons + protons."""
    return scalar_density(kF_n, M_star) + scalar_density(kF_p, M_star)


def kF_from_density(n):
    """Fermi momentum from number density: k_F = (3π² n)^(1/3)."""
    if n <= 0:
        return 0.0
    return (3.0 * np.pi ** 2 * n) ** (1.0 / 3.0)


def density_from_kF(kF):
    """Number density from Fermi momentum: n = k_F³ / (3π²)."""
    return kF ** 3 / (3.0 * np.pi ** 2)
