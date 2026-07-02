"""Physical constants and unit conversions — QMF18 reproduction.

Natural units: hbar = c = 1, [E] = MeV, [L] = MeV^-1.
Geometrized units: G = c = 1, [M] = km, [L] = km.
"""

import numpy as np

# ── Fundamental ──────────────────────────────────────────────────────
hbar_c = 197.3269804          # MeV·fm
MeV_to_erg = 1.602176634e-6   # erg/MeV
fm_to_cm = 1.0e-13            # cm/fm
fm3_to_cm3 = fm_to_cm ** 3    # cm³/fm³  (= 1e-39)
MeVfm3_to_ergcm3 = MeV_to_erg / fm3_to_cm3  # ≈ 1.6022e33 erg/cm³ per MeV/fm³

# ── Nuclear / particle masses ────────────────────────────────────────
m_n = 939.565379       # neutron mass [MeV]
m_p = 938.272046       # proton mass [MeV]
m_e = 0.510998950      # electron mass [MeV]
m_mu = 105.6583745     # muon mass [MeV]

# ── QMF model fixed parameters ──────────────────────────────────────
m_q = 300.0            # constituent quark mass [MeV]
m_sigma = 510.0        # σ meson mass [MeV]
m_omega = 783.0        # ω meson mass [MeV]
m_rho = 770.0          # ρ meson mass [MeV]

# ── Quark confining potential (fitted to vacuum nucleon) ──────────
a_fm3 = 0.534296       # confining strength [fm⁻³]
a_MeV3 = a_fm3 * (hbar_c ** 3)  # confining strength [MeV³]
V0 = -62.257187         # confining depth [MeV]

# ── Saturation properties (Table 1) ─────────────────────────────────
rho0_fm3 = 0.16         # saturation density [fm⁻³]
EA_sat = -16.0          # binding energy per nucleon [MeV]
K_sat = 240.0           # incompressibility [MeV]
Esym_sat = 31.0         # symmetry energy [MeV]
Ms_ratio_sat = 0.77     # M*/M at saturation

# ── Coupling parameters (Table 2) ────────────────────────────────────
# For L = 40 MeV (QMF18 best model)
COUPLING = {
    'L': 40.0,
    'g_sigma_q': 3.8620366,
    'g_omega_q': 2.9174838,
    'g_rho_q': 5.4129448,
    'g2': 14.6179599,         # fm⁻¹ → convert to MeV in use
    'g3': -66.3442468,
    'Lambda_v': 0.7693664,
}

# Quark counting rules
COUPLING['g_omega_N'] = 3.0 * COUPLING['g_omega_q']
COUPLING['g_rho_N'] = COUPLING['g_rho_q']

# All four L values (Table 2)
COUPLING_ALL = {
    20: {'g_sigma_q': 3.8620366, 'g_omega_q': 2.9174838, 'g_rho_q': 6.9588083,
         'g2': 14.6179599, 'g3': -66.3442468, 'Lambda_v': 1.1080665,
         'g_omega_N': 8.7524514, 'g_rho_N': 6.9588083},
    40: {'g_sigma_q': 3.8620366, 'g_omega_q': 2.9174838, 'g_rho_q': 5.4129448,
         'g2': 14.6179599, 'g3': -66.3442468, 'Lambda_v': 0.7693664,
         'g_omega_N': 8.7524514, 'g_rho_N': 5.4129448},
    60: {'g_sigma_q': 3.8620366, 'g_omega_q': 2.9174838, 'g_rho_q': 4.5830609,
         'g2': 14.6179599, 'g3': -66.3442468, 'Lambda_v': 0.4306662,
         'g_omega_N': 8.7524514, 'g_rho_N': 4.5830609},
    80: {'g_sigma_q': 3.8620366, 'g_omega_q': 2.9174838, 'g_rho_q': 4.0459574,
         'g2': 14.6179599, 'g3': -66.3442468, 'Lambda_v': 0.0919661,
         'g_omega_N': 8.7524514, 'g_rho_N': 4.0459574},
}

# ── Gravitational ───────────────────────────────────────────────────
G_cgs = 6.6730831e-8           # cm³/(g·s²)
c_cgs = 2.99792458e10          # cm/s
Msun_g = 1.988435e33           # g
Msun_km = G_cgs * Msun_g / (c_cgs ** 2) / 1e5  # ≈ 1.4766 km/M⊙
km_to_Msun = 1.0 / Msun_km

# MeV/fm³ → km⁻² (geometrized units)
_MEVFM3_TO_KM2 = (G_cgs / c_cgs ** 4) * MeVfm3_to_ergcm3 / 1e-10
_KM2_TO_MEVFM3 = 1.0 / _MEVFM3_TO_KM2


def mevfm3_to_km2(value):
    """Pressure/energy density: MeV/fm³ → km⁻²."""
    return value * _MEVFM3_TO_KM2


def km2_to_mevfm3(value):
    """Pressure/energy density: km⁻² → MeV/fm³."""
    return value * _KM2_TO_MEVFM3


def km_to_Msun(M_km):
    """Mass: km → M⊙."""
    return M_km * km_to_Msun


def Msun_to_km(M_Msun):
    """Mass: M⊙ → km."""
    return M_Msun * Msun_km


# ── Reference values (Table 4 benchmarks) ───────────────────────────
BENCHMARKS = {
    'M_TOV': 2.0805,      # M⊙ (Table 3, non-interpolated)
    'M_TOV_full': 2.0809, # M⊙ (complete data)
    'R_14': 11.77,        # km
    'MR_14': 0.1756,
    'Lambda_14': 331,
    'Lambda_tilde': (381.4, 388.4),  # q=0.7-1
}
