# -*- coding: utf-8 -*-
"""Pure-Python TOV solver with tidal deformability.

Solves the Tolman-Oppenheimer-Volkoff equations for neutron star structure
and computes the tidal Love number k2 and deformability Lambda using the
Hinderer (2008) method.

All internal computation in geometrized units (G = c = 1, length in km).

References:
    Tolman (1939), Phys. Rev. 55, 364
    Oppenheimer & Volkoff (1939), Phys. Rev. 55, 374
    Hinderer (2008), ApJ 677, 1216 — tidal deformability
    Damour & Nagar (2009), PRD 80, 084035 — Love number formula
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

from constants import (
    MeVfm3_to_ergcm3, G_cgs, c_cgs, Msun_km,
)


# ── Unit conversion ───────────────────────────────────────────────────
# MeV/fm³ → km⁻²  (geometrized, G=c=1)
_MEVFM3_TO_KM2 = (G_cgs / c_cgs ** 4) * MeVfm3_to_ergcm3 / 1e-10


class TovSolver:
    """Pure-Python TOV + tidal deformability solver.

    Uses RK4 integration with the Hinderer (2008) tidal ODE.

    Parameters
    ----------
    dr : float
        Integration step size [km].  Default 0.001 (1 m).
    rs : float
        Starting radius [km].  Default 1e-5.
    max_steps : int
        Maximum integration steps.  Default 200000.
    """

    def __init__(self, dr=0.001, rs=1e-5, max_steps=200000):
        self.dr = dr
        self.rs = rs
        self.max_steps = max_steps

    # ──────────────────────────────────────────────────────────────────
    #  EOS preparation
    # ──────────────────────────────────────────────────────────────────

    def _build_eos_interpolators(self, nB, eps, P):
        """Build all EOS interpolators from the raw arrays.

        Parameters
        ----------
        nB, eps, P : ndarray
            EOS arrays from eos.csv (nB in fm⁻³, eps and P in MeV/fm³).

        Returns
        -------
        P_of_eps : PchipInterpolator
            P(eps) in km⁻², domain in km⁻².
        eps_of_nB : PchipInterpolator
            eps(nB) in MeV/fm³, domain in fm⁻³.
        eps_of_P : PchipInterpolator
            eps(P) in km⁻², domain in km⁻².  (Inverse EOS for fast lookup.)
        """
        # Convert to geometrized units — sort by eps
        sort_idx = np.argsort(eps)
        eps_sorted = eps[sort_idx]
        P_sorted = P[sort_idx]

        eps_km2 = eps_sorted * _MEVFM3_TO_KM2
        P_km2 = P_sorted * _MEVFM3_TO_KM2

        # Deduplicate (PCHIP requires strictly monotonic x)
        mask_e = np.concatenate([[True], np.diff(eps_km2) > 0])
        P_of_eps = PchipInterpolator(eps_km2[mask_e], P_km2[mask_e])

        # Inverse: eps(P) — P is also monotonic in the core
        mask_p = np.concatenate([[True], np.diff(P_km2) > 0])
        eps_of_P = PchipInterpolator(P_km2[mask_p], eps_km2[mask_p])

        # eps(nB) for rho_c lookup
        order_nB = np.argsort(nB)
        nB_s = nB[order_nB]
        eps_s = eps[order_nB]
        mask_n = np.concatenate([[True], np.diff(nB_s) > 0])
        eps_of_nB = PchipInterpolator(nB_s[mask_n], eps_s[mask_n])

        return P_of_eps, eps_of_nB, eps_of_P

    # ──────────────────────────────────────────────────────────────────
    #  TOV + tidal RHS
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _rhs(r, m, P, y, eps, dP_deps):
        """Right-hand side of the coupled TOV + tidal ODE system.

        State: [m, P, y] where y = r·H'/H is the tidal perturbation.

        Parameters
        ----------
        r : float — radius [km]
        m : float — mass [km]
        P : float — pressure [km⁻²]
        y : float — tidal variable
        eps : float — energy density [km⁻²]
        dP_deps : float — sound speed squared (dimensionless)

        Returns
        -------
        [dm_dr, dP_dr, dy_dr] or None if integration should stop.
        """
        if P <= 0 or r <= 0 or dP_deps <= 0:
            return None

        r2 = r * r
        r3 = r2 * r

        # Mass equation
        dm_dr = 4.0 * np.pi * r2 * eps

        # TOV equation
        denom = r * (r - 2.0 * m)
        if abs(denom) < 1e-30:
            return None
        dP_dr = -(eps + P) * (m + 4.0 * np.pi * r3 * P) / denom

        # Tidal ODE (Hinderer 2008, Eq. 14-17)
        one_m_2m_r = 1.0 - 2.0 * m / r
        if abs(one_m_2m_r) < 1e-12:
            return None

        beta = (m + 4.0 * np.pi * r3 * P) / (r2 * one_m_2m_r)

        F = 2.0 / one_m_2m_r + r * beta * (eps - P) / (eps + P)

        G = (6.0 / one_m_2m_r
             - 4.0 * np.pi * r2
             * (5.0 * eps + 9.0 * P + (eps + P) / dP_deps)
             / one_m_2m_r
             + r2 * beta * beta * (1.0 + 1.0 / dP_deps))

        dy_dr = -(y * y + F * y + G) / r

        return [dm_dr, dP_dr, dy_dr]

    # ──────────────────────────────────────────────────────────────────
    #  Single-star integration
    # ──────────────────────────────────────────────────────────────────

    def _integrate_star(self, eps_c_km2, P_c_km2, P_of_eps, eps_of_P):
        """Integrate TOV + tidal from center to surface.

        Parameters
        ----------
        eps_c_km2 : float — central energy density [km⁻²]
        P_c_km2 : float — central pressure [km⁻²]
        P_of_eps, eps_of_P : EOS interpolators (geometrized units)

        Returns
        -------
        (R, M, y_R) — surface radius [km], mass [km], tidal variable y
        or None if integration failed.
        """
        dr = self.dr
        rs = self.rs
        max_steps = self.max_steps

        # PCHIP derivative for sound speed
        dP_of_deps = P_of_eps.derivative()

        # Initial conditions at r = rs
        r = rs
        m = (4.0 / 3.0) * np.pi * rs ** 3 * eps_c_km2
        P = P_c_km2 - (2.0 / 3.0) * np.pi * rs ** 2 \
            * (eps_c_km2 + P_c_km2) * (eps_c_km2 + 3.0 * P_c_km2)
        if P <= 0:
            P = P_c_km2 * 0.99
        y = 2.0  # analytic limit at center

        for _ in range(max_steps):
            if P <= 0:
                # Already at or past surface
                return r, m, y

            # Look up eps and dP/deps from P (inverse EOS)
            eps = float(eps_of_P(P))
            if eps <= 0:
                return None
            dP_deps = float(dP_of_deps(eps))
            if dP_deps <= 0:
                return None

            # k1
            k1 = self._rhs(r, m, P, y, eps, dP_deps)
            if k1 is None:
                return r, m, y

            # k2
            r_m = r + 0.5 * dr
            m2 = m + 0.5 * dr * k1[0]
            P2 = P + 0.5 * dr * k1[1]
            if P2 <= 0:
                return r + dr, m + dr * k1[0], y
            y2 = y + 0.5 * dr * k1[2]
            eps2 = float(eps_of_P(P2))
            dP2 = float(dP_of_deps(eps2))
            k2 = self._rhs(r_m, m2, P2, y2, eps2, dP2)
            if k2 is None:
                return r + dr, m + dr * k1[0], y

            # k3
            m3 = m + 0.5 * dr * k2[0]
            P3 = P + 0.5 * dr * k2[1]
            if P3 <= 0:
                return r + dr, m + dr * k2[0], y
            y3 = y + 0.5 * dr * k2[2]
            eps3 = float(eps_of_P(P3))
            dP3 = float(dP_of_deps(eps3))
            k3 = self._rhs(r_m, m3, P3, y3, eps3, dP3)
            if k3 is None:
                return r + dr, m + dr * k2[0], y

            # k4
            r4 = r + dr
            m4 = m + dr * k3[0]
            P4 = P + dr * k3[1]
            if P4 <= 0:
                return r4, m4, y + dr * k3[2]
            y4 = y + dr * k3[2]
            eps4 = float(eps_of_P(P4))
            dP4 = float(dP_of_deps(eps4))
            k4 = self._rhs(r4, m4, P4, y4, eps4, dP4)
            if k4 is None:
                return r4, m4, y + dr * k3[2]

            # Update
            m += (dr / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            P += (dr / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            y += (dr / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
            r = r4

        # Did not reach surface within max_steps
        return None

    # ──────────────────────────────────────────────────────────────────
    #  Batch solver (drop-in for CppTovSolver.solve_tidal_batch)
    # ──────────────────────────────────────────────────────────────────

    def solve_tidal_batch(self, nB, P, eps, rho_c_grid):
        """Solve TOV + tidal for a grid of central densities.

        Parameters
        ----------
        nB, P, eps : ndarray
            EOS arrays from eos.csv (nB in fm⁻³, eps and P in MeV/fm³).
        rho_c_grid : ndarray
            Central baryon densities [fm⁻³].

        Returns
        -------
        dict with arrays: success, M_Msun, R_km, Lambda, k2, C
        """
        P_of_eps, eps_of_nB, eps_of_P = self._build_eos_interpolators(nB, eps, P)

        n = len(rho_c_grid)
        success = np.zeros(n, dtype=bool)
        M_Msun = np.zeros(n)
        R_km = np.zeros(n)
        Lambda_arr = np.zeros(n)
        k2_arr = np.zeros(n)
        C_arr = np.zeros(n)

        for i, rho_c in enumerate(rho_c_grid):
            result = self._solve_one(rho_c, eps_of_nB, P_of_eps, eps_of_P)
            success[i] = result['success']
            M_Msun[i] = result['M_Msun']
            R_km[i] = result['R_km']
            Lambda_arr[i] = result['Lambda']
            k2_arr[i] = result['k2']
            C_arr[i] = result['C']

        return {
            'success': success,
            'M_Msun': M_Msun,
            'R_km': R_km,
            'Lambda': Lambda_arr,
            'k2': k2_arr,
            'C': C_arr,
        }

    def _solve_one(self, rho_c_fm3, eps_of_nB, P_of_eps, eps_of_P):
        """Solve one star with central baryon density rho_c_fm3."""
        nB_min = float(eps_of_nB.x[0])
        nB_max = float(eps_of_nB.x[-1])
        if rho_c_fm3 < nB_min or rho_c_fm3 > nB_max:
            return self._fail_result()

        # Central energy density from baryon density
        eps_c_MeVfm3 = float(eps_of_nB(rho_c_fm3))
        eps_c_km2 = eps_c_MeVfm3 * _MEVFM3_TO_KM2

        # Central pressure from energy density
        P_c_km2 = float(P_of_eps(eps_c_km2))
        if P_c_km2 <= 0:
            return self._fail_result()

        # Integrate TOV + tidal
        result = self._integrate_star(eps_c_km2, P_c_km2, P_of_eps, eps_of_P)
        if result is None:
            return self._fail_result()

        R, M, y_R = result

        if R <= 0 or M <= 0:
            return self._fail_result()

        C = M / R  # compactness
        k2_val = self._compute_k2(y_R, C)
        Lambda_val = (2.0 / 3.0) * k2_val / C ** 5 if C > 0 else 0.0

        return {
            'success': True,
            'M_Msun': M / Msun_km,  # km → M☉
            'R_km': R,
            'Lambda': Lambda_val,
            'k2': k2_val,
            'C': C,
        }

    # ──────────────────────────────────────────────────────────────────
    #  Love number k2
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_k2(y_R, C):
        """Compute Love number k2 from surface y and compactness C.

        Hinderer (2008), ApJ 677, 1216, Eq.(84):
        k2 = (8/5) C^5 (1-2C)^2 [2-y + 2C(y-1)] / A
        where A = 2C [6-3y+3C(5y-8)]
                 + 4C^3 [13-11y+C(3y-2)+2C^2(1+y)]
                 + 3(1-2C)^2 [2-y+2C(y-1)] ln(1-2C)
        """
        C2 = C * C
        C3 = C2 * C
        C4 = C3 * C
        C5 = C4 * C
        one_m_2C = 1.0 - 2.0 * C

        if one_m_2C <= 0:
            return 0.0

        bracket = 2.0 - y_R + 2.0 * C * (y_R - 1.0)
        numerator = (8.0 / 5.0) * C5 * one_m_2C ** 2 * bracket

        A = (2.0 * C * (6.0 - 3.0 * y_R + 3.0 * C * (5.0 * y_R - 8.0))
             + 4.0 * C3 * (13.0 - 11.0 * y_R
                           + C * (3.0 * y_R - 2.0)
                           + 2.0 * C2 * (1.0 + y_R))
             + 3.0 * one_m_2C ** 2 * bracket * np.log(one_m_2C))

        if abs(A) < 1e-30:
            return 0.0

        return numerator / A

    # ──────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _fail_result():
        """Return a failed-star result dict."""
        return {
            'success': False,
            'M_Msun': 0.0,
            'R_km': 0.0,
            'Lambda': 0.0,
            'k2': 0.0,
            'C': 0.0,
        }
