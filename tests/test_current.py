#!/usr/bin/env python3

"""Regression tests for gauge-covariant current observables.

The tests implement the checks listed in ``non_BTD.md``:

* a stationary state obeys the Hellmann--Feynman band-slope relation;
* the integral of a line-current profile is the charge times velocity;
* a driven state obeys the Ehrenfest and sampled continuity equations.
"""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.linalg import eigh

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from current import (
    charge_continuity_residual,
    integrated_line_current,
    line_current_density,
    wavefunction_and_derivative,
)
from hamiltonian import build_H
from params import AU_TO_NM, L, e, hbar, w_c
from time_coeffs import time_evolute
from reporting import report_check


class CurrentDensityTest(unittest.TestCase):
    NMAX_STATIC = 101
    X_STATIC = np.linspace(-np.sqrt(2) * L, np.sqrt(2) * L, 2001)

    @staticmethod
    def _static_state(ky_nm, ve_scale, level=0):
        """Return a static eigensystem in the units used by build_H."""
        ky = ky_nm * AU_TO_NM
        ve = ve_scale * hbar * w_c / e
        eigenvalues, eigenvectors = eigh(
            build_H(ky, ve, CurrentDensityTest.NMAX_STATIC)
        )
        return ky, ve, eigenvalues, eigenvectors[:, level]

    def test_static_band_slope_equals_integrated_y_current(self):
        """Verify v_y = hbar^-1 dE/dk_y = -(1/e) integral J_y dx."""
        delta_ky = 1.0e-4 * AU_TO_NM

        # Test both the double-well and interface-localized regimes.
        for ky_nm in (-0.05, 0.12):
            with self.subTest(ky_nm=ky_nm):
                ky, ve, eigenvalues, coeffs = self._static_state(ky_nm, 0.2)
                energy_minus = eigh(
                    build_H(ky - delta_ky, ve, self.NMAX_STATIC),
                    eigvals_only=True,
                )[0]
                energy_plus = eigh(
                    build_H(ky + delta_ky, ve, self.NMAX_STATIC),
                    eigvals_only=True,
                )[0]
                velocity_from_band = (energy_plus - energy_minus) / (
                    2 * delta_ky * hbar
                )

                psi, dpsi_dx = wavefunction_and_derivative(coeffs, self.X_STATIC)
                jx_line, jy_line = line_current_density(
                    psi, dpsi_dx, self.X_STATIC, ky
                )
                integrated_jx, integrated_jy = integrated_line_current(
                    jx_line, jy_line, self.X_STATIC
                )

                # A real static eigenvector has no x current.
                self.assertLess(np.max(np.abs(jx_line)), 1.0e-12)
                self.assertLess(abs(integrated_jx), 1.0e-12)
                velocity_from_current = -integrated_jy / e
                relative_error = abs(velocity_from_current - velocity_from_band) / max(
                    abs(velocity_from_band), 1.0e-15
                )
                limit = 2.0e-5
                report_check(
                    f"static current / band-slope relation (ky={ky_nm:+.2f} nm^-1)",
                    "static lowest eigenstate of the production nonuniform-B 2DEG Hamiltonian",
                    relative_error,
                    limit,
                    "relative error",
                )
                np.testing.assert_allclose(
                    velocity_from_current,
                    velocity_from_band,
                    rtol=limit,
                    atol=1.0e-9,
                )

    def test_driven_ehrenfest_and_continuity_residuals(self):
        """Check d<x>/dt = v_x and sampled charge continuity for CFM evolution."""
        nmax = 48
        nsteps = 64
        ky = -0.05 * AU_TO_NM
        ve = 0.2 * hbar * w_c / e
        drive_period = 2 * np.pi / w_c

        coefficients, times = time_evolute(
            ky,
            ve,
            level=0,
            nmax_value=nmax,
            nstep_value=nsteps,
            total_time=0.2 * drive_period,
            eig_k=32,
        )
        x = np.linspace(-np.sqrt(2) * L, np.sqrt(2) * L, 1201)
        psi_t, dpsi_dx_t = wavefunction_and_derivative(coefficients, x)
        jx_line_t, jy_line_t = line_current_density(psi_t, dpsi_dx_t, x, ky)

        # The propagation itself must remain norm preserving.
        spatial_norm = np.trapezoid(np.abs(psi_t) ** 2, x, axis=1)
        self.assertLess(np.max(np.abs(spatial_norm - 1.0)), 2.0e-9)

        integrated_jx, _ = integrated_line_current(jx_line_t, jy_line_t, x)
        x_expectation = np.trapezoid(np.abs(psi_t) ** 2 * x, x, axis=1)
        dx_dt = np.gradient(x_expectation, times, edge_order=2)
        velocity_from_current = -integrated_jx / e
        interior = slice(2, -2)
        velocity_scale = np.max(np.abs(velocity_from_current[interior]))
        relative_ehrenfest_error = np.max(
            np.abs(dx_dt[interior] - velocity_from_current[interior])
        ) / velocity_scale
        ehrenfest_limit = 7.0e-4
        report_check(
            "driven Ehrenfest relation",
            "AC-driven production nonuniform-B 2DEG evolved by the Upsilon_2/Lanczos propagator",
            relative_ehrenfest_error,
            ehrenfest_limit,
            "relative error",
        )
        self.assertLess(relative_ehrenfest_error, ehrenfest_limit)

        residual = charge_continuity_residual(psi_t, jx_line_t, times, x)
        residual_interior = residual[2:-2, 2:-2]
        charge_time_derivative = np.gradient(
            -e * np.abs(psi_t) ** 2, times, axis=0
        )
        current_divergence = np.gradient(jx_line_t, x, axis=1)
        continuity_scale = max(
            np.max(np.abs(charge_time_derivative[2:-2, 2:-2])),
            np.max(np.abs(current_divergence[2:-2, 2:-2])),
        )
        relative_continuity_residual = np.max(np.abs(residual_interior)) / continuity_scale
        continuity_limit = 1.0e-3
        report_check(
            "driven charge continuity equation",
            "same driven 2DEG, sampled on x and t grids",
            relative_continuity_residual,
            continuity_limit,
            "relative residual",
        )
        self.assertLess(relative_continuity_residual, continuity_limit)


if __name__ == "__main__":
    unittest.main()
