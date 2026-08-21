#!/usr/bin/env python3

"""Create visual diagnostics for the production 2DEG current tests.

This is deliberately separate from the automated tests: the test suite stays
fast and non-interactive, while this script leaves a figure a person can read.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from current import (  # noqa: E402
    charge_continuity_residual,
    integrated_line_current,
    line_current_density,
    wavefunction_and_derivative,
)
from hamiltonian import build_H  # noqa: E402
from params import AU_TO_NM, L, e, hbar, w_c  # noqa: E402
from time_coeffs import time_evolute  # noqa: E402


def main():
    nmax_static = 101
    x_static = np.linspace(-np.sqrt(2) * L, np.sqrt(2) * L, 2001)
    delta_ky = 1.0e-4 * AU_TO_NM
    ve = 0.2 * hbar * w_c / e
    ky_nm_values = np.array([-0.05, 0.12])
    slope_velocity = []
    current_velocity = []

    for ky_nm in ky_nm_values:
        ky = ky_nm * AU_TO_NM
        _, eigenvectors = eigh(build_H(ky, ve, nmax_static))
        energy_minus = eigh(build_H(ky - delta_ky, ve, nmax_static), eigvals_only=True)[0]
        energy_plus = eigh(build_H(ky + delta_ky, ve, nmax_static), eigvals_only=True)[0]
        slope_velocity.append((energy_plus - energy_minus) / (2 * delta_ky * hbar))
        psi, dpsi_dx = wavefunction_and_derivative(eigenvectors[:, 0], x_static)
        jx, jy = line_current_density(psi, dpsi_dx, x_static, ky)
        _, total_jy = integrated_line_current(jx, jy, x_static)
        current_velocity.append(-total_jy / e)

    nmax = 48
    ky = -0.05 * AU_TO_NM
    drive_period = 2 * np.pi / w_c
    coefficients, times = time_evolute(
        ky, ve, level=0, nmax_value=nmax, nstep_value=64,
        total_time=0.2 * drive_period, eig_k=32,
    )
    x = np.linspace(-np.sqrt(2) * L, np.sqrt(2) * L, 1201)
    psi_t, dpsi_dx_t = wavefunction_and_derivative(coefficients, x)
    jx_t, _ = line_current_density(psi_t, dpsi_dx_t, x, ky)
    total_jx, _ = integrated_line_current(jx_t, jx_t * 0, x)
    x_mean = np.trapezoid(np.abs(psi_t) ** 2 * x, x, axis=1)
    dx_dt = np.gradient(x_mean, times, edge_order=2)
    current_velocity_x = -total_jx / e
    residual = charge_continuity_residual(psi_t, jx_t, times, x)
    scale = max(np.max(np.abs(np.gradient(-e * np.abs(psi_t) ** 2, times, axis=0))), 1e-30)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    axes[0].plot(ky_nm_values, slope_velocity, "o-", label=r"$\hbar^{-1}dE/dk_y$")
    axes[0].plot(ky_nm_values, current_velocity, "s--", label=r"$-\int J_y dx/e$")
    axes[0].set(title="Static 2DEG: band slope = current", xlabel=r"$k_y$ (nm$^{-1}$)", ylabel="velocity (a.u.)")
    axes[0].legend(fontsize=8)

    time_cycles = times / drive_period
    axes[1].plot(time_cycles, dx_dt, label=r"$d\langle x\rangle/dt$")
    axes[1].plot(time_cycles, current_velocity_x, "--", label=r"$-\int J_x dx/e$")
    axes[1].set(title="Driven 2DEG: Ehrenfest relation", xlabel="time / drive period", ylabel="velocity (a.u.)")
    axes[1].legend(fontsize=8)

    image = axes[2].imshow(
        np.abs(residual / scale).T, aspect="auto", origin="lower",
        extent=(time_cycles[0], time_cycles[-1], x[0], x[-1]), cmap="magma",
    )
    axes[2].set(title="Driven 2DEG: continuity residual", xlabel="time / drive period", ylabel="x (a.u.)")
    fig.colorbar(image, ax=axes[2], label="relative residual")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Current-validation models and numerical checks", y=1.03)
    fig.tight_layout()

    output_dir = Path(__file__).resolve().parent / "artifacts"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "current_validation.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
