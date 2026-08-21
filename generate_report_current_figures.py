"""Generate current-density validation figures for non_BTD.md.

The numerical setup matches tests/plot_current_validation.py, but the styling
is shared with all report figures through report_plotsetting.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from current import (
    charge_continuity_residual,
    integrated_line_current,
    line_current_density,
    wavefunction_and_derivative,
)
from hamiltonian import build_H
from params import AU_TO_NM, L, e, hbar, w_c
from report_plotsetting import REPORT_COLORS, use_report_style
from time_coeffs import time_evolute


OUT_DIR = Path("figures/report")

# Static current check.
NMAX_STATIC = 101
X_STATIC = np.linspace(-np.sqrt(2) * L, np.sqrt(2) * L, 2001)
DELTA_KY = 1.0e-4 * AU_TO_NM
VE_SCALE = 0.2
KY_STATIC_NM = np.array([-0.05, 0.12])
LEVEL_STATIC = 0

# Driven current and continuity check.
NMAX_DRIVEN = 48
NSTEP_DRIVEN = 64
KY_DRIVEN_NM = -0.05
TOTAL_TIME_FRACTION = 0.2
X_DRIVEN = np.linspace(-np.sqrt(2) * L, np.sqrt(2) * L, 1201)


def latex_sci(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10**exponent
    return rf"{mantissa:.2f}\times 10^{{{exponent}}}"


def static_current_data() -> tuple[np.ndarray, np.ndarray]:
    ve = VE_SCALE * hbar * w_c / e
    slope_velocity = []
    current_velocity = []
    for ky_nm in KY_STATIC_NM:
        ky = ky_nm * AU_TO_NM
        _, eigenvectors = eigh(build_H(ky, ve, NMAX_STATIC))
        energy_minus = eigh(
            build_H(ky - DELTA_KY, ve, NMAX_STATIC), eigvals_only=True
        )[0]
        energy_plus = eigh(
            build_H(ky + DELTA_KY, ve, NMAX_STATIC), eigvals_only=True
        )[0]
        slope_velocity.append((energy_plus - energy_minus) / (2 * DELTA_KY * hbar))

        psi, dpsi_dx = wavefunction_and_derivative(eigenvectors[:, LEVEL_STATIC], X_STATIC)
        jx, jy = line_current_density(psi, dpsi_dx, X_STATIC, ky)
        _, total_jy = integrated_line_current(jx, jy, X_STATIC)
        current_velocity.append(-total_jy / e)
    return np.array(slope_velocity), np.array(current_velocity)


def plot_static_current_profiles() -> None:
    ve = VE_SCALE * hbar * w_c / e
    x_nm = X_STATIC * AU_TO_NM
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True)
    colors = [REPORT_COLORS["blue"], REPORT_COLORS["green"]]

    for ax, ky_nm, color in zip(axes, KY_STATIC_NM, colors):
        ky = ky_nm * AU_TO_NM
        _, eigenvectors = eigh(build_H(ky, ve, NMAX_STATIC))
        psi, dpsi_dx = wavefunction_and_derivative(
            eigenvectors[:, LEVEL_STATIC], X_STATIC
        )
        _, jy = line_current_density(psi, dpsi_dx, X_STATIC, ky)
        total_jy = np.trapezoid(jy, X_STATIC)
        ax.plot(x_nm, jy, color=color)
        ax.axhline(0, color=REPORT_COLORS["gray"], lw=0.9)
        ax.axvline(0, color=REPORT_COLORS["gray"], lw=0.9, ls="--")
        ax.set_title(
            rf"$k_y={ky_nm:+.2f}\,\mathrm{{nm}}^{{-1}}$, "
            rf"$n={LEVEL_STATIC}$"
        )
        ax.set_xlabel(r"$x\;(\mathrm{nm})$")
        ax.text(
            0.02,
            0.93,
            rf"$\int\mathcal{{J}}_y dx={latex_sci(total_jy)}$",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )

    axes[0].set_ylabel(r"$\mathcal{J}_y(x)=L_y j_y(x)$ (a.u.)")
    fig.suptitle(r"Static line-current profiles for both signs of $k_y$")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "static_current_profiles_ky_pm.png")
    plt.close(fig)


def driven_current_data():
    ve = VE_SCALE * hbar * w_c / e
    ky = KY_DRIVEN_NM * AU_TO_NM
    drive_period = 2 * np.pi / w_c
    coefficients, times = time_evolute(
        ky,
        ve,
        level=0,
        nmax_value=NMAX_DRIVEN,
        nstep_value=NSTEP_DRIVEN,
        total_time=TOTAL_TIME_FRACTION * drive_period,
        eig_k=32,
    )
    psi_t, dpsi_dx_t = wavefunction_and_derivative(coefficients, X_DRIVEN)
    jx_t, _ = line_current_density(psi_t, dpsi_dx_t, X_DRIVEN, ky)
    total_jx, _ = integrated_line_current(jx_t, jx_t * 0, X_DRIVEN)
    x_mean = np.trapezoid(np.abs(psi_t) ** 2 * X_DRIVEN, X_DRIVEN, axis=1)
    dx_dt = np.gradient(x_mean, times, edge_order=2)
    current_velocity_x = -total_jx / e
    residual = charge_continuity_residual(psi_t, jx_t, times, X_DRIVEN)
    scale = max(
        np.max(np.abs(np.gradient(-e * np.abs(psi_t) ** 2, times, axis=0))),
        1e-30,
    )
    return drive_period, times, dx_dt, current_velocity_x, residual / scale


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    use_report_style(use_tex=True)

    slope_velocity, current_velocity = static_current_data()
    drive_period, times, dx_dt, current_velocity_x, residual_scaled = driven_current_data()
    time_cycles = times / drive_period

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7))
    axes[0].plot(
        KY_STATIC_NM,
        slope_velocity,
        "o-",
        color=REPORT_COLORS["blue"],
        label=r"$\hbar^{-1}dE/dk_y$",
    )
    axes[0].plot(
        KY_STATIC_NM,
        current_velocity,
        "s--",
        color=REPORT_COLORS["orange"],
        label=r"$-\int\mathcal{J}_y\,dx/e$",
    )
    axes[0].set(
        title=r"Static band slope",
        xlabel=r"$k_y\;(\mathrm{nm}^{-1})$",
        ylabel=r"velocity (a.u.)",
    )
    axes[0].legend(fontsize=9)

    axes[1].plot(time_cycles, dx_dt, color=REPORT_COLORS["blue"], label=r"$d\langle x\rangle/dt$")
    axes[1].plot(
        time_cycles,
        current_velocity_x,
        "--",
        color=REPORT_COLORS["orange"],
        label=r"$-\int\mathcal{J}_x\,dx/e$",
    )
    axes[1].set(
        title=r"Driven Ehrenfest check",
        xlabel=r"time / drive period",
        ylabel=r"velocity (a.u.)",
    )
    axes[1].legend(fontsize=9)

    image = axes[2].imshow(
        np.abs(residual_scaled).T,
        aspect="auto",
        origin="lower",
        extent=(time_cycles[0], time_cycles[-1], X_DRIVEN[0], X_DRIVEN[-1]),
        cmap="magma",
    )
    axes[2].set(
        title=r"Continuity residual",
        xlabel=r"time / drive period",
        ylabel=r"$x$ (a.u.)",
    )
    fig.colorbar(image, ax=axes[2], label=r"relative residual")
    fig.suptitle(r"Gauge-covariant current validation", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "current_validation.png")
    plt.close(fig)
    plot_static_current_profiles()

    print("Generated", OUT_DIR / "current_validation.png")
    print("Generated", OUT_DIR / "static_current_profiles_ky_pm.png")
    print("Parameters:")
    print("  NMAX_STATIC =", NMAX_STATIC)
    print("  X_STATIC points =", X_STATIC.size)
    print("  VE_SCALE =", VE_SCALE)
    print("  KY_STATIC_NM =", KY_STATIC_NM.tolist())
    print("  LEVEL_STATIC =", LEVEL_STATIC)
    print("  NMAX_DRIVEN =", NMAX_DRIVEN)
    print("  NSTEP_DRIVEN =", NSTEP_DRIVEN)
    print("  KY_DRIVEN_NM =", KY_DRIVEN_NM)
    print("  TOTAL_TIME_FRACTION =", TOTAL_TIME_FRACTION)


if __name__ == "__main__":
    main()
