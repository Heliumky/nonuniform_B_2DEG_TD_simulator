"""Generate static-model figures for non_BTD.md.

Outputs:
    figures/report/static_potential.png
    figures/report/static_spectrum_ve0.png
    figures/report/static_density_ky_pm015.png

Numerical parameters are intentionally kept here, not hidden inside the slide
deck, so the report figures are reproducible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from hamiltonian import build_H
from params import AU_TO_MEV, AU_TO_NM, B0, L, e, hbar, m_star
from report_plotsetting import REPORT_COLORS, use_report_style
from wave_function import QHO_basis


OUT_DIR = Path("figures/report")

# Static model parameters.
VE = 0.0
KY_POTENTIAL_NM = np.array([0.15, -0.05, -0.15])
KY_DENSITY_NM = np.array([0.15, -0.15])
LEVEL_DENSITY = 0

# Numerical parameters.
NMAX_SPECTRUM = 221
NMAX_DENSITY = 221
NLEVEL = 8
KY_GRID_NM = np.linspace(-0.35, 0.35, 121)
X_GRID_AU = np.linspace(-np.sqrt(2) * L, np.sqrt(2) * L, 501)


def potential(x_au: np.ndarray, ky_au: float, ve: float = VE) -> np.ndarray:
    magnetic = (hbar * ky_au + e * B0 * x_au**2 / (2 * L)) ** 2 / (2 * m_star)
    electric = e * ve * x_au / L
    return magnetic + electric


def plot_static_potential() -> None:
    x_nm = X_GRID_AU * AU_TO_NM
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = [REPORT_COLORS["blue"], REPORT_COLORS["orange"], REPORT_COLORS["green"]]
    for ky_nm, color in zip(KY_POTENTIAL_NM, colors):
        ky_au = ky_nm * AU_TO_NM
        ax.plot(
            x_nm,
            AU_TO_MEV * potential(X_GRID_AU, ky_au),
            color=color,
            label=rf"$k_y={ky_nm:+.2f}\,\mathrm{{nm}}^{{-1}}$",
        )
    ax.set_xlabel(r"$x\;(\mathrm{nm})$")
    ax.set_ylabel(r"$V_{k_y}(x)\;(\mathrm{meV})$")
    ax.set_title(r"Static effective potential, $V_e=0$")
    ax.set_ylim(0, 30)
    ax.axvline(0, color=REPORT_COLORS["gray"], lw=1.0, ls="--")
    ax.legend(loc="upper center")
    fig.savefig(OUT_DIR / "static_potential.png")
    plt.close(fig)


def compute_spectrum() -> np.ndarray:
    evals = np.empty((KY_GRID_NM.size, NLEVEL))
    for i, ky_nm in enumerate(KY_GRID_NM):
        hamiltonian = build_H(ky_nm * AU_TO_NM, VE, NMAX_SPECTRUM)
        values = eigh(hamiltonian, eigvals_only=True)
        evals[i] = values[:NLEVEL] * AU_TO_MEV
    return evals


def plot_static_spectrum() -> None:
    evals = compute_spectrum()
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for band in range(NLEVEL):
        ax.plot(KY_GRID_NM, evals[:, band], color=REPORT_COLORS["blue"], lw=1.3)
    ax.set_xlabel(r"$k_y\;(\mathrm{nm}^{-1})$")
    ax.set_ylabel(r"$E_n(k_y)\;(\mathrm{meV})$")
    ax.set_title(r"Static band structure, $V_e=0$")
    ax.set_xlim(KY_GRID_NM[0], KY_GRID_NM[-1])
    ax.set_ylim(-2, 16)
    ax.axvline(0, color=REPORT_COLORS["gray"], lw=1.0, ls="--")
    fig.savefig(OUT_DIR / "static_spectrum_ve0.png")
    plt.close(fig)


def density_for_ky(ky_nm: float) -> np.ndarray:
    hamiltonian = build_H(ky_nm * AU_TO_NM, VE, NMAX_DENSITY)
    _, vectors = eigh(hamiltonian)
    coeffs = vectors[:, LEVEL_DENSITY]
    basis = QHO_basis(X_GRID_AU, NMAX_DENSITY)
    psi = coeffs @ basis
    return np.abs(psi) ** 2


def plot_static_density() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True)
    colors = [REPORT_COLORS["blue"], REPORT_COLORS["green"]]
    x_nm = X_GRID_AU * AU_TO_NM
    for ax, ky_nm, color in zip(axes, KY_DENSITY_NM, colors):
        density = density_for_ky(ky_nm)
        norm = np.trapezoid(density, X_GRID_AU)
        ax.plot(x_nm, density, color=color)
        ax.set_title(
            rf"$k_y={ky_nm:+.2f}\,\mathrm{{nm}}^{{-1}}$, "
            rf"$n={LEVEL_DENSITY}$, norm={norm:.4f}"
        )
        ax.set_xlabel(r"$x\;(\mathrm{nm})$")
        ax.axvline(0, color=REPORT_COLORS["gray"], lw=1.0, ls="--")
    axes[0].set_ylabel(r"$|\psi_{n,k_y}(x)|^2$")
    fig.suptitle(r"Static eigenstate density at $V_e=0$")
    fig.savefig(OUT_DIR / "static_density_ky_pm015.png")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    use_report_style(use_tex=True)
    plot_static_potential()
    plot_static_spectrum()
    plot_static_density()
    print("Generated report figures in", OUT_DIR)
    print("Parameters:")
    print("  VE =", VE)
    print("  NMAX_SPECTRUM =", NMAX_SPECTRUM)
    print("  NMAX_DENSITY =", NMAX_DENSITY)
    print("  NLEVEL =", NLEVEL)
    print("  KY_GRID_NM =", (float(KY_GRID_NM[0]), float(KY_GRID_NM[-1]), KY_GRID_NM.size))
    print("  KY_DENSITY_NM =", KY_DENSITY_NM.tolist())
    print("  LEVEL_DENSITY =", LEVEL_DENSITY)


if __name__ == "__main__":
    main()
