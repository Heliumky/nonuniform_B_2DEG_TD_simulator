"""Generate time-propagation validation figures for non_BTD.md.

This script reuses the standalone driven-SHO validation model from the test
suite and applies the report plotting style.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from report_plotsetting import REPORT_COLORS, use_report_style
from test_driven_sho import OMEGA_DRIVE, exact_classical_position, evolve_like_time_coeffs


OUT_DIR = Path("figures/report")
N_STEPS = 240
TOTAL_PERIODS = 2.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    use_report_style(use_tex=True)

    total_time = TOTAL_PERIODS * 2 * np.pi / OMEGA_DRIVE
    x0, _, t_array, state_times, states, x_mat = evolve_like_time_coeffs(
        total_time=total_time,
        n_steps=N_STEPS,
    )

    x_numeric = np.real(np.einsum("bi,ij,bj->b", states.conj(), x_mat, states))
    x_exact = exact_classical_position(state_times, x0)
    error = x_numeric - x_exact
    time_axis = OMEGA_DRIVE * state_times / (2 * np.pi)
    recorded_time_axis = OMEGA_DRIVE * t_array / (2 * np.pi)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(6.8, 5.1),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_top.plot(time_axis, x_exact / x0, color="black", lw=2.0, label=r"exact")
    ax_top.plot(
        time_axis,
        x_numeric / x0,
        color=REPORT_COLORS["blue"],
        ls="--",
        lw=1.8,
        label=r"numerical",
    )
    ax_top.scatter(
        recorded_time_axis[::12],
        x_numeric[::12] / x0,
        s=16,
        color=REPORT_COLORS["orange"],
        label=r"samples",
        zorder=3,
    )
    ax_top.set_ylabel(r"$\langle x\rangle/x_0$")
    ax_top.set_title(r"Driven SHO validation for the propagation pipeline")
    ax_top.legend(loc="best")

    ax_bottom.plot(time_axis, error / x0, color=REPORT_COLORS["red"], lw=1.4)
    ax_bottom.axhline(0.0, color="black", lw=0.8)
    ax_bottom.set_xlabel(r"drive periods, $\omega t/(2\pi)$")
    ax_bottom.set_ylabel(r"error$/x_0$")

    max_error = np.max(np.abs(error / x0))
    fig.text(0.99, 0.01, rf"max abs error = {max_error:.2e}$\,x_0$", ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "driven_sho_validation.png")
    plt.close(fig)

    print("Generated", OUT_DIR / "driven_sho_validation.png")
    print("Parameters:")
    print("  N_STEPS =", N_STEPS)
    print("  TOTAL_PERIODS =", TOTAL_PERIODS)
    print("  OMEGA_DRIVE =", OMEGA_DRIVE)
    print("  max_abs_error_over_x0 =", float(max_error))


if __name__ == "__main__":
    main()
