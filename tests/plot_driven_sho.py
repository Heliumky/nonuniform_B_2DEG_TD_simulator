#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_driven_sho import (  # noqa: E402
    OMEGA_DRIVE,
    exact_classical_position,
    evolve_like_time_coeffs,
)


def main():
    total_time = 4 * np.pi / OMEGA_DRIVE
    x0, _, t_array, state_times, states, x_mat = evolve_like_time_coeffs(
        total_time=total_time,
        n_steps=240,
    )

    x_numeric = np.real(np.einsum("bi,ij,bj->b", states.conj(), x_mat, states))
    x_exact = exact_classical_position(state_times, x0)
    error = x_numeric - x_exact
    time_axis = OMEGA_DRIVE * state_times / (2 * np.pi)
    recorded_time_axis = OMEGA_DRIVE * t_array / (2 * np.pi)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_top.plot(time_axis, x_exact / x0, color="black", lw=2.0, label="exact")
    ax_top.plot(
        time_axis,
        x_numeric / x0,
        color="tab:blue",
        ls="--",
        lw=1.8,
        label="numeric: midpoint exponential",
    )
    ax_top.scatter(
        recorded_time_axis[::12],
        x_numeric[::12] / x0,
        s=12,
        color="tab:orange",
        label="numerical samples",
        zorder=3,
    )
    ax_top.set_ylabel(r"$\langle x\rangle / x_0$")
    ax_top.set_title(
        r"Standalone 1D driven SHO benchmark from eigenstate of $H(0)$: "
        r"$H(t)=H_0 + A x \cos(\omega t)$"
    )
    ax_top.legend()
    ax_top.grid(alpha=0.25)

    ax_bottom.plot(time_axis, error / x0, color="tab:red", lw=1.4)
    ax_bottom.axhline(0.0, color="black", lw=0.8)
    ax_bottom.set_xlabel(r"drive periods, $\omega t / 2\pi$")
    ax_bottom.set_ylabel(r"error$/x_0$")
    ax_bottom.grid(alpha=0.25)

    max_error = np.max(np.abs(error / x0))
    fig.text(0.99, 0.01, f"max abs error = {max_error:.2e} x0", ha="right")
    fig.tight_layout()

    output_path = REPO_ROOT / "figures" / "driven_sho_validation.png"
    fig.savefig(output_path, dpi=180)
    print(output_path)


if __name__ == "__main__":
    main()
