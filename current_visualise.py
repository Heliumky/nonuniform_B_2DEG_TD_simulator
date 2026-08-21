"""Animate the charge-current density for one fixed-k_y channel.

The plotted quantities are L_y j_x and L_y j_y: charge current per unit
length in the translationally invariant y direction.  Divide by a chosen L_y
to obtain the current densities in the convention used for J^mu.
"""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from current import line_current_density, sho_basis_and_derivative
from params import L, hbar, w_c, AU_TO_NM
from time_coeffs import T, Ve, ky, level, nmax, time_evolute, w_ac


stride = 3
output_path = Path("figures") / "current_evo.gif"
output_path.parent.mkdir(exist_ok=True)


# Evolve the same initial state and drive used by time_visualise.py.
x = np.linspace(-2**0.5 * L, 2**0.5 * L, 401)
c_snapshots, t_arr = time_evolute(ky, Ve, level)
basis, dbasis_dx = sho_basis_and_derivative(x, nmax)

psi = c_snapshots @ basis
dpsi_dx = c_snapshots @ dbasis_dx
density = np.abs(psi) ** 2

# These are L_y times the j_x and j_y in the four-current convention.
jx_line, jy_line = line_current_density(psi, dpsi_dx, x, ky)


def symmetric_limit(values):
    limit = np.max(np.abs(values))
    return (-1.1 * limit, 1.1 * limit) if limit else (-1.0, 1.0)


x_nm = x * AU_TO_NM
jx_frames = jx_line[::stride]
jy_frames = jy_line[::stride]

fig, (ax_jx, ax_jy) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
fig.subplots_adjust(hspace=0.10)

line_jx, = ax_jx.plot(x_nm, jx_frames[0], color="crimson", lw=1.5)
line_jy, = ax_jy.plot(x_nm, jy_frames[0], color="seagreen", lw=1.5)

for ax in (ax_jx, ax_jy):
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.axhline(0, color="gray", lw=0.5, ls=":")

ax_jx.set_ylabel(r"$L_y j_x$ (a.u.)")
ax_jy.set_ylabel(r"$L_y j_y$ (a.u.)")
ax_jy.set_xlabel("x (nm)")
ax_jx.set_ylim(*symmetric_limit(jx_line))
ax_jy.set_ylim(*symmetric_limit(jy_line))
time_text = ax_jx.text(0.02, 0.92, "", transform=ax_jx.transAxes, fontsize=11)

fig.suptitle(
    rf"AC current: $\omega_{{ac}}={w_ac / w_c:g}\omega_c$,  "
    rf"$eV_e={Ve / (hbar * w_c):g}\hbar\omega_c$,  "
    rf"$k_y={ky / AU_TO_NM:.2f}\,\mathrm{{nm}}^{{-1}}$,  $n={level}$",
    fontsize=11,
)


def update(frame):
    line_jx.set_ydata(jx_frames[frame])
    line_jy.set_ydata(jy_frames[frame])
    time_text.set_text(f"t = {t_arr[frame * stride] / T:.2f} T")
    return line_jx, line_jy, time_text


movie = animation.FuncAnimation(
    fig, update, frames=len(jx_frames), interval=50, blit=True
)
movie.save(output_path, writer="pillow", fps=20)
print(f"Saved {output_path}")
