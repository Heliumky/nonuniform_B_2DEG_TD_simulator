## time_visualize.py — replace entirely with this:
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from params import hbar, B0, L, m_star, w_c, AU_TO_MEV, AU_TO_NM
from time_coeffs import compute_prob_evolute, w_ac, T, nstep, Ve, ky, level
from potential import V as potential_V

stride = 3
output_path = Path("figures") / "prob_evo2.gif"
output_path.parent.mkdir(exist_ok=True)

## ── precompute probability density evolution ──
prob, t_arr, x = compute_prob_evolute(ky, Ve, level)
x_nm = x * AU_TO_NM

## ── precompute potential at every stride frame ──
## V(x, Ve*cos(w_ac*t)) for each timestep
n_frames = len(range(0, len(t_arr), stride))
pot_frames = []
for i in range(0, len(t_arr), stride):
    t = t_arr[i]
    Ve_t = Ve * np.cos(w_ac * t)                    ## tunable: sin or cos
    pot = potential_V(x, ky, Ve_t) * AU_TO_MEV      ## in meV
    pot_frames.append(pot)
pot_frames = np.array(pot_frames)
prob_frames = prob[::stride]

## ── set up two-panel figure ──
fig, (ax_pot, ax_psi) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
fig.subplots_adjust(hspace=0.1)

## potential panel
line_pot, = ax_pot.plot(x_nm, pot_frames[0], color='darkorange', lw=1.5)
ax_pot.set_ylabel('V(x, t)  (meV)')
ax_pot.set_ylim(pot_frames.min() - 0.5, pot_frames.max() + 0.5)
ax_pot.axvline(0, color='gray', lw=0.5, ls='--')
ax_pot.axhline(0, color='gray', lw=0.5, ls=':')
time_text = ax_pot.text(0.02, 0.92, '', transform=ax_pot.transAxes, fontsize=11)

## probability density panel
line_psi, = ax_psi.plot(x_nm, prob_frames[0], color='steelblue', lw=1.5)
ax_psi.set_ylabel('|ψ(x, t)|²')
ax_psi.set_xlabel('x  (nm)')
ax_psi.set_ylim(0, prob_frames.max() * 1.15)
ax_psi.axvline(0, color='gray', lw=0.5, ls='--')

fig.suptitle(f'AC driving: $\\omega_{{ac}}$ = {w_ac/w_c}$\\omega_c$,  '
             f'$eV_e$ = {Ve/(hbar*B0/m_star)}$\\hbar\\omega_c$,  '
             f'$k_y$ = {ky/AU_TO_NM:.2f} nm⁻¹,  '
             f'n = {level}', fontsize=11,)

## ── animation ──
frames = range(len(prob_frames))

def update(fi):
    line_pot.set_ydata(pot_frames[fi])
    line_psi.set_ydata(prob_frames[fi])
    t_real = t_arr[fi * stride]
    time_text.set_text(f't = {t_real/T:.2f} T')
    return line_pot, line_psi, time_text

ani = animation.FuncAnimation(fig, update, frames=frames,
                               interval=50, blit=True)
ani.save(output_path, writer='pillow', fps=20)
print(f"Saved {output_path}")
