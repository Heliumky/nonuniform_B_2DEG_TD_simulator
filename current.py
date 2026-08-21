"""Current-density observables for a fixed-:math:`k_y` channel.

The one-dimensional wavefunction ``psi(x)`` is normalized in ``x``.  Hence
the functions below return *line currents* ``J = L_y j``.  The physical 2D
current density of a single box-normalized channel is obtained by dividing by
``L_y``.  See ``non_BTD.md`` for the derivation from the gauge-covariant
Schroedinger Lagrangian.
"""

from __future__ import annotations

import numpy as np

from params import B0, L, e, hbar, m_star, w_c
from wave_function import QHO_basis


def sho_basis_and_derivative(x: np.ndarray, nmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized SHO basis functions and their spatial derivatives.

    The derivative follows exactly from ladder operators within the truncated
    basis, rather than from a finite difference of sampled basis functions.
    """
    basis_with_upper = QHO_basis(x, nmax + 1)
    oscillator_length = np.sqrt(hbar / (m_star * w_c))
    derivative = np.empty((nmax, len(x)))

    for n in range(nmax):
        lower = np.sqrt(n) * basis_with_upper[n - 1] if n else 0.0
        upper = np.sqrt(n + 1) * basis_with_upper[n + 1]
        derivative[n] = (lower - upper) / (np.sqrt(2) * oscillator_length)

    return basis_with_upper[:nmax], derivative


def wavefunction_and_derivative(
    coefficients: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct ``psi`` and ``dpsi/dx`` from one or more SHO vectors.

    ``coefficients`` may have shape ``(nmax,)`` or ``(ntime, nmax)``.  The
    returned arrays have the corresponding final spatial axis.
    """
    coefficients = np.asarray(coefficients)
    if coefficients.ndim not in (1, 2):
        raise ValueError("coefficients must have shape (nmax,) or (ntime, nmax)")
    basis, derivative = sho_basis_and_derivative(x, coefficients.shape[-1])
    return coefficients @ basis, coefficients @ derivative


def line_current_density(
    psi: np.ndarray, dpsi_dx: np.ndarray, x: np.ndarray, ky: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(L_y*j_x, L_y*j_y)`` for an electron in the selected gauge.

    Parameters broadcast over all leading dimensions of ``psi``.  ``ky`` and
    ``x`` must be in the atomic units used by the simulator.
    """
    psi = np.asarray(psi)
    dpsi_dx = np.asarray(dpsi_dx)
    x = np.asarray(x)
    if psi.shape != dpsi_dx.shape or psi.shape[-1] != x.size:
        raise ValueError("psi, dpsi_dx, and x have incompatible shapes")

    density = np.abs(psi) ** 2
    jx_line = -e * hbar / m_star * np.imag(np.conj(psi) * dpsi_dx)
    mechanical_py = hbar * ky + e * B0 * x**2 / (2 * L)
    jy_line = -e / m_star * mechanical_py * density
    return jx_line, jy_line


def integrated_line_current(
    jx_line: np.ndarray, jy_line: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate line-current profiles over x for each leading index."""
    return (
        np.trapezoid(jx_line, x, axis=-1),
        np.trapezoid(jy_line, x, axis=-1),
    )


def charge_continuity_residual(
    psi_t: np.ndarray, jx_line_t: np.ndarray, t: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """Return the finite-difference residual ``d_t(-e|psi|^2)+d_x J_x``.

    This diagnostic is deliberately evaluated on the same sampled x/t grids
    as visualizations, so it checks both the propagated state and current
    reconstruction.  Use interior points for quantitative convergence tests.
    """
    psi_t = np.asarray(psi_t)
    jx_line_t = np.asarray(jx_line_t)
    t = np.asarray(t)
    x = np.asarray(x)
    if psi_t.ndim != 2 or psi_t.shape != jx_line_t.shape:
        raise ValueError("psi_t and jx_line_t must both have shape (ntime, nx)")
    if psi_t.shape != (t.size, x.size):
        raise ValueError("t and x must match the sampled state shape")
    if t.size < 3 or x.size < 3:
        raise ValueError("at least three time and space samples are required")

    charge_line_density = -e * np.abs(psi_t) ** 2
    d_charge_dt = np.gradient(charge_line_density, t, axis=0, edge_order=2)
    d_jx_dx = np.gradient(jx_line_t, x, axis=1, edge_order=2)
    return d_charge_dt + d_jx_dx
