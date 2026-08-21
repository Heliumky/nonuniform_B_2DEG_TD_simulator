#!/usr/bin/env python3

import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.linalg import eigh, expm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from basis import operators_matrixize
from params import hbar, m_star, w_c
from reporting import report_check


NMAX = 28
OMEGA_DRIVE = 0.7 * w_c
X0_DRIVE = 0.3


def build_driven_sho_terms(nmax=NMAX):
    x0, p0, x_mat, x2_mat, _, p2_mat = operators_matrixize(nmax)
    h0 = p2_mat / (2 * m_star) + 0.5 * m_star * w_c**2 * x2_mat
    drive_amplitude = X0_DRIVE * m_star * w_c**2 * x0
    return x0, p0, x_mat, h0, drive_amplitude


def build_driven_sho_hamiltonian(t, nmax=NMAX):
    _, _, x_mat, h0, drive_amplitude = build_driven_sho_terms(nmax)
    return h0 + drive_amplitude * np.cos(OMEGA_DRIVE * t) * x_mat


def evolve_like_time_coeffs(total_time, n_steps, nmax=NMAX):
    x0, p0, x_mat, _, _ = build_driven_sho_terms(nmax)
    dt = total_time / n_steps
    _, evecs0 = eigh(build_driven_sho_hamiltonian(0.0, nmax))
    coeffs = evecs0[:, 0].astype(complex)

    states = []
    t_array = []
    state_times = []

    for step in range(n_steps):
        t = step * dt
        h_mid = build_driven_sho_hamiltonian(t + dt / 2, nmax)
        coeffs = expm(-1j * h_mid * dt / hbar) @ coeffs
        states.append(coeffs.copy())
        t_array.append(t)
        state_times.append(t + dt)

    return x0, p0, np.array(t_array), np.array(state_times), np.array(states), x_mat


def exact_classical_position(times, x0):
    force_amplitude = X0_DRIVE * m_star * w_c**2 * x0
    static_position = -force_amplitude / (m_star * w_c**2)
    driven_position = -force_amplitude / (m_star * (w_c**2 - OMEGA_DRIVE**2))
    natural_position = static_position - driven_position
    return (
        natural_position * np.cos(w_c * times)
        + driven_position * np.cos(OMEGA_DRIVE * times)
    )


def exact_classical_momentum(times, x0):
    force_amplitude = X0_DRIVE * m_star * w_c**2 * x0
    static_position = -force_amplitude / (m_star * w_c**2)
    driven_position = -force_amplitude / (m_star * (w_c**2 - OMEGA_DRIVE**2))
    natural_position = static_position - driven_position
    velocity = (
        -natural_position * w_c * np.sin(w_c * times)
        - driven_position * OMEGA_DRIVE * np.sin(OMEGA_DRIVE * times)
    )
    return m_star * velocity


def coherent_state_coeffs(beta, nmax=NMAX):
    coeffs = np.zeros(nmax, dtype=complex)
    gaussian = np.exp(-0.5 * np.abs(beta) ** 2)
    for n in range(nmax):
        coeffs[n] = gaussian * beta**n / math.sqrt(math.factorial(n))
    return coeffs


class DrivenSHOTest(unittest.TestCase):
    def test_position_matches_exact_solution(self):
        total_time = 4 * np.pi / OMEGA_DRIVE
        x0, _, _, state_times, states, x_mat = evolve_like_time_coeffs(
            total_time=total_time,
            n_steps=240,
        )

        x_expectation = np.real(np.einsum("bi,ij,bj->b", states.conj(), x_mat, states))
        x_exact = exact_classical_position(state_times, x0)
        max_error = np.max(np.abs(x_expectation - x_exact))

        relative_error = max_error / x0
        limit = 2e-3
        report_check(
            "driven harmonic oscillator position",
            "1D harmonic oscillator with force proportional to cos(omega t); reference is its analytic classical/coherent-state trajectory",
            relative_error,
            limit,
            "relative error",
        )
        self.assertLess(relative_error, limit)

    def test_final_state_matches_exact_coherent_state(self):
        total_time = 2 * np.pi / OMEGA_DRIVE
        x0, p0, _, state_times, states, _ = evolve_like_time_coeffs(
            total_time=total_time,
            n_steps=180,
        )

        final_state = states[-1]
        final_time = state_times[-1]
        x_exact = exact_classical_position(np.array([final_time]), x0)[0]
        p_exact = exact_classical_momentum(np.array([final_time]), x0)[0]
        beta_exact = x_exact / (2 * x0) + 1j * p_exact / (2 * p0)
        exact_state = coherent_state_coeffs(beta_exact)
        fidelity = np.abs(np.vdot(exact_state, final_state)) ** 2

        norm_error = abs(np.linalg.norm(final_state) - 1.0)
        infidelity = 1.0 - fidelity
        report_check(
            "driven harmonic oscillator norm preservation",
            "same 1D analytic benchmark, midpoint exponential propagator",
            norm_error,
            5e-11,
        )
        report_check(
            "driven harmonic oscillator final-state fidelity",
            "same benchmark; exact state is the coherent state with analytic x(t), p(t)",
            infidelity,
            1e-3,
            "1 - fidelity",
        )
        self.assertAlmostEqual(np.linalg.norm(final_state), 1.0, places=10)
        self.assertGreater(fidelity, 0.999)


if __name__ == "__main__":
    unittest.main()
