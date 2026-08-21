#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.linalg import expm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lanczos import (
    lanczos_eigenpairs,
    lanczos_expm_multiply,
    lanczos_ground_state,
)
from spectrum import compute_spectrum
from reporting import report_check


class LanczosTest(unittest.TestCase):
    def test_expm_multiply_matches_dense_expm_in_full_krylov_space(self):
        rng = np.random.default_rng(10)
        raw = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        H = raw + raw.conj().T
        psi = rng.normal(size=8) + 1j * rng.normal(size=8)
        psi /= np.linalg.norm(psi)
        scale = -0.07j

        actual = lanczos_expm_multiply(H, psi, scale, k=8, dtype=complex)
        expected = expm(scale * H) @ psi

        error = np.linalg.norm(actual - expected)
        limit = 1e-10
        report_check(
            "Lanczos exponential action",
            "8-dimensional random Hermitian Hamiltonian; exact reference is scipy.linalg.expm",
            error,
            limit,
        )
        self.assertLess(error, limit)

    def test_ground_state_matches_dense_in_full_krylov_space(self):
        rng = np.random.default_rng(20)
        raw = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
        H = raw + raw.conj().T
        psi0 = rng.normal(size=9) + 1j * rng.normal(size=9)

        energy, state = lanczos_ground_state(H, psi0=psi0, k=9)
        expected_energies, _ = np.linalg.eigh(H)
        residual = np.linalg.norm(H @ state - energy * state)

        energy_error = abs(energy - expected_energies[0])
        limit = 1e-10
        report_check(
            "Lanczos ground-state energy",
            "9-dimensional random Hermitian Hamiltonian; exact reference is numpy.linalg.eigh",
            energy_error,
            limit,
        )
        report_check(
            "Lanczos ground-state residual",
            "same model; checks H|psi> = E|psi>",
            residual,
            limit,
        )
        self.assertAlmostEqual(energy, expected_energies[0], places=10)
        self.assertLess(residual, limit)

    def test_lanczos_eigenpairs_matches_dense_for_multiple_values(self):
        rng = np.random.default_rng(40)
        raw = rng.normal(size=(20, 20))
        H = raw + raw.T

        actual, _ = lanczos_eigenpairs(H, nlevel=5, k=14)
        expected, _ = np.linalg.eigh(H)

        error = np.max(np.abs(actual - expected[:5]))
        limit = 1e-10
        report_check(
            "Lanczos lowest five eigenvalues",
            "20-dimensional real Hermitian Hamiltonian; exact reference is numpy.linalg.eigh",
            error,
            limit,
        )
        self.assertLess(error, limit)

    def test_spectrum_lanczos_matches_dense_for_multiple_bands(self):
        ky_array = np.array([-0.2, 0.0, 0.2])
        dense = compute_spectrum(
            ky_array,
            Ve=0.0,
            nmax=64,
            nlevel=8,
            method="dense",
        )
        lanczos = compute_spectrum(
            ky_array,
            Ve=0.0,
            nmax=64,
            nlevel=8,
            method="lanczos",
            lanczos_k=40,
        )

        error = np.max(np.abs(lanczos - dense))
        limit = 1e-8
        report_check(
            "2DEG spectrum: Lanczos versus dense diagonalization",
            "production quartic 2DEG Hamiltonian at three ky values and eight bands",
            error,
            limit,
            "Hartree",
        )
        self.assertLess(error, limit)


if __name__ == "__main__":
    unittest.main()
