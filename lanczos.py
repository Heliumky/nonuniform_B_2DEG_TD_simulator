"""Sparse Lanczos helpers for eigenpairs and exponential actions.

This module uses SciPy's sparse linear-algebra routines instead of maintaining
a custom Lanczos recurrence here:

- `eigsh` is ARPACK's implicitly restarted Lanczos solver for Hermitian
  eigenproblems.
- `expm_multiply` applies the matrix exponential to a vector without building
  the full exponential matrix.
"""

import numpy as np
from scipy.sparse.linalg import eigsh, expm_multiply


def _normalized_start_vector(n, psi0=None, complex_start=False):
    if psi0 is not None:
        psi0 = np.asarray(psi0)
        norm = np.linalg.norm(psi0)
        if norm == 0:
            raise ValueError("psi0 should not be the zero vector")
        return psi0 / norm

    rng = np.random.default_rng(1234)
    if complex_start:
        psi0 = rng.normal(size=n) + 1j * rng.normal(size=n)
    else:
        psi0 = rng.normal(size=n)
    return psi0 / np.linalg.norm(psi0)


def lanczos_eigenpairs(H, nlevel=1, k=60, tol=1e-10, psi0=None):
    """Return the lowest `nlevel` eigenpairs using restarted Lanczos.

    Args:
        H: Hermitian matrix or sparse/linear operator.
        nlevel: Number of lowest eigenpairs to compute.
        k: Krylov subspace dimension passed to ARPACK as `ncv`.
        tol: ARPACK convergence tolerance.
        psi0: Optional starting vector.
    """
    n = H.shape[0]
    if H.shape[0] != H.shape[1]:
        raise ValueError("H should be a square matrix")
    if nlevel < 1:
        raise ValueError("nlevel should be at least 1")
    if nlevel >= n:
        raise ValueError("nlevel should be smaller than matrix dimension")

    if nlevel >= n - 1:
        evals, evecs = np.linalg.eigh(H)
        return evals[:nlevel], evecs[:, :nlevel]

    start = _normalized_start_vector(
        n,
        psi0=psi0,
        complex_start=np.iscomplexobj(H),
    )
    ncv = min(n, max(k, 2 * nlevel + 1))
    evals, evecs = eigsh(
        H,
        k=nlevel,
        which="SA",
        ncv=ncv,
        tol=tol,
        v0=start,
    )
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def lanczos_ground_state(H, psi0=None, k=60, dtype=complex, tol=1e-10):
    """Return the lowest eigenvalue/eigenvector using restarted Lanczos."""
    evals, evecs = lanczos_eigenpairs(
        H,
        nlevel=1,
        k=k,
        tol=tol,
        psi0=psi0,
    )
    return evals[0], evecs[:, 0].astype(dtype, copy=False)


def lanczos_expm_multiply(H, psi0, dt, k=None, dtype=complex):
    """Return `expm(dt * H) @ psi0` using SciPy's sparse expm action.

    The `k` argument is accepted for compatibility with older call sites, but
    `expm_multiply` chooses its internal Krylov parameters adaptively.
    """
    psi0 = np.asarray(psi0, dtype=dtype)
    if psi0.ndim != 1:
        raise ValueError("psi0 should be a vector")
    if H.shape[1] != psi0.shape[0]:
        raise ValueError("Shape of H doesn't match len of psi0.")
    return expm_multiply(dt * H, psi0)
