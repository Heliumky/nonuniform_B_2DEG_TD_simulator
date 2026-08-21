import numpy as np
from scipy.linalg import expm, eigh
from params import hbar, B0, e, L, m_star, w_c, AU_TO_MEV, AU_TO_NM
from hamiltonian import build_H
from basis import operators_matrixize
from lanczos import lanczos_eigenpairs, lanczos_expm_multiply
from wave_function import QHO_basis

"""
Ve = ne * (hbar*B0/m_star), for ne=0.5 -> 1st flat band
Ve -> Ve*cos(w_ac*t)
w_ac: ac frequency
Default time evolution: 6th-order Upsilon_2 commutator-free propagator
from method.pdf, with Lanczos Ritz vectors and Lanczos exponential actions.
"""

w_ac = 1 * w_c
T = 20 * np.pi / w_ac    ## consider 10 full period
nstep = 1000
dt = T / nstep
nmax = 301

ky = -0.15 * AU_TO_NM            ## tunable
Ve = 0.5 * (hbar*B0/m_star)      ## tunable
level = 1                        ## tunable: 0 for ground state

Y6_2_C = np.array([
    0.5 - np.sqrt(15) / 10,
    0.5,
    0.5 + np.sqrt(15) / 10,
])

Y6_2_A1 = np.array([
    (10 + np.sqrt(15)) / 180,
    -1 / 9,
    (10 - np.sqrt(15)) / 180,
])

Y6_2_A2 = np.array([
    (15 + 8 * np.sqrt(15)) / 90,
    2 / 3,
    (15 - 8 * np.sqrt(15)) / 90,
])


def build_H_ac(ky, Ve, t, nmax_value=nmax, w_ac_value=w_ac):
    Ve_ac = Ve * np.cos(w_ac_value * t)      ## tunable: sin or cos
    return build_H(ky, Ve_ac, nmax_value)


def build_split_operators(ky, Ve, nmax_value=nmax):
    _, _, x_mat, x2_mat, x4_mat, p2_mat = operators_matrixize(nmax_value)
    kinetic = p2_mat / (2 * m_star)
    static_potential = (
        (hbar**2 * ky**2 / (2 * m_star)) * np.eye(nmax_value)
        + (hbar * ky * e * B0 / (2 * m_star * L)) * x2_mat
        + (e**2 * B0**2 / (8 * m_star * L**2)) * x4_mat
    )
    drive_potential = (e * Ve / L) * x_mat
    return kinetic, static_potential, drive_potential


def coordinate_potential(static_potential, drive_potential, t, w_ac_value=w_ac):
    return static_potential + np.cos(w_ac_value * t) * drive_potential


def y6_2_modified_potential(Ve, t, tau, nmax_value=nmax, w_ac_value=w_ac):
    f1 = np.cos(w_ac_value * (t + Y6_2_C[0] * tau))
    f3 = np.cos(w_ac_value * (t + Y6_2_C[2] * tau))
    drive_gradient = e * Ve / L
    scalar = -((f3 - f1) ** 2) * drive_gradient**2 / (m_star * 25920)
    return scalar * np.eye(nmax_value)


def time_evolute_midpoint_dense(ky, Ve, level):
    """
    Original midpoint evolution using dense eigensolve and dense matrix expm.

    return:
        1. c_snapshots(coefficient of "level", shape: (nstep, nmax))
        2. t_array(shape: nstep)
    """
    H = build_H_ac(ky, Ve, 0)
    _, evecs0 = eigh(H)
    coeffs = evecs0[:, level].astype(complex)     ## initial state coefficient

    c_snapshots = []
    t_array = []

    for step in range(nstep):
        t = step * dt
        H_t = build_H_ac(ky, Ve, t + dt/2)        ## mid of each dt
        U = expm (-1j * H_t * dt / hbar)
        coeffs = U @ coeffs
        c_snapshots.append(coeffs.copy())
        t_array.append(t + dt)

    return np.array(c_snapshots), np.array(t_array)


def time_evolute_y6_2_lanczos(
    ky,
    Ve,
    level,
    lanczos_k=40,
    eig_k=120,
    nmax_value=nmax,
    nstep_value=nstep,
    total_time=T,
    w_ac_value=w_ac,
):
    """
    Sixth-order Upsilon_2 propagator from method.pdf in the SHO basis.

    The Hamiltonian is split as H(t)=T+V(t). Since V(t) is not diagonal in the
    SHO basis, every exponential action is evaluated by Lanczos.
    """
    if nstep_value < 1:
        raise ValueError("nstep_value must be positive")
    dt_value = total_time / nstep_value
    kinetic, static_potential, drive_potential = build_split_operators(
        ky, Ve, nmax_value
    )
    H0 = kinetic + coordinate_potential(
        static_potential, drive_potential, 0.0, w_ac_value
    )
    evals0, evecs0 = lanczos_eigenpairs(H0, nlevel=level + 1, k=eig_k)
    coeffs = evecs0[:, level].astype(complex)

    c_snapshots = []
    t_array = []

    for step in range(nstep_value):
        t = step * dt_value
        V1, V2, V3 = [
            coordinate_potential(
                static_potential,
                drive_potential,
                t + c * dt_value,
                w_ac_value,
            )
            for c in Y6_2_C
        ]

        Vbar1 = Y6_2_A1[0] * V1 + Y6_2_A1[1] * V2 + Y6_2_A1[2] * V3
        Vbar2 = Y6_2_A2[0] * V1 + Y6_2_A2[1] * V2 + Y6_2_A2[2] * V3
        Vbar3 = Y6_2_A2[2] * V1 + Y6_2_A2[1] * V2 + Y6_2_A2[0] * V3
        Vbar4 = Y6_2_A1[2] * V1 + Y6_2_A1[1] * V2 + Y6_2_A1[0] * V3
        V_tilde = y6_2_modified_potential(
            Ve, t, dt_value, nmax_value, w_ac_value
        )

        coeffs = lanczos_expm_multiply(
            Vbar1 + dt_value**2 * V_tilde,
            coeffs,
            -1j * dt_value / hbar,
            k=lanczos_k,
            dtype=complex,
        )
        coeffs = lanczos_expm_multiply(
            kinetic + Vbar2,
            coeffs,
            -1j * (dt_value / 2) / hbar,
            k=lanczos_k,
            dtype=complex,
        )
        coeffs = lanczos_expm_multiply(
            kinetic + Vbar3,
            coeffs,
            -1j * (dt_value / 2) / hbar,
            k=lanczos_k,
            dtype=complex,
        )
        coeffs = lanczos_expm_multiply(
            Vbar4 + dt_value**2 * V_tilde,
            coeffs,
            -1j * dt_value / hbar,
            k=lanczos_k,
            dtype=complex,
        )

        c_snapshots.append(coeffs.copy())
        t_array.append(t + dt_value)

    return np.array(c_snapshots), np.array(t_array)


def time_evolute(
    ky,
    Ve,
    level,
    method="y6_2_lanczos",
    lanczos_k=40,
    eig_k=120,
    nmax_value=nmax,
    nstep_value=nstep,
    total_time=T,
    w_ac_value=w_ac,
):
    if method == "y6_2_lanczos":
        return time_evolute_y6_2_lanczos(
            ky,
            Ve,
            level,
            lanczos_k=lanczos_k,
            eig_k=eig_k,
            nmax_value=nmax_value,
            nstep_value=nstep_value,
            total_time=total_time,
            w_ac_value=w_ac_value,
        )
    if method == "midpoint_dense":
        return time_evolute_midpoint_dense(ky, Ve, level)
    raise ValueError("method should be 'y6_2_lanczos' or 'midpoint_dense'")

if __name__ == '__main__':
    import time
    print(f"Evolving ky = -0.15 for 1/4 period ({nstep} steps)")
    
    ## runtime
    t0 = time.time()
    c_snaps, t_arr = time_evolute(ky, Ve, level)
    print(f"Done in {time.time()-t0:.1f}s. Shape: {c_snaps.shape}")

    ## check norm (expect ~1.0 throughout)
    norm = np.array([np.linalg.norm(coeffs) for coeffs in c_snaps])
    print(f"Norm: min={norm.min():.6f}, max={norm.max():.6f}")


def compute_prob_evolute(ky, Ve, level, method="y6_2_lanczos"):
    """
    Returns:
        prob  : shape ()
        t_arr : shape (nstep)
        x     : x grid in atomic units
    """
    x = np.linspace(-2**0.5*L, 2**0.5*L, 401)        ## x in au
    c_snaps, t_arr = time_evolute(ky, Ve, level, method=method)
    all_basis = QHO_basis(x, nmax)
    ## shape: c_snaps=(nstep, nmax), all_basis=(nmax, lan(x)), psi_t=(nstep, len(x))
    psi_t = c_snaps @ all_basis
    prob = np.abs(psi_t)**2

    return prob, t_arr, x

if __name__ == '__main__':
    ## quick check: prob should stay normalized over time
    prob, t_arr, x = compute_prob_evolute(ky, Ve, level)
    dx = x[1] - x[0]
    norms = prob.sum(axis=1) * dx
    print(f"Spatial norm: min={norms.min():.4f}, max={norms.max():.4f}")  ## expect ~1.0
