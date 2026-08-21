import numpy as np
from scipy.linalg import eigh
from params import hbar, B0, m_star, AU_TO_MEV, AU_TO_NM
from hamiltonian import build_H
from lanczos import lanczos_eigenpairs


def compute_spectrum(
    ky_array,
    Ve,
    nmax,
    nlevel,
    method,
    lanczos_k=60,
    tol=1e-10,
    warm_start=True,
):
    """
    ky_array  : 1D array of ky grid
    Ve        : electric potential
    nmax      : number of basis ## = 51 for trial
    nlevel    : number of displayed band
    method    : "lanczos" uses scipy.sparse.linalg.eigsh; "dense" uses eigh
    lanczos_k : Krylov subspace dimension for eigsh
    """
    evals = []
    ground_v0 = np.zeros(nmax)
    ground_v0[0] = 1.0

    for ky in ky_array:
        H = build_H(ky * AU_TO_NM, Ve, nmax)
        if method == "dense":
            evals_all, _ = eigh(H)
        elif method == "lanczos":
            psi0 = ground_v0 if warm_start and nlevel == 1 else None
            evals_all, evecs = lanczos_eigenpairs(
                H,
                nlevel=nlevel,
                k=lanczos_k,
                tol=tol,
                psi0=psi0,
            )
            if warm_start and nlevel == 1:
                ground_v0 = evecs[:, 0]
        else:
            raise ValueError("method should be 'lanczos' or 'dense'")
        evals.append(evals_all[:nlevel])

    return np.array(evals) * AU_TO_MEV      ## shape: (len(ky_array), nlevel)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ky_array = np.linspace(-0.5, 0.5, 101)       ## tunable: ky grid
    ne = 0.5                                    ## tunable: electric field magnitude
    nmax = 601                                  ## tunable: number of basis
    nlevel = 12                                 ## tunable: number of bands
    method = "lanczos"                          ## tunable: "lanczos" or "dense"
    lanczos_k = 30         
    tol = 1e-10                     ## tunable: Krylov dimension
    evals = compute_spectrum(
        ky_array,
        Ve=ne * (hbar * B0 / m_star),
        nmax=nmax,
        nlevel=nlevel,
        method=method,
        lanczos_k=lanczos_k,
        tol=tol,
    )
    print(
        "Computing spectrum ",
        "| ky grid:", evals.shape[0],
        "| basis number:", nmax,
        "| bands:", nlevel,
        "| method:", method,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    for band in range(evals.shape[1]):
        ## evals.shape[0]: len(ky_array);
        ## evals.shape[1]: nlevel
        ax.plot(ky_array, evals[:, band], color='steelblue', lw=1.2)
        ## evals[:, band]: every ky, certain (band) level

    ax.set_xlabel('$k_y$(nm$^-1$)')
    ax.set_ylabel('Energy(meV)')
    ax.set_title('Fig. 2(b)')
    ax.set_ylim(-5, 10)
    ax.axvline(0, color='gray', lw=0.5, ls='dashed')
    ax.axhline(0, color='gray', lw=0.5, ls='solid')
    plt.tight_layout()
    plt.savefig('spectrum_Ve0.png', dpi=150)
    plt.show()
