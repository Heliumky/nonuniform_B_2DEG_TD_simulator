import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermval
from math import factorial
from scipy.linalg import eigh
from params import hbar, B0, L, m_star, w_c, AU_TO_MEV, AU_TO_NM
from hamiltonian import build_H

nmax=201        ## number of basis

def probability_mat(ky, Ve, level):
    H = build_H(ky*AU_TO_NM, Ve, nmax=nmax)
    _, evecs = eigh(H)
    return evecs[:,level]

#print(probability_mat(0.15, 0, [0,1,2]))

def QHO_basis(x, nmax):
    """
    Returns all nmax normalized QHO eigenfunctions evaluated at x,
    using stable upward recursion. Never computes H_n or factorial explicitly.
    shape: (nmax, len(x_au))
    
    Recursion: psi_{n+1} = sqrt(2/(n+1)) * (x/x0_full) * psi_n
                           - sqrt(n/(n+1)) * psi_{n-1}
    """
    x0 = np.sqrt(hbar / (m_star * w_c))
    x = x / x0                       ## dimensionless position
    N = len(x)
    psi = np.zeros((nmax, N))

    ## seed: analytically stable ground and first excited states
    psi[0] = (np.pi * x0**2)**(-0.25) * np.exp(-x**2 / 2)
    if nmax > 1:
        psi[1] = np.sqrt(2) * x * psi[0]

    ## recurse upward
    for n in range(1, nmax - 1):
        psi[n+1] = (np.sqrt(2/(n+1)) * x * psi[n]
                  - np.sqrt(n/(n+1))      * psi[n-1])

    return psi   ## psi[n] is the n-th eigenfunction

def probability(ky, Ve, level, x):
    coeffs = probability_mat(ky, Ve, level) ## len = nmax
    all_bases = QHO_basis(x, nmax) ## nmax * nmax matrix
    
    p0 = coeffs @ all_bases  ## dot product
    return p0**2

if __name__ == '__main__':
    x = np.linspace(-2**0.5 * L, 2**0.5 * L, 201)
    ky = -0.15
    ne = 0
    level = 1
    fig, ax = plt.subplots(figsize=(6, 5))

    prob = probability(ky, ne * (hbar * B0 / m_star), level, x)
    total_probability = np.trapezoid(prob, x)
    print(f"level= {level}, {total_probability:.6f}")

    ax.plot(x * AU_TO_NM, prob, color='blue', lw=1.2)
    ax.set_xlabel('x(nm)')
    ax.set_ylabel('probability')
    ax.set_title('Fig. 2(c)')
    ax.axvline(0, color='gray', lw=0.5, ls='dashed')
    ax.axhline(0, color='gray', lw=0.5, ls='solid')
    plt.tight_layout()
    plt.savefig('Probability_ky0.15.png', dpi=150)
    plt.show()
