import numpy as np
from scipy.linalg import eigh
from params import hbar, e, B0, m_star, L, w_c, AU_TO_MEV, AU_TO_NM
from basis import operators_matrixize

def build_H(ky, Ve, nmax):
    x0, p0, x_mat, x2_mat, x4_mat, p2_mat = operators_matrixize(nmax)
    ns = np.arange(nmax)

    ## coefficient
    c_p2 = 1 / (2*m_star)  # ~p^2
    c_x = e*Ve / L         # ~x
    c_x2 = hbar*ky*e*B0 / (2*m_star*L)       # ~x^2
    c_x4 = e**2*B0**2 / (8*m_star*L**2)      # ~x^4
    c_x0 = hbar**2*ky**2 / (2*m_star)        # ~x^0

    return c_p2 * p2_mat + c_x * x_mat + c_x2 * x2_mat + c_x4 * x4_mat + c_x0 * np.eye(nmax)

if __name__ == '__main__':
    H = build_H(ky=(-0.15*AU_TO_NM), Ve=0.5 * hbar * w_c / e, nmax=51)
    evals, _ = eigh(H)
    print("5 lowest eigenvalues(meV):")
    print(evals[:5]*AU_TO_MEV)
    
