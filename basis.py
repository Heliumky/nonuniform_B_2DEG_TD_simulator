import numpy as np
from params import hbar, e, m_star, B0, L, w_c, AU_TO_NM, AU_TO_MEV

#w = 1e-2 * np.sqrt(hbar*e*B0/L**2)
def operators_matrixize(nmax):
    x0 = np.sqrt(hbar / (2*m_star*w_c))     #length scale
    p0 = np.sqrt(m_star*hbar*w_c / 2)      #length scale
    ns = np.arange(nmax)

    ## x matrix: sub&super-diaginal only
    x_mat = np.zeros((nmax, nmax))
    x_mat += np.diag(x0 * np.sqrt(ns[1:]), k=-1)                  #sub
    x_mat += np.diag(x0 * np.sqrt(ns[1:]), k=+1)                  #super

    ## x^2 matrix: diagonal, k±2
    x2_mat = np.zeros((nmax,nmax))
    x2_mat += np.diag(x0**2 * (2*ns+1))                           #diag
    x2_mat += np.diag(x0**2 * np.sqrt(ns[2:]*ns[1:-1]),  k=-2)    #sub^2
    x2_mat += np.diag(x0**2 * np.sqrt(ns[2:]*ns[1:-1]),  k=+2)    #super^2

    ## x^4 matrix: diag, k±2, k±4
    x4_mat = np.zeros((nmax,nmax))
    x4_mat += np.diag(x0**4 * (6*ns**2+6*ns+3))                                    #diag
    x4_mat += np.diag(x0**4 * (4*ns[:-2]+6) * np.sqrt(ns[2:]*ns[1:-1]), k=-2)      #sub^2
    x4_mat += np.diag(x0**4 * (4*ns[:-2]+6) * np.sqrt(ns[2:]*ns[1:-1]), k=+2)      #super^2
    x4_mat += np.diag(x0**4 * np.sqrt(ns[4:]*ns[3:-1]*ns[2:-2]*ns[1:-3]), k=-4)    #sub^4
    x4_mat += np.diag(x0**4 * np.sqrt(ns[4:]*ns[3:-1]*ns[2:-2]*ns[1:-3]), k=+4)    #super^4

    ## p^2 matrix: diagonal, k±2
    p2_mat = np.zeros((nmax,nmax))
    p2_mat += np.diag(p0**2 * (2*ns+1))                            #diag
    p2_mat += np.diag(-p0**2 * np.sqrt(ns[2:]*ns[1:-1]),  k=-2)    #sub^2
    p2_mat += np.diag(-p0**2 * np.sqrt(ns[2:]*ns[1:-1]),  k=+2)    #super^2

    return x0, p0, x_mat, x2_mat, x4_mat, p2_mat

if __name__=='__main__':
    x0, p0, x_mat, x2_mat, x4_mat, p2_mat = operators_matrixize(nmax=20)
    print("x symmetric?", np.allclose(x_mat,x_mat.T))
    print("x^2 symmetric?", np.allclose(x2_mat,x2_mat.T))
    print("x^4 symmetric?", np.allclose(x4_mat,x4_mat.T))
    print("p^2 symmetric?", np.allclose(p2_mat,p2_mat.T))
    print("x0=", f"{AU_TO_NM * x0:.4}")