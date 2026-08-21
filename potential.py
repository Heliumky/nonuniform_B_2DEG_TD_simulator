import numpy as np
import matplotlib.pyplot as plt
from params import hbar, e, m_star, B0,  L, AU_TO_NM, AU_TO_MEV

def V(x, ky, Ve=0.0):
    kinetic_part = (hbar * ky + e * B0 * x**2 / (2 * L))**2 / (2 * m_star)
    electric_part = e * Ve * x / L
    return kinetic_part + electric_part

if __name__ == "__main__":
    x = np.linspace(-2**0.5 * L, 2**0.5 * L, 500)
    ky_vals = [0.15 / 18.9, -0.05 / 18.9, -0.15 / 18.9]
    labels = ['ky = 0.15', 'ky = -0.05', 'ky = -0.15']

    for ky, lab in zip(ky_vals, labels):
        plt.plot(x * AU_TO_NM, AU_TO_MEV * V(x, ky), label=lab)

    plt.xlabel('x (nm)')
    plt.ylabel('V(x) (meV)')
    plt.ylim(0, 30)
    plt.legend()
    plt.tight_layout()
    plt.savefig('potential.png', dpi=150)
    plt.show()
