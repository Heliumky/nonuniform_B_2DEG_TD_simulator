import numpy as np

#parameters are defined in atomic unit(au)
AU_TO_MEV =     2.7e4                    # au to meV
AU_TO_NM  =     5.29e-2                  # au to nm
hbar      =     1                        # au
e         =     1                        # au
me        =     1                        # au

m_star    =     0.067 * me               # au
B0        =     1.65/(2.35e5)            # Tesla to au
w_c       =     e * B0 / m_star          # cyclotron_frequency
L         =     16 * np.pi**2/AU_TO_NM   # nm to au (16pi^2nm)

if __name__ == "__main__":
    print(f"w_c = {w_c:.4f} rad/s")
    print(f"hbar*w_c = {AU_TO_MEV * B0/m_star:.4f} meV")
