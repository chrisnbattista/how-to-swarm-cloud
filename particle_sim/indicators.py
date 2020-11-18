import numpy as np
import numexpr as ne

def hamiltonian(world):
    '''
    Calculate Hamiltonian of entire particle system.
    '''
    vel_mags = np.linalg.norm(world[:,4:6], axis=1)
    T = np.sum( ((vel_mags**2) * world[:,3] / 2),
                axis=0)
    V = 0
    return T + V
