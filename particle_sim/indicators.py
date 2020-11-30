import numpy as np
import numexpr as ne

def potential_energy(world, potentials):
    '''
    Calculates potential energy of a whole particle system.
    '''

    # Requires a list of all potentials, which are to return scalar
    # global values
    V = 0

    for potential in potentials:
        v = potential(world)
        V += v

    return V

def kinetic_energy(world):
    '''
    Calculates kinetic energy of a whole particle system.
    '''

    vel_mags = np.linalg.norm(world[:,4:6], axis=1)
    T = np.sum( ((vel_mags**2) * world[:,3] / 2),
                axis=0)

    return T

def hamiltonian(world, potentials=[]):
    '''
    Calculate Hamiltonian (kinetic and potential energy) of entire particle system.
    '''

    return \
        kinetic_energy(world) + \
        potential_energy(world, potentials)
