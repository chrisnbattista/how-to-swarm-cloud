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

def mse_trajectories(reference_trajectories, test_trajectories, n_particles):
    '''
    Calculates the aggregate mean squared error between two sets of particle
    trajectories, with the sets containing an arbitrary but corresponding
    number of particles.
    '''

    differences = test_trajectories[:, 1:3] - reference_trajectories[:, 1:3]
    norms = np.linalg.norm(differences, axis=1)
    return np.sum(norms**2) / n_particles
