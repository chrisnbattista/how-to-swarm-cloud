import numpy as np

def lennard_jones_potential(epsilon, omega, r):
    return 4 * epsilon * ( (omega / r)**12 - (omega / r)**6 )

def pairwise_world_lennard_jones_potential(world):
    return np.zeros( (world.shape[0], 2) )