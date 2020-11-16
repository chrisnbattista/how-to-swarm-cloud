import numpy as np
import scipy, math, itertools
from scipy.sparse import csr_matrix
from scipy.special import comb
from sklearn.preprocessing import normalize

conversion_matrices = {}
def make_conversion_matrix(size):
    if not size in conversion_matrices.keys():
        conversion_matrices[size] = csr_matrix( (size, int(comb(size, 2))) )
        for i in range(size):
            print(conversion_matrices[size][i,:].shape)
            input(np.tile( [0]*i + [1], size ).shape)
            conversion_matrices[size][ i, 0:size*(i+1) ] = np.tile( [0]*i + [1], size )

    return conversion_matrices[size]

def lennard_jones_potential(epsilon, omega, r):
    return (4 * epsilon * omega**12)/r**12 - (4 * epsilon * omega**6)/r**6

def pairwise_world_lennard_jones_potential(world, epsilon, omega):
    '''
    Not timestep dependent as it is not time dependent as it is a potential field.
    '''

    # isolate particle position data
    coords = world[['b_1', 'b_2']].to_numpy()

    # calculate pairwise distances
    distances = scipy.spatial.distance.pdist(coords)

    # calculate pairwise potentials
    potentials = lennard_jones_potential(
        epsilon,
        omega,
        distances
    )
    print(potentials)

    # get absolute pairwise displacements
    indexes = np.array(list(itertools.combinations(range(coords.shape[0]), 2)))
    displacements = coords[ indexes[:,0], : ] - coords[ indexes[:,1], : ]

    # previous attempt
    ##chunks = []
    ##for d in range( 0, math.floor(coords.shape[0] / 2) ):
    ##    chunks.append(coords - np.roll(coords, d))

    # normalize displacements using L2-norm
    displacements = normalize(displacements, axis=1)
    
    # scale displacements by potentials
    forces = displacements * potentials[:, np.newaxis]
    ##print(forces)

    # convert to outcome force matrix
    return np.sum( np.apply_along_axis(scipy.spatial.distance.squareform, 0, forces), axis=1 )
    ##return displacements.T * make_conversion_matrix(coords.shape[0])


def slow_pairwise_world_lennard_jones_potential(world, epsilon, omega):
    '''
    '''

    potential_matrix = np.zeros( (world.shape[0], 2) )
    for i in range(world.shape[0]):
        for j in range(world.shape[0]):
            if i == j: continue
            norm = np.linalg.norm(world.iloc[j][['b_1', 'b_2']] - world.iloc[i][['b_1', 'b_2']])
            magnitude = lennard_jones_potential(epsilon, omega, norm)
            direction = ((world.iloc[j] - world.iloc[i])/norm)[['b_1', 'b_2']]
            potential_matrix[j] += direction * magnitude
    
    return potential_matrix

def viscous_damping_force(world, c):
    '''
    F_damping = -cv
    '''

    forces = -c * \
        (
            #np.multiply(
                world[['v_1', 'v_2']].to_numpy()
            #    np.abs(world[['v_1', 'v_2']])
            #)
        )
    return forces