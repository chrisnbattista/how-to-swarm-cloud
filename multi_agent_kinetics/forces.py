





import numpy as np
import scipy
import numexpr as ne
from sklearn.preprocessing import normalize
from . import properties, kernels

def lennard_jones_potential(epsilon, sigma, r):
    return ne.evaluate('( (4 * epsilon * sigma**12)/r**12 - (4 * epsilon * sigma**6)/r**6 )')


def sum_world_lennard_jones_potential(world, epsilon, sigma):
    '''
    '''

    # isolate particle position data
    coords = world[:,1:3]

    # calculate pairwise potentials and distances
    p_dists = scipy.spatial.distance.pdist(coords)

    return np.sum(lennard_jones_potential(epsilon, sigma, p_dists))


def lennard_jones_force(epsilon, sigma, r):
    return ne.evaluate('24 * epsilon / r * ( (2)*(sigma/r)**12 - (sigma/r)**6 )')


def pairwise_world_lennard_jones_force(world, epsilon, sigma, **kwargs):
    '''
    '''

    # isolate particle position data
    coords = world[:,1:3]

    # calculate pairwise forces and distances
    p_dists = scipy.spatial.distance.pdist(coords)
    forces = scipy.spatial.distance.squareform(
            lennard_jones_force(epsilon, sigma, p_dists) / p_dists
    )

    # get absolute pairwise displacements
    ##size = coords.shape[0]
    differences = (coords[:, np.newaxis] - coords)
    ##differences = normalize(differences.reshape(-1, 2)).reshape(size, size, 2)

    # slow draft code
    # # normalize displacements using L2-norm
    # differences = np.nan_to_num(
    #     differences / \
    #         np.sqrt(
    #             np.sum(
    #                 np.power(
    #                     differences,
    #                     2
    #                 ),
    #                 axis=2
    #             )
    #         )[:, :, np.newaxis]
    # )

    # scale displacements by potentials
    expanded_forces = forces[:, :, np.newaxis]
    return ne.evaluate('sum(differences * expanded_forces, axis=1)')

# def slow_pairwise_world_lennard_jones_potential(world, epsilon, sigma):
#     '''
#     '''
#
#     potential_matrix = np.zeros( (world.shape[0], 2) )
#     for i in range(world.shape[0]):
#         for j in range(world.shape[0]):
#             if i == j: continue
#             norm = np.linalg.norm(world.iloc[j][['b_1', 'b_2']] - world.iloc[i][['b_1', 'b_2']])
#             magnitude = lennard_jones_potential(epsilon, sigma, norm)
#             direction = ((world.iloc[j] - world.iloc[i])/norm)[['b_1', 'b_2']]
#             potential_matrix[j] += direction * magnitude
#
#     return potential_matrix


def viscous_damping_force(world, c, **kwargs):
    '''
    F_damping = -cv
    '''

    return -c * world[:, 4:6]


def gravity_well(world, lamb, **kwargs):
    '''
    Exerts a constant gravitational force from the origin - assuming
    the ground level simplified form of constant acceleration gravity.
    '''

    return -lamb / world[:, 3, None] * world[:, 1:3]


def sum_world_gravity_potential(world, lamb, **kwargs):
    '''
    '''

    G = np.sum( np.abs( 0.5 * lamb * np.linalg.norm(world[:, 1:3], axis=1)**2 / world[:, 3] ) )
    ##print(G)
    return G


### SPH Section

def F_pressure(world, d_kernel, h):
    '''
    '''

    ## Accumulator
    F_p = np.zeros((world.shape[0],2))

    ## Rho
    densities = np.zeros((world.shape[0], 1))
    for i in range(world.shape[0]):
        densities[i] = properties.density(world, i, h)
    
    ## r
    p_dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))

    ## F_p
    for i in range(world.shape[0]):
        for j in range(world.shape[0]):
            if i == j: continue
            F_p[i] += \
                world[i, 3] * world[j, 3] \
                * (densities[i] / densities[j]**2 + densities[j] / densities[j]**2) \
                * d_kernel(p_dists[i,j], h)
    
    return F_p

def F_viscosity(world, dd_kernel, h, eta, visc):
    '''
    '''

    ## Accumulator
    F_v = np.zeros((world.shape[0],2))
    
    ## r
    p_dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))

    ## F_v
    for i in range(world.shape[0]):
        for j in range(world.shape[0]):
            if i == j: continue
            F_p[i] += \
                world[i, 3] * world[j, 3] \
                * (densities[i] / densities[j]**2 + densities[j] / densities[j]**2) \
                * d_kernel(p_dists[i,j], h)
    
    ## eta
    return F_v * eta

def pressure_force(i, state):
    '''
    Computes the pressure force for one particle from the world state
    i = particle index
    state = world state matrix
    '''

    densities = properties.density_all(state)

    # isolate particle position data
    coords = state[:,1:3]
    p_dists = scipy.spatial.distance.pdist(coords)
    dists_i = scipy.spatial.distance.squareform(
        p_dists
    )[:,i]
    print("distances")
    print(dists_i)

    # https://lucasschuermann.com/writing/implementing-sph-in-2d
    # HARDCODED MASS, PRESSURE
    mass = 10
    pressures = np.full(state.shape[0], 0.00001)

    # m * p / rho matrix
    pairwise_force_mags = np.nan_to_num(
        np.multiply(
            -1 * \
            mass**2 * \
            (pressures / densities**2) + (pressures[i] / densities[i]**2),
            [kernels.cubic_spline_grad(d) for d in dists_i]
        )
    )

    print("pairwise force mags")
    print(pairwise_force_mags)

    differences = (coords[:, np.newaxis] - coords)[i]
    print(differences)
    forces = differences * pairwise_force_mags[:, None]
    
    return np.sum(
        np.delete(
            forces,
            i
        )
    )

def world_pressure_force(world):
    '''
    Apply the pressure force to all particles.
    '''
    force_accumulator = np.zeros((world.shape[0], 2))
    for i in range(world.shape[0]):
        force_accumulator[i,:] += pressure_force(i, world)
    
    print("forces")
    print(force_accumulator)
    return force_accumulator

def viscosity_force(i, state, nu=1):
    '''
    Computes the viscosity force for one particle from the world state
    i = particle index
    state = world state matrix
    '''

    densities = properties.density_all(state)

    # isolate particle position data
    coords = state[:,1:3]
    p_dists = scipy.spatial.distance.pdist(coords)
    dists_i = scipy.spatial.distance.squareform(
        p_dists
    )[:,i]
    print("distances")
    print(dists_i)

    # https://lucasschuermann.com/writing/implementing-sph-in-2d
    # HARDCODED MASS, PRESSURE
    mass = 10
    nu = 1

    velocities = np.linalg.norm(state[:, 3:5], axis=1)

    # m * p / rho matrix
    pairwise_force_mags = np.nan_to_num(
        np.multiply(
            nu * \
            mass * \
            np.abs(velocities - velocities[i]) / densities,
            [kernels.cubic_spline_grad_double(d) for d in dists_i]
        )
    )

    print("pairwise force mags")
    print(pairwise_force_mags)

    differences = (coords[:, np.newaxis] - coords)[i]
    print(differences)
    forces = differences * pairwise_force_mags[:, None]
    
    return np.sum(
        np.delete(
            forces,
            i
        )
    )

     # finish

    return nu * np.sum(10 * math.abs(velocities - velocities[i])/densities * kernels.cubic_spline_grad_double(dists_i))

def world_viscosity_force(world):
    '''
    Apply the viscosity force to all particles.
    '''
    force_accumulator = np.zeros((world.shape[0], 2))
    for i in range(world.shape[0]):
        force_accumulator[i,:] += viscosity_force(i, world)
    
    print("forces")
    print(force_accumulator)
    return force_accumulator