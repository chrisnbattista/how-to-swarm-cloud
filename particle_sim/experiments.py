import numpy as np
import random, math

def set_up_experiment(n_particles, radius, center=(0,0), particle_props=[]):
    '''
        n_particles:    initial number of particles []
        radius:         initial radius of particle distribution [m]
    '''

    ## Set up initial conditions (ICs)

    # b_1 ... b_n are position expressed in basis vectors

    # Set state machine standard vars
    # id, b_1, b_2, m, v_1, v_2, t
    #  0   1    2   3   4    5   6
    world_state = np.empty ( (n_particles, 7) )

    # create a random distribution of particles
    for i in range(n_particles):

        smallest_interparticle_distance = 0

        while smallest_interparticle_distance < 0.1:
            theta = random.random() * 2 * math.pi
            r = random.random() * radius
            candidate_b_1 = center[0] + r * math.cos(theta) + radius
            candidate_b_2 = center[1] + r * math.sin(theta) + radius
            test_df = world_state[:i, 1:3] - [candidate_b_1, candidate_b_2]
            norms = np.linalg.norm( test_df[:i, 1:3], axis=1 )
            ##print(norms)
            if i > 0:
                smallest_interparticle_distance = norms.min()
            else:
                break
            ##print(norms.min())

        world_state[i, :] = (i, candidate_b_1, candidate_b_2, 10, 0, 0, 0)

    return world_state

def advance_timestep(world, timestep, integrator, forces=[]):
    '''
        world:              state machine dataframe containing all particles [pd.Dataframe]
        timestep:           length of time to integrate over [s]
        pairwise_forces:    potentials to be calculated between each unique pair of particles and applied as forces
    '''

    ## Calculate forces

    # Initialize matrix to hold forces keyed to id
    force_matrix = np.zeros ( (world.shape[0], 2) )

    for force in forces:
        force_matrix += force(world)

    ## Advance the timestep itself
    world[6] += timestep

    ## Integrate forces over timestep
    return integrator(world, force_matrix, timestep)
