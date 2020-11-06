import pandas as pd
import numpy as np
import random, math

def set_up_experiment(n_particles, radius, particle_props=[]):
    '''
        n_particles:    initial number of particles []
        radius:         initial radius of particle distribution [m]
    '''

    ## Set up initial conditions (ICs)

    # b_1 ... b_n are position expressed in basis vectors

    # Set state machine standard vars
    world_state = pd.DataFrame(
        {
            'id':   [],
            'b_1':    [],
            'b_2':    [],
            't':    []
        }
    )

    # set additiona vars
    for prop in particle_props:
        world_state[prop] = []

    # create a random distribution of particles
    for i in range(n_particles):
        theta = random.random() * 2 * math.pi
        r = random.random() * radius
        world_state = world_state.append(
            {
                'id':       i,
                'b_1':      r * math.cos(theta),
                'b_2':      r * math.sin(theta),
                't':        0
            },
            ignore_index=True
        )
    
    world_state['id'] = world_state['id'].astype(int)

    return world_state.set_index('id')

def advance_timestep(world, timestep, integrator, pairwise_forces=[]):
    '''
        world:              state machine dataframe containing all particles [pd.Dataframe]
        timestep:           length of time to integrate over [s]
        pairwise_forces:    potentials to be calculated between each unique pair of particles and applied as forces
    '''

    ## Calculate forces

    # Initialize matrix to hold forces keyed to id
    force_matrix = pd.DataFrame(
                        index=world.index
                        )
    force_matrix[['b_1', 'b_2']] = np.zeros( (len(world.index), 2) )

    for force in pairwise_forces:
        force_matrix += force(world)
    
    ## Advance the timestep itself
    world['t'] += timestep
    
    ## Integrate forces over timestep
    return integrator(world, force_matrix, timestep)
