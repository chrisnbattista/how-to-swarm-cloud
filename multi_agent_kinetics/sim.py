





from . import indicators, experiments, forces, integrators, worlds
import numpy as np
import time, random




def run_random_circle_lj_sim(params, seed):
    '''
    Sets up a random experiment using the random seed and params and runs it to completion.
    '''

    random.seed(seed)

    ## ICs
    initial_state = experiments.initialize_random_circle(
        radius=params['size']/2,
        center=(params['size']/2, params['size']/2),
        n_particles=params['n_agents'],
        min_dist=params['min_dist'],
        random_speed=params['init_speed'])

    ## Create World object
    world = worlds.World(
        initial_state=initial_state,
        n_agents=params['n_agents'],
        n_timesteps=params['n_timesteps'],
        timestep=params['timestep'],
        forces=[lambda world: forces.pairwise_world_lennard_jones_force(world, epsilon=params['epsilon'], sigma=params['sigma'])]
    )
    
    ## Forward run
    world.advance_state(params['n_timesteps']-1)
    
    return world




### FOR REFERENCE ONLY - ORIGINAL IMPLEMENTATION
def run_from_ic(world, params):
    '''
    DEPRECATED
    Given an initial system state, runs the simulation forward from that state.
    '''

    start_time = time.time()
    last_loop_time = start_time

    indicator_functions = [
        lambda world: indicators.hamiltonian(world, [lambda world: forces.sum_world_lennard_jones_potential(world, params['epsilon'], params['sigma']), lambda world: forces.sum_world_gravity_potential(world, params['lambda'])]),
        lambda world: indicators.kinetic_energy(world),
        lambda world: indicators.potential_energy(world, [lambda world: forces.sum_world_lennard_jones_potential(world, params['epsilon'], params['sigma']), lambda world: forces.sum_world_gravity_potential(world, params['lambda'])]),
    ]

    n_indicators = len(indicator_functions)

    long_world_history = np.empty( (int(params['n_steps'] * params['n_particles'] / params['record_sparsity']), 7) )
    long_indicator_history = np.empty( (int(params['n_steps'] / params['record_sparsity']), n_indicators + 1) )

    for i in range(params['n_steps']):

        ## Time keeping
        loop_duration = time.time() - last_loop_time
        last_loop_time += loop_duration

        ## Sim step
        world, indicator_results = experiments.advance_timestep(
            world,
            params['timestep'],
            integrators.integrate_rect_world,
            [
                lambda world: forces.pairwise_world_lennard_jones_force(world, epsilon=params['epsilon'], sigma=params['sigma']),
                lambda world: forces.viscous_damping_force(world, params['c']),
                lambda world: forces.gravity_well(world, params['lambda'])
            ],
            indicator_functions
        )

        if (i % params['record_sparsity'] == 0):

            ## Trajectory recording
            long_world_history[ int(i * params['n_particles'] / params['record_sparsity']) : (int(i / params['record_sparsity']) + 1) * params['n_particles'], : ] = world

            ## Indicator recording
            # Index
            long_indicator_history[ int(i / params['record_sparsity']) , 0 ] = i
            # Indicators
            long_indicator_history[ int(i / params['record_sparsity']), 1: ] = indicator_results

    return long_world_history, long_indicator_history
