





from particle_sim import indicators, experiments, forces, integrators
import numpy as np
import time, random






def run_sim(params, seed):

    random.seed(seed)

    start_time = time.time()
    last_loop_time = start_time

    indicator_functions = [
        lambda world: indicators.hamiltonian(world, [lambda world: forces.sum_world_lennard_jones_potential(world, params['epsilon'], params['sigma']), lambda world: forces.sum_world_gravity_potential(world, params['lambda'])]),
        lambda world: indicators.kinetic_energy(world),
        lambda world: indicators.potential_energy(world, [lambda world: forces.sum_world_lennard_jones_potential(world, params['epsilon'], params['sigma']), lambda world: forces.sum_world_gravity_potential(world, params['lambda'])]),
    ]

    n_indicators = len(indicator_functions)

    ## ICs
    world = experiments.set_up_experiment(
        radius=params['size']/2,
        center=(params['size']/2, params['size']/2),
        n_particles=params['n_particles'],
        min_dist=params['min_dist'],
        random_speed=params['init_speed'])

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
