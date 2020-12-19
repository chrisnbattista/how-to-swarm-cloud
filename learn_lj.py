# Demonstrate proof-of-concept convergence for gradient descent to learn Lennard-Jones potential parameters





from particle_sim import indicators, forces, integrators, experiments, sim
import numpy as np
import datetime as dt
import math, random





base_params = {
    'timestep': 0.01,
    'size': 800, # rename to initialization_radius
    'n_particles': 20,
    'n_steps': 1000,
    'min_dist': 20, # rename for clarity
    'init_speed': 50,
    'c': 0.01,
    'lambda': 0.01,
    'record_sparsity':10,
    'seed': 234235374563,
}

learned_params = {
    'epsilon': 1,
    'sigma': 25
}

true_params = {
    'epsilon': 1,
    'sigma': 25
}

true_params_aug = {**base_params, **true_params}
true_trajs, true_inds = sim.run_sim(true_params_aug, base_params['seed'])

hyperparams = {
    'alpha': 0.1,
    'd': 0.1
}

for i in range(100):

    ### Test

    ## Prep test parameters
    learned_params_aug = {**base_params, **learned_params}

    ## Run test sim
    test_trajs, test_inds = sim.run_sim(learned_params_aug, base_params['seed'])

    cost = indicators.mse_trajectories(true_trajs, test_trajs, base_params['n_particles'])
    print(f'Error: {cost}\tEps: {learned_params["epsilon"]:.2f}\tSigma: {learned_params["sigma"]:.2f}')

    ### Update

    ## Compute gradient
    for p in learned_params.keys():

        plus_d_trajs, plus_d_inds = sim.run_sim({**learned_params_aug, **{p: learned_params[p] + hyperparams['d']}}, base_params['seed'])
        minus_d_trajs, minus_d_inds = sim.run_sim({**learned_params_aug, **{p: learned_params[p] - hyperparams['d']}}, base_params['seed'])

        plus_d_cost = indicators.mse_trajectories(true_trajs, plus_d_trajs, base_params['n_particles'])
        minus_d_cost = indicators.mse_trajectories(true_trajs, minus_d_trajs, base_params['n_particles'])

        d_param = (plus_d_cost - minus_d_cost)
        learned_params[p] -= hyperparams['alpha'] * math.copysign(1, d_param)
