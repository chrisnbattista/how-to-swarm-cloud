# Demonstrate proof-of-concept convergence for gradient descent to learn Lennard-Jones potential parameters






import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import datetime as dt
import math, random, itertools

from hts.multi_agent_kinetics import indicators, forces, integrators, experiments, sim





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

if input("Enter any input to generate cost landscape >"):

    sigma_range = np.linspace(22, 28, 25)
    epsilon_range = np.linspace(0.5, 1.5, 25)
    combinations = list(itertools.product(sigma_range, epsilon_range))

    cost_data = np.empty( (len(combinations), 3) )
    
    for combo in tqdm(range(len(combinations))):

        test_params = {**base_params, **{'sigma': combinations[combo][0], 'epsilon': combinations[combo][1]}}
        trajs, inds = sim.run_sim(test_params, base_params['seed'])
        loss = indicators.mse_trajectories(true_trajs, trajs, base_params['n_particles'])
        cost_data[combo] = combinations[combo] + (loss,)
    
    print(cost_data)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(cost_data[:,0], cost_data[:,1], cost_data[:,2])
    ax.set_xlabel('sigma')
    ax.set_ylabel('epsilon')
    ax.set_zlabel("cost")
    plt.show()

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
        