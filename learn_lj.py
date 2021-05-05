# Demonstrate proof-of-concept convergence for gradient descent to learn Lennard-Jones potential parameters

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import datetime as dt
import math, random, itertools

from multi_agent_kinetics import indicators, forces, integrators, experiments, sim

## A general setting for how fine-grained the simulation is. Higher = more fine grained
FIDELITY = 1

## Physical parameters for the simulation
base_params = {
    'timestep': 0.01,
    'size': 800, # rename to initialization_radius
    'n_agents': 20,
    'n_timesteps': 1000,
    'min_dist': 20, # rename for clarity
    'init_speed': 50,
    'c': 0.01,
    'lambda': 0.01,
    'record_sparsity':10,
    'seed': 234235374563,
}

## True coefficient values
true_params = {
    'epsilon': 1,
    'sigma': 25
}

## Reference (true) world
true_params_aug = {**base_params, **true_params}
true_world = sim.run_random_circle_lj_sim(true_params_aug, base_params['seed'])

## Preview the cost landscape
if input("Enter any input to generate cost landscape >"):

    sigma_range = np.linspace(22, 28, 25)
    epsilon_range = np.linspace(0.5, 1.5, 25)
    combinations = list(itertools.product(sigma_range, epsilon_range))

    cost_data = np.empty( (len(combinations), 3) )
    
    for combo in tqdm(range(len(combinations))):

        test_params = {**base_params, **{'sigma': combinations[combo][0], 'epsilon': combinations[combo][1]}}
        test_world = sim.run_random_circle_lj_sim(test_params, base_params['seed'])
        loss = indicators.mse_trajectories(test_world.get_history(), true_world.get_history(), base_params['n_agents'])
        cost_data[combo] = combinations[combo] + (loss,)
    
    print(cost_data)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(cost_data[:,0], cost_data[:,1], cost_data[:,2])
    ax.set_xlabel('sigma')
    ax.set_ylabel('epsilon')
    ax.set_zlabel("cost")
    plt.show()


## Hyperparameters for gradient descent
hyperparams = {
    'alpha': 0.1 / FIDELITY,
    'd': 0.1 / FIDELITY
}

if not input ("Enter anything to activate Monte Carlo mode>"):
    learned_params = [{
        'epsilon': float(input("Epsilon guess>")),
        'sigma': float(input("Sigma guess>"))
    }]
    loops = 1


for j in range(loops):
    ## Run gradient descent algorithm from guess param values to find real param values
    for i in range(100 * FIDELITY):

        ### Test

        ## Prep test parameters
        learned_params_aug = {**base_params, **learned_params[j]}

        ## Run test sim
        test_world = sim.run_random_circle_lj_sim(learned_params_aug, base_params['seed'])

        cost = indicators.mse_trajectories(true_world.get_history(), test_world.get_history(), base_params['n_agents'])
        if i % 10 * FIDELITY == 0:
            print(f'Error: {cost}\tEps: {learned_params[j]["epsilon"]:.2f}\tSigma: {learned_params[j]["sigma"]:.2f}')

        ### Update

        ## Compute gradient
        for p in learned_params[j].keys():

            plus_d_world = sim.run_random_circle_lj_sim({**learned_params_aug, **{p: learned_params[j][p] + hyperparams['d']}}, base_params['seed'])
            minus_d_world = sim.run_random_circle_lj_sim({**learned_params_aug, **{p: learned_params[j][p] - hyperparams['d']}}, base_params['seed'])

            plus_d_cost = indicators.mse_trajectories(true_world.get_history(), plus_d_world.get_history(), base_params['n_agents'])
            minus_d_cost = indicators.mse_trajectories(true_world.get_history(), minus_d_world.get_history(), base_params['n_agents'])

            d_param = (plus_d_cost - minus_d_cost)
            learned_params[j][p] -= hyperparams['alpha'] * math.copysign(1, d_param)
            