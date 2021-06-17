# Demonstrate proof-of-concept convergence for gradient descent to learn Lennard-Jones potential parameters
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import datetime as dt
import math, random, itertools

from multi_agent_kinetics import indicators, forces, integrators, experiments, sim

np.seterr(all="ignore")

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
    'seed': 234235374563,
}

## True coefficient values
true_params = {
    'epsilon': 1,
    'sigma': 25,
}

## Hyperparameters for gradient descent
hyperparams = {
    'alpha': 0.1 / FIDELITY,
    'd': 0.1 / FIDELITY,
    'steps': int(11 * FIDELITY)
}

## Gravity option
sim_force = 'lj'
if input("Enter anything to use gravity>"):
    sim_force = 'gravity'
    true_params = {
        'G':(6.674*(10**(-11))),
    }
    base_params = {
        'timestep': 0.01,
        'size': 3, # rename to initialization_radius
        'n_agents': 2,
        'n_timesteps': 10000,
        'min_dist': 1.2, # rename for clarity
        'init_speed': 0,
        'seed': 23353675,
    }
    hyperparams = {
        'alpha': 0.001 / FIDELITY,
        'd': 0.001 / FIDELITY,
        'steps': int(11 * FIDELITY)
    }

## Define the functionality for Monte Carlo
def seed_monte_carlo(true_params, size, steps=10):
    """Given a dictionary of true parameters and a size to vary them by, returns all combinations within that range."""
    ranges = {}
    for param in true_params.keys():
        ranges[param] = np.linspace(true_params[param] - size, true_params[param] + size, steps)
    return list(itertools.product(*ranges.values()))

def run_monte_carlo(seed_combinations, optimizer, base_params, true_params, initial_guess_params, hyperparams, accuracy, search_diameter=10):
    """Given a NumPy array with a column for each parameter, world params, an optimizer function, hyperparams for that function,
        and an accuracy at which the algorithm is considered to have converged, runs a Monte Carlo analysis
        and returns a Series of binary converged/didn't for each parameter combination."""
    return run_one_monte_carlo_sample(base_params, true_params, initial_guess_params, hyperparams)

def run_one_monte_carlo_sample(base_params, true_params, initial_guess_params, hyperparams):
    """Run gradient descent algorithm from guess param values to find real param values."""

    if sim_force == 'lj':
        true_force = lambda world, context: forces.pairwise_world_lennard_jones_force(world, epsilon=true_params['epsilon'], sigma=true_params['sigma'])
    elif sim_force  == 'gravity':
        true_force = lambda world, context: forces.newtons_law_of_gravitation(world, true_params['G'], context)
    
    ## Reference (true) world
    true_params_aug = {**base_params, **true_params}
    true_world = sim.run_random_circle_sim(
        true_params_aug,
        base_params['seed'],
        forces=[
            true_force
        ])

    ## Prep test parameters
    learned_params = deepcopy(initial_guess_params)

    for i in range(hyperparams['steps']):
        ## Run test sim
        learned_params_aug = {**base_params, **learned_params}
        if sim_force == 'lj':
            test_force = lambda world, context: forces.pairwise_world_lennard_jones_force(world, epsilon=learned_params['epsilon'], sigma=learned_params['sigma'])
        elif sim_force  == 'gravity':
            test_force = lambda world, context: forces.newtons_law_of_gravitation(world, learned_params['G'], context)
    
        test_world = sim.run_random_circle_sim(
            learned_params_aug,
            base_params['seed'],
            forces=[
                test_force
                ])
        cost = indicators.mse_trajectories(true_world.get_history(), test_world.get_history(), base_params['n_agents'])
        if i % 10 * FIDELITY == 0:
            if sim_force == 'lj':
                print(f'Error: {cost}\tEps: {learned_params["epsilon"]:.2f}\tSigma: {learned_params["sigma"]:.2f}')
        
        ## Manual GD
        for p in learned_params.keys():

            plus_d_params = {**learned_params_aug, **{p: learned_params[p] + hyperparams['d']}}
            if sim_force == 'lj':
                plus_d_force = lambda world, context: forces.pairwise_world_lennard_jones_force(world, epsilon=plus_d_params['epsilon'], sigma=plus_d_params['sigma'])
            elif sim_force  == 'gravity':
                plus_d_force = lambda world, context: forces.newtons_law_of_gravitation(world, plus_d_params['G'], context)
            plus_d_world = sim.run_random_circle_sim(
                plus_d_params,
                base_params['seed'],
                forces=[
                    plus_d_force
                ])

            minus_d_params = {**learned_params_aug, **{p: learned_params[p] - hyperparams['d']}}
            if sim_force == 'lj':
                minus_d_force = lambda world, context: forces.pairwise_world_lennard_jones_force(world, epsilon=minus_d_params['epsilon'], sigma=minus_d_params['sigma'])
            elif sim_force  == 'gravity':
                minus_d_force = lambda world, context: forces.newtons_law_of_gravitation(world, minus_d_params['G'], context)
            minus_d_world = sim.run_random_circle_sim(
                minus_d_params,
                base_params['seed'],
                forces=[
                    minus_d_force
                ])

            plus_d_cost = indicators.mse_trajectories(true_world.get_history(), plus_d_world.get_history(), base_params['n_agents'])
            minus_d_cost = indicators.mse_trajectories(true_world.get_history(), minus_d_world.get_history(), base_params['n_agents'])

            d_param = (plus_d_cost - minus_d_cost)
            learned_params[p] -= hyperparams['alpha'] * math.copysign(1, d_param)
        
    return learned_params

## Preview the cost landscape
if input("Enter any input to generate cost landscape >"):

    size = float(input("What radius?>"))
    combinations = seed_monte_carlo(true_params, size)

    ## Reference (true) world
    true_params_aug = {**base_params, **true_params}
    true_world = sim.run_random_circle_lj_sim(true_params_aug, base_params['seed'])

    cost_data = np.empty( (len(combinations), 3) )
    for combo in tqdm(range(len(combinations))):
        test_params = {**base_params, **{'sigma': combinations[combo][0], 'epsilon': combinations[combo][1]}}
        test_world = sim.run_random_circle_lj_sim(test_params, base_params['seed'])
        loss = indicators.mse_trajectories(test_world.get_history(), true_world.get_history(), base_params['n_agents'])
        cost_data[combo] = combinations[combo] + (loss,)
    
    print(cost_data)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(cost_data[:,0], cost_data[:,1], np.log(cost_data[:,2]))
    ax.set_xlabel('sigma')
    ax.set_ylabel('epsilon')
    ax.set_zlabel("log(cost)")
    plt.tight_layout()
    plt.ion()
    plt.show()
    plt.draw()
    plt.pause(0.001)

if sim_force == 'lj':
    print(f'True Eps: {true_params["epsilon"]:.2f}\tTrue Sigma: {true_params["sigma"]:.2f}')

if not input ("Enter anything to activate Monte Carlo mode>"):
    # single run
    if sim_force == 'lj':
        initial_guess_params = {
            'epsilon': float(input("Epsilon guess>")),
            'sigma': float(input("Sigma guess>"))
        }
    elif sim_force == 'gravity':
        initial_guess_params = {
            'G': float(input("G guess>"))
        }
    run_one_monte_carlo_sample(base_params, true_params, initial_guess_params, hyperparams)
else:
    # Monte Carlo!
    size = float(input("What radius?>"))
    combinations = seed_monte_carlo(true_params, size, steps=3)
    results = []
    for i in range(len(combinations)):
        if sim_force == 'lj':
            print(f"Starting from seed position\tEps: {combinations[i][1]:.2f}\tSigma: {combinations[i][0]:.2f}")
            results.append(
                run_one_monte_carlo_sample(
                    base_params,
                    true_params,
                    {
                        'sigma': combinations[i][0],
                        'epsilon': combinations[i][1]
                    },
                    hyperparams
                )
            )
        elif sim_force == 'gravity':
            print(f"Starting from seed position\tG: {combinations[i][0]:.2f}")
            results.append(
                run_one_monte_carlo_sample(
                    base_params,
                    true_params,
                    {
                        'G': combinations[i][0]
                    },
                    hyperparams
                )
            )
        print(f'Final: {results[-1]}')
    df = pd.DataFrame(results)
    if sim_force == 'lj':
        plt.scatter(
            df.index,
            df['sigma']
        )
        plt.scatter(
            df.index,
            df['epsilon']
        )
    elif sim_force == 'gravity':
        plt.scatter(
            df.index,
            df['G']
        )
    plt.show()
    plt.draw()
    plt.pause(0.001)
    input()