# Generate a ton of data files for use in learning the Lennard-Jones potential function





from hts.multi_agent_kinetics import indicators, forces, integrators, experiments, sim, serialize
import numpy as np
import datetime as dt
import math, random
from multiprocessing import Pool


SIM_COUNT = 1


base_params = {
    'timestep': 0.01,
    'size': 8, # rename to initialization_radius
    'n_agents': 3,
    'n_timesteps': 10000,
    'min_dist': 2, # rename for clarity
    'init_speed': 0
}

true_params = {
    'epsilon': 25,
    'sigma': 1
}

def run_sim_and_write(seed):
    true_params_aug = {**base_params, **true_params}
    world = sim.run_random_circle_lj_sim(true_params_aug, seed)
    serialize.save_world(world, './data', true_params_aug, seed)

with Pool(SIM_COUNT) as p:

    p.map(run_sim_and_write, [random.randint(0, 1000000) for x in range(SIM_COUNT)])