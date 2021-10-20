# Generate a ton of data files for use in learning the Lennard-Jones potential function

from multi_agent_kinetics import indicators, forces, integrators, experiments, sim, serialize
import numpy as np
import datetime as dt
import math, random
from multiprocessing import Pool
from tqdm import tqdm

SIM_COUNT = 1000

n = 3
path = ''
if input("two_particles?>"):
    n = 2
    path = '/two_particle_pt'

base_params = {
    'timestep': 0.001,
    'size': 3, # rename to initialization_radius
    'n_agents': n,
    'n_timesteps': 1000,
    'min_dist': 1.2, # rename for clarity
    'init_speed': 0
}

true_params = {
    'epsilon': 25,
    'sigma': 1
}

def run_sim_and_write(seed):
    true_params_aug = {**base_params, **true_params}
    world = sim.run_random_circle_sim(
        true_params_aug,
        seed,
        forces=[
            lambda world, context: forces.pairwise_world_lennard_jones_force(world, epsilon=true_params['epsilon'], sigma=true_params['sigma'])
        ])
    serialize.save_world(world, './data'+path, true_params_aug, seed)

if __name__ == '__main__':
    for i in tqdm(range(SIM_COUNT)):
        run_sim_and_write(random.randint(0, 32235134542))