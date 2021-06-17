# Generate a ton of data files for use in learning the gravity function

from multi_agent_kinetics import indicators, forces, integrators, experiments, sim, serialize
import numpy as np
import datetime as dt
import math, random
from multiprocessing import Pool
from tqdm import tqdm

np.seterr(all="ignore")

SIM_COUNT = 100

n = 3
path = '/two_particle/gravity'

base_params = {
    'timestep': 0.1,
    'size': 30, # rename to initialization_radius
    'n_agents': n,
    'n_timesteps': 10000,
    'min_dist': 12, # rename for clarity
    'init_speed': 0,
    'mass': 10**(9)
}

true_params = {
    'G':(6.674*(10**(-11))),
}

def run_sim_and_write(seed):
    true_params_aug = {**base_params, **true_params}
    world = sim.run_random_circle_sim(
        true_params_aug,
        seed,
        forces=[
            lambda world, context: forces.newtons_law_of_gravitation(world, true_params['G'], context)
        ]
    )
    serialize.save_world(world, './data'+path, true_params_aug, seed)

if __name__ == '__main__':
    for i in tqdm(range(SIM_COUNT)):
        run_sim_and_write(random.randint(0, 32235134542))