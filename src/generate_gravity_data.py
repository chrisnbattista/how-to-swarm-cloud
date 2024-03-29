# Generate a ton of data files for use in learning the gravity function

from multi_agent_kinetics import indicators, forces, integrators, experiments, sim, serialize, potentials
import numpy as np
import datetime as dt
import math, random
from multiprocessing import Pool
from tqdm import tqdm
import sys

np.seterr(all="ignore")

SIM_COUNT = 100

if len(sys.argv) > 1: spatial_dims = int(sys.argv[1])
else: spatial_dims = 3
n = 3
path = f'../data/{spatial_dims}D/scaled-gravity'

base_params = {
    'timestep': 0.1,
    'size': 1500, # rename to initialization_radius
    'n_agents': n,
    'n_timesteps': 10000 * 5,
    'min_dist': 900, # rename for clarity
    'init_speed': 0.6283185307179586,
    'mass': 401.04339263553965 / 2
}

true_params = {
    'G':1
}

def run_sim_and_write(seed):
    true_params_aug = {**base_params, **true_params}
    world = sim.run_random_circle_sim(
        true_params_aug,
        seed,
        forces=[
            lambda world, context: forces.newtons_law_of_gravitation(world=world, G=true_params['G'], context=context)
        ],
        indicators=[lambda w: indicators.hamiltonian(w, global_potentials=[lambda w: potentials.gravitational_potential_energy(w, G=true_params['G'])])],
        indicator_schema=['Hamiltonian'],
        spatial_dims=spatial_dims
    )
    serialize.save_world(world, path, true_params_aug, seed)

if __name__ == '__main__':
    for i in tqdm(range(SIM_COUNT)):
        run_sim_and_write(random.randint(0, 32235134542))