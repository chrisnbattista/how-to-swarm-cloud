# Generate a ton of data files for use in learning the Lennard-Jones potential function





from particle_sim import indicators, forces, integrators, experiments, sim
import numpy as np
import datetime as dt
import math, random
from multiprocessing import Pool


SIM_COUNT = 100


base_params = {
    'timestep': 0.01,
    'size': 800, # rename to initialization_radius
    'n_particles': 20,
    'n_steps': 1000,
    'min_dist': 20, # rename for clarity
    'init_speed': 50,
    'c': 0.01,
    'lambda': 0.01,
    'record_sparsity':1
}

true_params = {
    'epsilon': 1,
    'sigma': 25
}

true_params_aug = {**base_params, **true_params}

def run_sim_and_write(seed):
    test_trajs, test_inds = sim.run_sim(true_params_aug, seed)
    np.savetxt(f"./sim_data/lj_{seed}.csv", test_trajs, comments='', delimiter=',', fmt='%10.3f', header='id,b_1,b_2,m,v_1,v_2,t')


with Pool(SIM_COUNT) as p:

    p.map(run_sim_and_write, [random.randint(0, 1000000) for x in range(SIM_COUNT)])