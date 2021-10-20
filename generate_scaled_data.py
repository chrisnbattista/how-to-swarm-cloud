# Generate a ton of data files for use in learning the gravity function

from multi_agent_kinetics import indicators, forces, integrators, experiments, sim, serialize, potentials, noise, worlds
import numpy as np
import torch
import datetime as dt
import math, random, time
from multiprocessing import Pool
from tqdm import tqdm

np.seterr(all="ignore")

## TODO: downsample high fidelity data set
## TODO: quantify numerical error using optimal/correct solver discrepancy
## maybe find reference data?
## TODO: scale down / nondimensionalize using this technique https://archive.siam.org/books/textbooks/MM13Sample.pdf

SIM_COUNT = 100

do_decay = input("Decay [y]/[N] >")
if do_decay == 'y':
    path = '/two_particle/scaled-gravity-decay'
    decay_force = [lambda w,c: forces.viscous_damping_force(world=w, c=float(0.1), context=c)]
else:
    path = '/two_particle/scaled-gravity'
    decay_force = []
    do_noise = input("Noise [y]/[N] >")
    if do_noise == 'y':
        path = '/two_particle/scaled-gravity-noise'

# TODO: try larger delta t, but longer trajectory. to show nonlinear behavior
# TODO: at least one period
base_params = {
    'timestep': 0.02, #  of orbital period
    'n_timesteps': 10000, #  period
    'separation': 384.4, # earth-moon distance
    'small_m': 7.0, # moon mass
    'large_m': 600.0, # earth mass,
    'tangent_speed': 1.0, # moon orbital velocity,
    'std': 0.1,
    'mean': 0.1
}

true_params = {
    'G':1,
}

def add_noise(world):
    hist = world.get_history()
    pos_hist = hist[:, worlds.pos[world.spatial_dims]]
    vel_hist = hist[:, worlds.vel[world.spatial_dims]]
    pos_mean = torch.abs(torch.mean(pos_hist) / 100.0)
    pos_std = pos_mean
    vel_mean = torch.abs(torch.mean(vel_hist) / 100.0)
    vel_std = vel_mean
    world.history[:, worlds.pos[world.spatial_dims]] = noise.gaussian_noise(pos_hist, pos_mean, pos_std)
    world.history[:, worlds.vel[world.spatial_dims]] = noise.gaussian_noise(vel_hist, vel_mean, vel_std)

def run_sim_and_write(seed):
    true_params_aug = {**base_params, **true_params}
    world = sim.run_two_body_sim(
        true_params_aug,
        seed,
        forces=[
            lambda world, context: forces.newtons_law_of_gravitation(world=world, G=true_params['G'], context=context)
        ]+decay_force,
        indicators=[
            lambda w: indicators.hamiltonian(w, global_potentials=[lambda w: potentials.gravitational_potential_energy(w, G=true_params['G'])]),
            lambda w: potentials.gravitational_potential_energy(w, G=true_params_aug['G'])
            ],
        indicator_schema=['Hamiltonian', 'GPE'],
    )
    add_noise(world)
    serialize.save_world(world, './data'+path, true_params_aug, seed)

if __name__ == '__main__':
    for i in tqdm(range(SIM_COUNT)):
        run_sim_and_write(time.time())