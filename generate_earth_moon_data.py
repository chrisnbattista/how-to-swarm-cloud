# Generate a ton of data files for use in learning the gravity function

from multi_agent_kinetics import indicators, forces, integrators, experiments, sim, serialize, potentials, noise
import numpy as np
import datetime as dt
import math, random, time
from multiprocessing import Pool
from tqdm import tqdm

np.seterr(all="ignore")

SIM_COUNT = 100

decay = input("Decay [y]/[N] >")
if decay == 'y':
    path = '/two_particle/earth-moon-decay'
    decay_force = [lambda w,c: forces.viscous_damping_force(world=w, c=float(10**15), context=c)]
else:
    path = '/two_particle/earth-moon'
    decay_force = []

base_params = {
    'timestep': 2.4*10**6 / 10000, # 1/1000 of orbital period
    'n_timesteps': 10000, # 1 period
    'separation': 384.4*10**6, # earth-moon distance
    'small_m': 7.348*10**22, # moon mass
    'large_m': 5.972*10**24, # earth mass,
    'tangent_speed': 1.022*10**3, # moon orbital velocity,
    'std': 1*10**6,
    'mean': 1*10**7
}

true_params = {
    'G':(6.674*(10**(-11))),
}

def run_sim_and_write(seed):
    true_params_aug = {**base_params, **true_params}
    world = sim.run_two_body_sim(
        true_params_aug,
        seed,
        forces=[
            lambda world, context: forces.newtons_law_of_gravitation(world, true_params['G'], context)
        ]+decay_force,
        indicators=[
            lambda w: indicators.hamiltonian(w, global_potentials=[lambda w: potentials.gravitational_potential_energy(w, G=true_params['G'])]),
            lambda w: potentials.gravitational_potential_energy(w, G=true_params_aug['G'])
            ],
        indicator_schema=['Hamiltonian', 'GPE'],
     )
    serialize.save_world(world, './data'+path, true_params_aug, seed)

if __name__ == '__main__':
    for i in tqdm(range(SIM_COUNT)):
        run_sim_and_write(time.time())