from particle_sim import *

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

plt.ion()
plt.show()
sns.set_theme()

def run_sim():

    ## ICs
    world = experiments.set_up_experiment(radius=5, n_particles=15)
    long_world_history = world

    ## Sim loop
    for i in range(100000):

        # Data viz
        if i % 10 == 0:
            plt.clf()
            p = sns.scatterplot(
                x='b_1',
                y='b_2',
                s=5,
                hue='t',
                data=long_world_history
            )
            p.legend_.remove()
            p2 = sns.scatterplot(
                x='b_1',
                y='b_2',
                color='k',
                data=world
            )
            p2.legend_.remove()
            plt.title(f"LJ Sim Timestep {i}")
            plt.pause(0.001)
        
        # Trajectory recording
        long_world_history = pd.concat([long_world_history, world], axis=0, ignore_index=True)

        # Sim step
        world = experiments.advance_timestep(
            world,
            0.1,
            integrators.integrate_rect_world,
            [lambda x: potentials.pairwise_world_lennard_jones_potential(x, 5, 1)]
        )
        
        # BCs (periodic square)
        world['b_1'] = world['b_1'] % 10
        world['b_2'] = world['b_2'] % 10

if __name__ == '__main__':
    run_sim()