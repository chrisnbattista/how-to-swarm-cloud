from particle_sim import *

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

plt.ion()
plt.show()
sns.set_theme()

def run_sim():

    world = experiments.set_up_experiment(radius=5, n_particles=15)
    history = []
    long_world_history = world

    for i in range(1000):

        if i % 10 == 0:
            plt.clf()
            p = sns.scatterplot(
                x='b_1',
                y='b_2',
                hue='t',
                data=long_world_history
            )
            plt.title(f"LJ Sim Timestep {i}")
            p.legend_.remove()
            plt.pause(0.001)
        
        history.append(world)
        long_world_history = pd.concat([long_world_history, world], axis=0, ignore_index=True)

        world = experiments.advance_timestep(
            world,
            0.1,
            integrators.integrate_rect_world,
            [lambda x: potentials.pairwise_world_lennard_jones_potential(x, 25, 1)]
        )
        print(world.info())

if __name__ == '__main__':
    run_sim()