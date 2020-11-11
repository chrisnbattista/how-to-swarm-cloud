from particle_sim import *

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import datetime as dt

plt.ion()
plt.show()
sns.set_theme()

def run_sim():

    ## Parameters
    timestep = 0.01
    size = 100

    ## ICs
    world = experiments.set_up_experiment(
        radius=10,
        center=(size/2, size/2),
        n_particles=25)
    long_world_history = world.reset_index()

    ## Sim loop
    try:
        for i in range(100000):

            # Data viz
            if i % 100 == 0:
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
            long_world_history = pd.concat([long_world_history, world.reset_index()], axis=0, ignore_index=True)

            # Sim step
            world = experiments.advance_timestep(
                world,
                timestep,
                integrators.integrate_rect_world,
                [
                    lambda x: forces.pairwise_world_lennard_jones_potential(x, 1, 1),
                    lambda x: forces.viscous_damping_force(x, 0.005)
                ]
            )
            
            # BCs (periodic square)
            world['b_1'] = world['b_1'] % size
            world['b_2'] = world['b_2'] % size

    except KeyboardInterrupt:
        pass

    # Save data
    long_world_history.to_csv("LJ Sim Run_" + str(dt.datetime.now()) + ".csv")

if __name__ == '__main__':
    run_sim()