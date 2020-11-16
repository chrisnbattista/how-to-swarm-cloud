from particle_sim import *

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import datetime as dt
import time, math

def setup_plots():
    plt.ion()
    plt.show()
    sns.set_theme()

def lj_desktop_data_viz(world, long_world_history, i):
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
        ##plt.pause(0.00001)
        fig = plt.gcf()
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.0001)

def run_sim(data_viz=lj_desktop_data_viz):

    ## Record keeping
    start_time = time.time()
    last_loop_time = start_time

    ## Parameters
    timestep = 0.01
    size = 100
    n_particles = 100

    epsilon = 1
    omega = 1

    ## ICs
    world = experiments.set_up_experiment(
        radius=size/2,
        center=(size/2, size/2),
        n_particles=n_particles)
    long_world_history = world.reset_index()

    ## Sim loop
    try:
        for i in range(10000):

            ## Record keeping
            loop_duration = time.time() - last_loop_time
            last_loop_time += loop_duration
            print(loop_duration)

            ## Data viz
            data_viz(world, long_world_history, i)
            
            ## Trajectory recording
            long_world_history = pd.concat([long_world_history, world.reset_index()], axis=0, ignore_index=True)

            ## Sim step
            world = experiments.advance_timestep(
                world,
                timestep,
                integrators.integrate_rect_world,
                [
                    lambda x: forces.pairwise_world_lennard_jones_potential(x, epsilon=epsilon, omega=omega),
                    lambda x: forces.viscous_damping_force(x, 0.005)
                ]
            )
            
            ## BCs (periodic square)
            ##world['b_1'] = world['b_1'] % size
            ##world['b_2'] = world['b_2'] % size

    except KeyboardInterrupt:
        pass

    # Save data
    long_world_history.to_csv("./data/LJ Sim Run_" + str(dt.datetime.now()) + ".csv")

if __name__ == '__main__':
    setup_plots()
    run_sim()