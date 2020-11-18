from particle_sim import *

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import datetime as dt
import time, math

def setup_plots():
    plt.ion()
    plt.show()
    sns.set_theme()

def lj_desktop_data_viz(world, long_world_history, i, label='', note=''):
    if i % 100 == 0:
        plt.clf()
        # p = sns.scatterplot(
        #     x='b_1',
        #     y='b_2',
        #     s=5,
        #     hue='t',
        #     data=long_world_history
        # )
        p2 = sns.scatterplot(
            x=world[:,1],
            y=world[:,2],
            color='k'
        )
        plt.title(label)
        ##plt.pause(0.00001)
        fig = plt.gcf()
        fig.text(0.01, 0.01, note)
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.01)

def run_sim(data_viz=lj_desktop_data_viz):

    ## Record keeping
    start_time = time.time()
    last_loop_time = start_time

    ## Parameters
    timestep = 0.01
    size = 350
    n_particles = 100
    n_steps = 100000

    epsilon = 100
    omega = 3
    c = 0.03

    ## ICs
    world = experiments.set_up_experiment(
        radius=size/2,
        center=(size/2, size/2),
        n_particles=n_particles)
    long_world_history = np.empty( (n_steps * n_particles, 7) )

    ## Sim loop
    try:
        for i in range(n_steps):

            ## Record keeping
            loop_duration = time.time() - last_loop_time
            last_loop_time += loop_duration

            ## Trajectory recording
            long_world_history[ i*n_particles : (i+1)*n_particles, : ] = world

            ## Sim step
            world, indicator_results = experiments.advance_timestep(
                world,
                timestep,
                integrators.integrate_rect_world,
                [
                    lambda x: forces.pairwise_world_lennard_jones_force(x, epsilon=epsilon, omega=omega),
                    lambda x: forces.viscous_damping_force(x, c)
                ],
                {
                    'hamiltonian': indicators.hamiltonian
                }
            )

            ## Data viz
            data_viz(world,
                    long_world_history,
                    i,
                    label='Lennard Jones Particle Sim Timestep',
                    note=f'Hamiltonian: {indicator_results["hamiltonian"]:.1f} | Timestep: {i} | Wall time per timestep: {loop_duration:.5f}'
            )

            ## BCs (periodic square)
            ##world['b_1'] = world['b_1'] % size
            ##world['b_2'] = world['b_2'] % size

    except KeyboardInterrupt:
        pass

    # Save data
    np.savetxt("./data/LJ Sim Run_" + str(dt.datetime.now()) + ".csv", long_world_history, delimiter=",")

if __name__ == '__main__':
    setup_plots()
    run_sim()
