# Demonstrate graphically the Lennard-Jones potential simulation scenario





from hts.multi_agent_kinetics import indicators, forces, integrators, experiments

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import datetime as dt
import time, math

def setup_plots():
    plt.ion()
    plt.show()
    sns.set_theme()
    return plt.subplots(2,1,
                        gridspec_kw={'height_ratios': [4, 1]},
                        figsize=(7.5, 9)
                        )

def lj_desktop_data_viz(world, long_world_history, long_indicator_history, i, indicator_labels=[], label='', note='', fig=None, ax=None):
    '''
    '''
    if i % 100 == 0:

        ax[0].clear()
        p = sns.scatterplot(
            x=world[:,1],
            y=world[:,2],
            color='k',
            ax=ax[0]
        )

        if i % 5000 == 0:
            n_indicators = len(indicator_labels)
            ax[1].clear()
            for ind in range(1, n_indicators+1):
                sns.lineplot(
                    x=long_indicator_history[:i, 0],
                    y=long_indicator_history[:i, ind],
                    ax=ax[1],
                    legend=False
            )

            plt.legend(loc='lower right', labels=indicator_labels)

        ##plt.title(label)
        fig.canvas.set_window_title(label)

        if not len(fig.texts):
            fig.text(0.01, 0.01, note)
        else:
            fig.texts[0].set_text(note)

        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.01)

def run_sim(data_viz=lj_desktop_data_viz, fig=None, ax=None):

    ## Record keeping
    start_time = time.time()
    last_loop_time = start_time

    ## Parameters
    timestep = 0.01
    size = 800
    n_particles = 20
    n_steps = 1000000
    min_dist = 20

    epsilon = 1
    sigma = 25
    c = 0.01
    lamb = 0.01

    indicator_functions = [
        lambda world: indicators.hamiltonian(world, [lambda world: forces.sum_world_lennard_jones_potential(world, epsilon, sigma), lambda world: forces.sum_world_gravity_potential(world, lamb)]),
        lambda world: indicators.kinetic_energy(world),
        lambda world: indicators.potential_energy(world, [lambda world: forces.sum_world_lennard_jones_potential(world, epsilon, sigma), lambda world: forces.sum_world_gravity_potential(world, lamb)]),
    ]

    n_indicators = len(indicator_functions)

    ## ICs
    world = experiments.set_up_experiment(
        radius=size/2,
        center=(size/2, size/2),
        n_particles=n_particles,
        min_dist=min_dist,
        random_speed=50)

    long_world_history = np.empty( (n_steps * n_particles, 7) )
    long_indicator_history = np.empty( (n_steps, n_indicators + 1) )

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
                    lambda world: forces.pairwise_world_lennard_jones_force(world, epsilon=epsilon, sigma=sigma),
                    lambda world: forces.viscous_damping_force(world, c),
                    lambda world: forces.gravity_well(world, lamb)
                ],
                indicator_functions
            )

            ## Indicator recording
            # Index
            long_indicator_history[ i, 0 ] = i
            # Indicators
            long_indicator_history[ i, 1: ] = indicator_results

            ## Data viz
            data_viz(world,
                    long_world_history,
                    long_indicator_history,
                    i,
                    indicator_labels=["Hamiltonian", "Kinetic Energy", "Potential Energy"],
                    label='Lennard Jones Particle Dynamics Simulation',
                    note=f'Timestep: {i} | Wall time per timestep: {loop_duration:.5f}',
                    fig=fig,
                    ax=ax
            )

            ## BCs (periodic square)
            ##world['b_1'] = world['b_1'] % size
            ##world['b_2'] = world['b_2'] % size

    except KeyboardInterrupt:
        pass

    # Save data
    print("\nSaving data...")
    np.savetxt("./data/LJ Sim Run_" + str(dt.datetime.now()) + ".csv", long_world_history, delimiter=",")
    print("Saved")

if __name__ == '__main__':
    fig, ax = setup_plots()
    run_sim(fig=fig, ax=ax)
