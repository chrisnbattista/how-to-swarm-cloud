from numpy.core.fromnumeric import trace
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from multi_agent_kinetics import viz, forces, worlds, serialize
import plot_training

def plot_trajectory_bundle(trace_world, sampling_width, sample_count, true_value):
    low = (true_value - sampling_width)
    high = (true_value + sampling_width)
    G_values = np.linspace(low, high, sample_count)
    steps = int(trace_world.get_history().shape[0] / trace_world.n_agents)

    for G in G_values:
        reconstructed_world = worlds.World(
            initial_state=trace_world.get_history()[0:trace_world.n_agents,:7],
            forces=[
                lambda world, context: forces.newtons_law_of_gravitation(world, G)
            ],
            n_timesteps=steps,
            timestep=0.02
        )
        reconstructed_world.advance_state(steps - 1)
        viz.trace_trajectories(
            reconstructed_world,
            fig,
            ax,
            'Trajectory Bundle Trace',
            trajectory_legend=['True Trace'] + \
            [
                f'Trace with G = {G_val}' for G_val in G_values
            ]
        )
    plt.show(block=True)

if __name__ == '__main__':
    ## TODO: crosslink training data with results data so we can plot the actual trajectories as the system learns
    ##df = pd.read_csv(sys.argv[1])
    ##plot_training.plot_training_graph(df)
    filepath = sys.argv[1]
    trace_world = serialize.load_world(filepath)[0]
    fig, ax = viz.set_up_figure()
    viz.trace_trajectories(trace_world, fig, ax, 'Trajectory Bundle Trace')

    sampling_width = float(sys.argv[2])
    sample_count = int(sys.argv[3])
    true_value = 1

    plot_trajectory_bundle(trace_world, sampling_width, sample_count, true_value)