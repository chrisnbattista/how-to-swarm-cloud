import random, datetime, sys
import pandas
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchviz import make_dot
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from multi_agent_kinetics import serialize, worlds, forces, potentials, indicators, viz

## TODO: create bundle of color coded G trajectories

if len(list(sys.argv)) != 3:
    print("Usage: reconstruction_test.py filepath parameter_value")
    sys.exit(1)

fp = sys.argv[1]
G_guess = torch.tensor(float(sys.argv[2]), requires_grad=False)

true_world, true_params = serialize.load_world(fp)
true_history = true_world.get_full_history_with_indicators()
X = true_history[0:true_world.n_agents,:]

reconstructed_world = worlds.World(
        initial_state=X[:,:7],
        forces=[
            lambda world, context: forces.newtons_law_of_gravitation(world, G_guess, context)
        ],
        indicators=[
            lambda w: indicators.hamiltonian(w, global_potentials=[lambda w: potentials.gravitational_potential_energy(w, G=G_guess)]),
            lambda w: potentials.gravitational_potential_energy(w, G=G_guess)
            ],
        indicator_schema=['Hamiltonian', 'GPE'],
        n_timesteps=true_params['n_timesteps'],
        timestep=0.02
    )
reconstructed_world.advance_state(true_params['n_timesteps'] - 1)
reconstructed_history = reconstructed_world.get_full_history_with_indicators()

fig, ax = viz.set_up_figure()
viz.trace_trajectories(true_world, fig, ax, 'Trace')
viz.trace_trajectories(
    reconstructed_world,
    fig,
    ax,
    'Reconstruction Trace',
    indicator_legend=[
        'True Hamiltonian',
        'True GPE',
        'Reconstructed Hamiltonian',
        'Reconstructed GPE'
    ]
)
plt.show(block=True)