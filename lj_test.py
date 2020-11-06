from particle_sim import *

import seaborn as sns
import matplotlib.pyplot as plt

plt.ion()
plt.show()

world = experiments.set_up_experiment(10, 10)

for i in range(1000):
    if i % 10 == 0:
        ##plt.clf()
        sns.scatterplot(
            x=world['b_1'],
            y=world['b_2']
        )
        plt.title(f"LJ Sim Timestep {i}")
        plt.pause(0.001)
    world = experiments.advance_timestep(
        world,
        0.1,
        integrators.integrate_rect_world,
        [lambda x: potentials.pairwise_world_lennard_jones_potential(x, 10, 1)]
    )