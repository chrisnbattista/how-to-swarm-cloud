from particle_sim import *

world = experiments.set_up_experiment(2, 10)

for i in range(10):
    world = experiments.advance_timestep(
        world,
        1,
        integrators.integrate_rect_world,
        [potentials.pairwise_world_lennard_jones_potential]
    )