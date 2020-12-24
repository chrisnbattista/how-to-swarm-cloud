





from particle_sim import indicators, forces, integrators, experiments, sim
import models, data_loaders
import numpy as np
import datetime as dt
import math, random, os
import torch, torchviz
from torch.utils.tensorboard import SummaryWriter





## Define simulation and learning parameters
base_params = {
    'timestep': 0.01,
    'size': 800, # rename to initialization_radius
    'n_particles': 20,
    'n_steps': 1000,
    'min_dist': 20, # rename for clarity
    'init_speed': 50,
    'c': 0.01,
    'lambda': 0.01
}

learned_params = {
    'epsilon': 2,
    'sigma': 25
}

# TODO: put in .ini file
true_params = {
    'epsilon': 2,
    'sigma': 25,
    'c': 0.01,
    'lamb': 0.01
}

true_params_aug = {**base_params, **true_params}

hyperparams = {
    'learning_rate': 0.1,
    'momentum': 0.9
}


## Get data
data = data_loaders.SimSamples("./sim_data/" + random.choice(os.listdir('./sim_data')))


## Initialize model

# Define world params
world_params = {
    'timestep': data.step_indices[1] - data.step_indices[0],
    'integrator': integrators.integrate_rect_world,
    'forces': [
        lambda world: forces.viscous_damping_force(world, **true_params),
        lambda world: forces.gravity_well(world, **true_params)
    ]
}

# Create model object to call sim
model = models.PhysicsModel(**world_params)


## Import loss function
criterion = torch.nn.MSELoss(reduction='sum')


## Initialize optimizer
# learning method:
# stocastic gradient descent
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=hyperparams['learning_rate'],
    momentum=hyperparams['momentum']
)


# Samples come in pairs, covering two timesteps (overlapping)
# For each sample:
# Input: (x): All agent positions at the timestep (count per timestep: n_agents * n_dims)
# Output (y): Each agent position at the next timestep (count per timestep: n_dims)

## Initialize TensorBoard visualization
writer = SummaryWriter()


for i in range(min(int(len(data)/10), 100)): # step through iterations
    # define x and y from state pairs
    x, y = data[i]
    x.requires_grad = True

    # Get model predictions
    y_pred = model(i % data.n_agents, x)

    # Compute loss
    loss = criterion(y_pred, y)
    writer.add_scalar("Loss/train", loss, i)
    print(loss.data)

    ##torchviz.make_dot(loss).render("graph", format="png")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

writer.flush()
writer.close()