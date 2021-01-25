





import numpy as np
import datetime as dt
import math, random, os

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from hts.multi_agent_kinetics import indicators, forces, integrators, experiments, sim, viz
import models, data_loaders
import tb_logging





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

choice = 'single step'
## Get data
if input("Enter something if you want to use single step cost function >"):
    data = data_loaders.SimStateToOneAgentStepSamples("./data/sim_data/" + random.choice(os.listdir('./data/sim_data')))
else:
    choice = 'forward run'
    data = data_loaders.SimICsToFullAgentTrajectorySamples("./data/sim_data/" + random.choice(os.listdir('./data/sim_data')))


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
if choice == 'single step':
    model = models.PhysicsSingleStepModel(**world_params)
else:
    model = models.PhysicsForwardRunModel(**world_params)


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
tb_logging.writer = SummaryWriter()

if not input("Please enter any input to skip learning >"):

    for i in range(int(len(data))): # step through iterations
        
        tb_logging.epoch = i # synchronize epoch across functions

        # define x and y from state pairs
        x, y = data[i]
        x.requires_grad = True

        # Get model predictions
        y_pred = model(i % data.n_agents, x)

        # Compute loss
        loss = criterion(y_pred, y)

        # Log loss and current parameters
        tb_logging.writer.add_scalar("Loss/train", loss, i)
        tb_logging.writer.add_scalar("Parameter/sigma", model.sigma.data, i)
        tb_logging.writer.add_scalar("Parameter/epsilon", model.epsilon.data, i)

        if i % int(len(data)/10) == 0:
            print("#", flush=True, end='')

        #print(f"Loss: {loss.data}")
        #print(f"Sigma: {model.sigma.data}")
        #print(f"Epsilon: {model.epsilon.data}")

        ##torchviz.make_dot(loss).render("graph", format="png")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tb_logging.writer.flush()
    tb_logging.writer.close()

else:
    viz.generate_cost_plot(model, data, criterion,
                                    {
                                        'sigma':range(-50, 50, 5),
                                        'epsilon':range(-10, 10, 2)
                                    },
                                    range(0,len(data), 1)
    )