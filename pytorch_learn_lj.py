





import numpy as np
import datetime as dt
import math, random, os, glob

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from multi_agent_kinetics import indicators, forces, integrators, experiments, sim, viz, serialize
from hts.learning import models, data_loaders, tb_logging



## Get data
path = random.choice(
    serialize.list_worlds(
        random.choice(
            glob.glob('./data/*/')
        )
    )
)
print(f"Choosing world {path}")

choice = 'single step'

if input("Enter something if you want to use single step cost function >"):
    data = data_loaders.SimStateToOneAgentStepSamples(path)
else:
    choice = 'forward run'
    data = data_loaders.SimICsToFullAgentTrajectorySamples(path)

loaded_params = serialize.load_world(path)[1]



## Initialize model

# Define world params, excluding the force to be learned
# i.e., background dynamics
world_params = {**loaded_params, \
    **{
        'integrator': integrators.integrate_rect_world,
    } \
}

# Define initial guesses
learned_params = {
    'epsilon': 2,
    'sigma': 25
}

# Define learning hyperparameters
hyperparams = {
    'learning_rate': 0.1,
    'momentum': 0.9
}

# Create model object
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
                                        'sigma':np.linspace(0, 2, 30),
                                        'epsilon':np.linspace(20, 30, 30)
                                    },
                                    range(0,len(data), 1)
    )