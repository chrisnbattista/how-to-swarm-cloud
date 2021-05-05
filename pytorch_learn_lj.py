





import numpy as np
import datetime as dt
import math, random, os, glob

from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from torchviz import make_dot
import matplotlib.pyplot as plt

from multi_agent_kinetics import indicators, forces, integrators, experiments, sim, viz, serialize
from hts.learning import models, data_loaders, tb_logging

torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float)

## Get data
path = random.choice(
    serialize.list_worlds(
        random.choice(
            glob.glob('./data/two_particle_pt/*/')
        )
    )
)
print(f"Choosing world {path}")

choice = 'single step'

if input("Enter something if you want to use single step learning >"):
    data = data_loaders.SimStateToOneAgentStepSamples(path)
elif input("Enter something if you want to use full trajectory learning >"):
    choice = 'forward run'
    data = data_loaders.SimICsToFullAgentTrajectorySamples(path)
else:
    choice = 'neural net'
    data = data_loaders.SimStateToSimState(path)

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
random.seed()
learned_params = {
    'epsilon': float(random.randint(10,40)),
    'sigma': float(random.randint(0,5))
}
print(f"\ninitial guesses\nepsilon: {learned_params['epsilon']}\nsigma: {learned_params['sigma']}")
world_params = {**world_params, **learned_params}

# Define learning hyperparameters
hyperparams = {
    'learning_rate': 0.1,
    'momentum': 0.9
}

# Create model object
if choice == 'single step':
    model = models.PhysicsSingleStepModel(**world_params)
elif choice == 'forward run':
    model = models.PhysicsForwardRunModel(**world_params)
else:
    model = model = torch.nn.Sequential(
                        torch.nn.Linear(12, 12),
                        torch.nn.ReLU(),
                        torch.nn.Linear(12, 6),
            ).float()


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

    # Gradient Descent routine to learn params

    for i in tqdm(range(int(len(data)))): # step through iterations
        
        tb_logging.epoch = i # synchronize epoch across functions

        # define x and y from state pairs
        x, y = data[i]
        y = y.float()
        ##x[1].requires_grad = True

        print(x)

        # Get model predictions
        if choice == 'neural net':
            y_pred = model(x.float())
        else:
            y_pred = model(*x)

        # Compute loss
        print(y_pred)
        print(y)
        loss = criterion(y_pred, y)
        
        # Plot loss computation graph
        if i ==0: make_dot(loss).render("loss_graph")

        # Log loss and current parameters
        tb_logging.writer.add_scalar("Loss/train", loss, i)
        tb_logging.writer.add_scalar("Parameter/sigma", model.sigma.data, i)
        tb_logging.writer.add_scalar("Parameter/epsilon", model.epsilon.data, i)

        #print(f"Loss: {loss.data}")
        #print(f"Sigma: {model.sigma.data}")
        #print(f"Epsilon: {model.epsilon.data}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tb_logging.writer.flush()
    tb_logging.writer.close()
    print(f"\nFinal values:\nSigma: {model.sigma.data}\nEpsilon: {model.epsilon.data}")
    input()

else:

    # Visualization routine to show cost landscape

    viz.generate_cost_plot(model, data, criterion,
                                    {
                                        'sigma':np.linspace(0, 2, 11),
                                        'epsilon':np.linspace(20, 30, 11)
                                    },
                                    range(0,len(data), 1)
    )