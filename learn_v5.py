import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchviz import make_dot
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from multi_agent_kinetics import serialize, worlds, forces, potentials, indicators

torch.autograd.set_detect_anomaly(True)

## Split data
filepaths = serialize.list_worlds("./data/two_particle/earth-moon")
print(f'Number of training files: {len(filepaths)}')
train_filepaths, test_filepaths = train_test_split(filepaths, test_size=0.2)

## Define model
class GravityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.G = torch.nn.Parameter(torch.tensor(6.674*(10**(-11))))
        self.delta = 10**(-12)
        self.l1 = nn.L1Loss()
    
    def run_sim(self, input, steps, G):
        temp_world = worlds.World(
            initial_state=input[:,:7],
            forces=[
                lambda world, context: forces.newtons_law_of_gravitation(world, G, context)
            ],
            indicators=[
                lambda w: indicators.hamiltonian(w, global_potentials=[lambda w: potentials.gravitational_potential_energy(w, G=G)]),
                lambda w: potentials.gravitational_potential_energy(w, G=G)
                ],
            indicator_schema=['Hamiltonian', 'GPE'],
            n_timesteps=steps
        )
        temp_world.advance_state(steps-1)
        return temp_world.get_full_history_with_indicators()
    
    def forward(self, input, steps):
        self.last_input = input
        self.last_steps = steps
        return self.run_sim(input, steps, self.G)

    def compute_grad(self, actual):
        ## Prepare indexes
        pos = worlds.pos[2]
        ham = slice(len(worlds.schemas['2d']), len(worlds.schemas['2d'])+1)

        ## Scaling factor for Hamiltonian regularization
        ham_factor = 10e-18

        ## Compute loss for slightly perturbed G values
        plus_G_history = self.run_sim(self.last_input, self.last_steps, self.G+self.delta)
        plus_G_loss = self.l1(actual[pos], plus_G_history[pos]) + self.l1(actual[ham], plus_G_history[ham]) * ham_factor
        minus_G_history = self.run_sim(self.last_input, self.last_steps, self.G-self.delta)
        minus_G_loss = self.l1(actual[pos], minus_G_history[pos]) + self.l1(actual[ham], plus_G_history[ham]) * ham_factor

        ## Calculate the gradient numerically based on the above values of G
        self.G.grad = Variable(
            torch.clamp(
                torch.tensor(
                    (plus_G_loss - minus_G_loss) / (self.delta*2),
                    requires_grad=False
                ).type(torch.FloatTensor),
                min=-10**(-12),
                max=10**(-12)
            )
        )

model = GravityNet()

## Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

## Define indexes (2D case)
pos = worlds.pos[2]
vel = worlds.vel[2]
ham = slice(len(worlds.schemas['2d']), len(worlds.schemas['2d'])+1)

## Prepare to record training graphs
losses_over_time = {
    'pos':[],
    'vel':[],
    'ham':[]
}

## Train model with data
for fp in train_filepaths:
    print(f'Training on {fp}')

    ## Load data
    world, params = serialize.load_world(fp)
    history = world.get_full_history_with_indicators()

    ## Define loss functions and initialize loss values
    pos_reconstruction_loss_fn = nn.L1Loss()
    vel_reconstruction_loss_fn = nn.L1Loss()
    hamiltonian_loss_fn = nn.L1Loss()
    steps = int(history.shape[0] / world.n_agents)

    ## Run sim
    prediction = model(history[0:world.n_agents,:], steps)

    ## Compute loss
    pos_reconstruction_loss = pos_reconstruction_loss_fn(prediction[:,pos], history[:,pos])
    vel_reconstruction_loss = vel_reconstruction_loss_fn(prediction[:,vel], history[:,vel])
    hamiltonian_loss = hamiltonian_loss_fn(prediction[0,ham], history[0,ham])
    ##d_ham_dt_loss += (prediction[0,ham] - last_ham)/world.timestep_length

    ## Compute G gradient
    model.zero_grad()
    model.compute_grad(history)

    ## Adjust parameters: learning step
    optimizer.step()

    ## Print and record progress
    print(f'pos: {pos_reconstruction_loss}\tvel: {vel_reconstruction_loss}')
    print(f'ham: {hamiltonian_loss}')
    losses_over_time['pos'].append(pos_reconstruction_loss)
    losses_over_time['vel'].append(vel_reconstruction_loss)
    losses_over_time['ham'].append(hamiltonian_loss)

## Plot results
plt.plot(losses_over_time['pos'])
plt.plot(losses_over_time['vel'])
plt.plot(losses_over_time['ham'])
plt.show()