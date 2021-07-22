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
from multi_agent_kinetics import serialize, worlds, forces, potentials, indicators

G_seed_guess = sys.argv[1]

torch.autograd.set_detect_anomaly(True)

## Split data
filepaths = serialize.list_worlds("./data/two_particle/scaled-gravity")
print(f'Number of training files: {len(filepaths)}')
train_filepaths, test_filepaths = train_test_split(filepaths, test_size=0.2)

## Define model
class GravityNet(nn.Module):
    def __init__(self, G_guess):
        super().__init__()
        self.G = torch.nn.Parameter(torch.tensor(G_guess))
        self.delta = 0.01
        self.l1 = nn.L1Loss()
    
    def run_sim(self, X, steps, G):
        ##input(f'initial state: {X[:,:7]}')
        temp_world = worlds.World(
            initial_state=X[:,:7],
            forces=[
                lambda world, context: forces.newtons_law_of_gravitation(world, G, context)
            ],
            indicators=[
                lambda w: indicators.hamiltonian(w, global_potentials=[lambda w: potentials.gravitational_potential_energy(w, G=G)]),
                lambda w: potentials.gravitational_potential_energy(w, G=G)
                ],
            indicator_schema=['Hamiltonian', 'GPE'],
            n_timesteps=steps,
            timestep=0.02
        )
        temp_world.advance_state(steps-1)
        return temp_world.get_full_history_with_indicators()
    
    def forward(self, X, steps):
        self.last_X = X
        self.last_steps = steps
        return self.run_sim(X, steps, self.G)

    def compute_grad(self, actual):
        ## Prepare indexes
        pos = worlds.pos[2]
        ham = slice(len(worlds.schemas['2d']), len(worlds.schemas['2d'])+1)

        ## Scaling factor for Hamiltonian regularization
        ham_factor = 0.0001

        ## TODO: note in paper that velocity must be measurable for this technique to work (part of state vector) - difficult in practice
        ## Compute loss for slightly perturbed G values
        plus_G_history = self.run_sim(self.last_X, self.last_steps, self.G+self.delta)
        # TODO add vel term
        # be explicit about state vector
        # TODO: don't reference true Hamiltonian after IC
        plus_G_loss = self.l1(actual[pos], plus_G_history[pos]) + self.l1(actual[ham], plus_G_history[ham]) * ham_factor
        minus_G_history = self.run_sim(self.last_X, self.last_steps, self.G-self.delta)
        minus_G_loss = self.l1(actual[pos], minus_G_history[pos]) + self.l1(actual[ham], plus_G_history[ham]) * ham_factor

        ## Calculate the gradient numerically based on the above values of G
        self.G.grad = Variable(
            torch.tensor(
                (plus_G_loss - minus_G_loss) / (self.delta*2),
                requires_grad=False
            ).type(torch.FloatTensor)
        )

model = GravityNet(float(G_seed_guess))

## Define optimizer
hyperparams = {
    'lr': 1e-2,
    'p':0.1
}
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=hyperparams['lr'],
    momentum=hyperparams['p'])
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

## Define indexes (2D case)
pos = worlds.pos[2]
vel = worlds.vel[2]
ham = slice(len(worlds.schemas['2d']), len(worlds.schemas['2d'])+1)

## Prepare to record training graphs
losses_over_time = {
    'pos':[],
    'vel':[],
    'ham':[],
    'guess':[]
}

try:

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

        ## Define input and expected output
        X = history[0:world.n_agents,:]
        y = history

        ## Run sim
        y_pred = model(X, steps)

        ##input(f"frame 0 of y_pred: {y_pred[0,:]}")

        ## Compute loss
        pos_reconstruction_loss = pos_reconstruction_loss_fn(y_pred[:,pos], y[:,pos])
        vel_reconstruction_loss = vel_reconstruction_loss_fn(y_pred[:,vel], y[:,vel])
        hamiltonian_loss = hamiltonian_loss_fn(y_pred[0,ham], history[0,ham]) # always compare against initial hamiltonian
        ##d_ham_dt_loss += (prediction[0,ham] - last_ham)/world.timestep_length

        ## Compute G gradient
        model.zero_grad()
        model.compute_grad(history)

        ## Adjust parameters: learning step
        optimizer.step()

        ## Print and record progress
        print(f'pos L: {pos_reconstruction_loss}\tvel L: {vel_reconstruction_loss}')
        print(f'ham L: {hamiltonian_loss}')
        print(f'G guess: {model.G}')
        losses_over_time['pos'].append(pos_reconstruction_loss.detach().numpy())
        losses_over_time['vel'].append(vel_reconstruction_loss.detach().numpy())
        losses_over_time['ham'].append(hamiltonian_loss.detach().numpy())
        losses_over_time['guess'].append(float(model.G.detach().numpy()))

    ## Plot results
    plt.plot(losses_over_time['pos'])
    plt.plot(losses_over_time['vel'])
    plt.plot(losses_over_time['ham'])
    plt.show()

except:
    pass

finally:
    print("Logging run...")
    pandas.DataFrame(losses_over_time).to_csv('./data/results/losses'+str(datetime.datetime.now())+".csv")