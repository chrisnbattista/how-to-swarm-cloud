import random
import torch
import torch.nn as nn
from torchviz import make_dot
import numpy as np
from sklearn.model_selection import train_test_split
from multi_agent_kinetics import serialize, worlds, forces, potentials, indicators

np.seterr(all="ignore")

## Split data
filepaths = serialize.list_worlds("./data/two_particle/earth-moon")
print(f'Number of training files: {len(filepaths)}')
train_filepaths, test_filepaths = train_test_split(filepaths, test_size=0.2)

## Define model
class GravityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.G = torch.nn.Parameter(torch.tensor(0.0000000001), requires_grad=False)
    
    def forward(self, input):
        temp_world = worlds.World(
            initial_state=input[:,:7],
            forces=[
                lambda world, context: forces.newtons_law_of_gravitation(world, self.G.numpy(), context)
            ],
            indicators=[
                lambda w: indicators.hamiltonian(w, global_potentials=[lambda w: potentials.gravitational_potential_energy(w, G=self.G.numpy())]),
                lambda w: potentials.gravitational_potential_energy(w, G=self.G.numpy())
                ],
            indicator_schema=['Hamiltonian', 'GPE'],
            n_timesteps=2
        )
        temp_world.advance_state()
        return torch.tensor(temp_world.get_state_with_indicators(), requires_grad=True)
    
    def backward(self):
        print("hiiiii")

model = GravityNet()

## Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)

## Define indexes (2D case)
pos = worlds.pos[2]
vel = worlds.vel[2]
ham = slice(len(worlds.schemas['2d']), len(worlds.schemas['2d'])+1)

## Train model with data
for fp in train_filepaths:
    print(f'Training on {fp}')

    ## Load data
    world, params = serialize.load_world(fp)
    history = world.get_full_history_with_indicators()

    ## Define loss functions and initialize loss values
    loss_fn = nn.L1Loss()
    pos_reconstruction_loss = vel_reconstruction_loss = hamiltonian_loss = d_ham_dt_loss = last_ham = 0
    all_hams = np.zeros((int(history.shape[0]/world.n_agents),))

    for pair in range(int(history.shape[0]/world.n_agents-1)):
        prediction = model(history[pair*world.n_agents:(pair+1)*(world.n_agents),:])
        actual = torch.tensor(history[(pair+1)*(world.n_agents):(pair+2)*(world.n_agents),:])
        ##print(f'predicted pos: {prediction[:,pos]}\nactual pos: {actual[:,pos]}')
        pos_reconstruction_loss += loss_fn(prediction[:,pos], actual[:,pos])
        vel_reconstruction_loss += loss_fn(prediction[:,vel], actual[:,vel])
        hamiltonian_loss += loss_fn(prediction[0,ham], actual[0,ham])
        d_ham_dt_loss += (prediction[0,ham] - last_ham)/world.timestep_length
        last_ham = prediction[0,ham]
        all_hams[pair] = last_ham
    
    ## Adjust parameters: learning step
    std_ham = np.std(all_hams)
    model.zero_grad()
    pos_reconstruction_loss.backward()
    optimizer.step()
    print(f'pos: {pos_reconstruction_loss}\tvel: {vel_reconstruction_loss}')
    print(f'ham: {hamiltonian_loss}\tdham/dt: {d_ham_dt_loss}\tStD Ham: {std_ham}')
    print(prediction.grad_fn)