import random, datetime, sys, itertools
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

# TODO: fix paper
# maggioni paper or more generic in methodology
# methodology needs to be more independent of test case. Show dynamical system equation - make it match conventional multiple agent
# autonomous system - x_dot, j_dot equation - generic way that shows fundamental assumptions
# don't bring in gravity until results
# point out immediately that it is autonomous system - no control input
# not time dependent interaction laws - phi
# think about answering readers questions as they pop up
# do we have nonlinearity? what kind of relationships do the variables bear to each other

# TODO: prepare for reviewer questions
# complications / more complex / less ideal systems
# use it on something that's not a textbook example
# noise, disturbances, data error, (not fully characterized systems?)

# TODO: primary assumption of form - address violations of assumption
# equation form known: what if we give it erroneous equation form?
# add random noise, characterize data as superposition of analytical system and random processes, modeled as noise
# the noise should be added to both velocity and position, but post facto (measurement error). we are assuming a pure dynamical system (no unmodeled dynamics)
# model how noise degrades algo performance (seek maximum threshold)
# physics are very simplified, note this for reviewers
# talking about noise, flagging may avoid more reviews

# TODO: tests
# try with real-world data. Try sun-moon, earth-moon. NASA JPL Horizon data sets.
# IVs: type of system, amount of noise, optimizer, number of bodies, use of Hamiltonian (perhaps quantify by adjusting weighting)
# DVs: time to convergence, steady state error

# note: if we wanted to do a partially-stochastic system, we would want to regress the noise parameters. maybe future note. assumption of gaussian distribution might help solve

# TODO: Don't normalize to unknowns.

## TODO: pick specific case for diagram. Show how process works. Reduce time to understand paper.

r_seed_guess = float(1)
G_seed_guess = float(sys.argv[1])
if len(sys.argv) > 2:
    optimizer_choice = sys.argv[2]
else:
    optimizer_choice = 'sgd'

torch.autograd.set_detect_anomaly(True)

## Split data
filepaths = serialize.list_worlds("./data/two_particle/scaled-gravity")
print(f'Number of training files: {len(filepaths)}')
train_filepaths, test_filepaths = train_test_split(filepaths, test_size=0.2)

## NOTE: highly dependent on initial guess for learning rate

def percentage_error(actual, predicted):
    '''Computes the classic error function (actual - predicted) / actual for each row and returns the average of the result.'''
    if torch.numel(actual) == 1:
        return torch.nan_to_num(
            torch.abs((actual - predicted) / actual),
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
    else:
        return \
            torch.mean(
                torch.linalg.norm(
                    torch.nan_to_num(
                        torch.div(actual - predicted, actual),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0
                    ),
                    dim=1
                )
            )

## Define model
class GravityNet(nn.Module):
    def __init__(self, G_guess, r_guess, hyperparams):
        super().__init__()
        self.G = torch.nn.Parameter(torch.tensor(G_guess))
        self.r = torch.nn.Parameter(torch.tensor(r_guess))
        self.hyperparams = hyperparams
        self.delta_G = 0.01 # TODO: decay this value over time
        self.delta_r = 0.1
        self.first_X = None
    
    def run_sim(self, X, steps, G, r):
        '''utility func'''
        ##input(f'initial state: {X[:,:7]}')
        temp_world = worlds.World(
            initial_state=X[:,:7],
            forces=[
                lambda world, context: forces.newtons_law_of_gravitation(world=world, G=G, r=r.detach().clone(), context=context)
            ],
            indicators=[
                lambda w: indicators.hamiltonian(w, global_potentials=[lambda w: potentials.gravitational_potential_energy(world=w, G=G)]),
                lambda w: potentials.gravitational_potential_energy(world=w, G=G)
                ],
            indicator_schema=['Hamiltonian', 'GPE'],
            n_timesteps=steps,
            timestep=0.02
        )
        temp_world.advance_state(steps-1)
        return temp_world.get_full_history_with_indicators()
    
    def forward(self, X, steps):
        if self.first_X is None: self.first_X = X
        self.last_X = X
        self.last_steps = steps
        return self.run_sim(X=X, steps=steps, G=self.G, r=self.r)
    
    def compute_loss(self, actual, predicted, return_components=False):
        '''Computes loss weighted according to hyperparameters.'''
        baseline = torch.tile(
            self.first_X[0,:], ## TODO: Change this to use *GLOBAL* IC instead of segment IC (information leak to the learning algorithm)
            (actual.shape[0], 1)
        ) # TODO: Normalize from common point so losses are directly comparable across samples
        baselined_actual = torch.div(actual, baseline)
        baselined_predicted = torch.div(predicted, baseline)
        p_err = self.hyperparams['p_err_weight'] * percentage_error(baselined_actual[pos], baselined_predicted[pos])
        v_err = self.hyperparams['v_err_weight'] * percentage_error(baselined_actual[vel], baselined_predicted[vel])
        h_err = self.hyperparams['h_err_weight'] * percentage_error(baselined_actual[ham], baselined_predicted[ham])
        sum = p_err + v_err + h_err
        if return_components:
            return (
                sum,
                {
                    'p_err':p_err,
                    'v_err':v_err,
                    'h_err':h_err
                }
            )
        else:
            return sum

    def compute_grad(self, actual, delta=None):
        if not delta is None:
            d_r = delta
            d_G = delta
        else:
            d_r = self.delta_r
            d_G = self.delta_G

        ## Prepare indexes
        pos = worlds.pos[2]
        ham = slice(len(worlds.schemas['2d']), len(worlds.schemas['2d'])+1)

        ## Central Difference Theorem

        ## Compute loss for slightly perturbed G values
        plus_G_history = self.run_sim(X=self.last_X, steps=self.last_steps, G=self.G+d_G, r=self.r)
        plus_G_loss = self.compute_loss(actual, plus_G_history)
        minus_G_history = self.run_sim(X=self.last_X, steps=self.last_steps, G=self.G-d_G, r=self.r)
        minus_G_loss = self.compute_loss(actual, minus_G_history)

        ## Compute loss for slightly perturbed r values
        plus_r_history = self.run_sim(X=self.last_X, steps=self.last_steps, G=self.G, r=self.r+d_r)
        plus_r_loss = self.compute_loss(actual, plus_r_history)
        minus_r_history = self.run_sim(X=self.last_X, steps=self.last_steps, G=self.G, r=self.r-d_r)
        minus_r_loss = self.compute_loss(actual, minus_r_history)

        ## Calculate the gradient numerically based on the above values of G
        self.G.grad = Variable(
            ((plus_G_loss - minus_G_loss).clone().detach() / (d_G*2)).type(torch.FloatTensor)
        )
        self.r.grad = Variable(
            ((plus_r_loss - minus_r_loss).clone().detach() / (d_r*2)).type(torch.FloatTensor)
        )

## Define optimizer and model
hyperparams = {
    'lr': 1e-2,
    'p':0.1,
    'segment_count': 2000,   # try lengthening segment. figure out why it changes from quadratic to linear learning rate. show zoom in during paper.
    'accuracy_threshold': 0.005,
    'p_err_weight': 1.0,
    'v_err_weight': 0.5,
    'h_err_weight': 0.1,
    'betas':(0.9, 0.999)
}

## TODO: investigate root causes of spikes.
## Try choosing different segment
## Investigate whether simulated dynamical system analytical ICs causing transient behavior -> spike in error, overcome by algorithm
## Discoveries!

model = GravityNet(G_guess=G_seed_guess, r_guess=r_seed_guess, hyperparams=hyperparams)
if optimizer_choice == 'sgd':
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=hyperparams['lr'],
        momentum=hyperparams['p'])
elif optimizer_choice == 'adam':
    optimizer = torch.optim.Adam(
        (model.G, model.r),
        lr=hyperparams['lr'],
        betas=hyperparams['betas'],
        amsgrad=True
    )
else:
    raise Exception("Invalid choice of optimizer.")

## Define indexes (2D case)
pos = worlds.pos[2]
vel = worlds.vel[2]
ham = slice(len(worlds.schemas['2d']), len(worlds.schemas['2d'])+1)

## Prepare to record training graphs
losses_over_time = {
    'pos':[],
    'vel':[],
    'ham':[],
    'g':[],
    'r':[]
}

try:

    ## Train model with data
    fp = random.choice(filepaths)
    # we train with segments all taken from same trajectory. We don't combine trajectories
    # since we will only be looking at one dynamical system at a time, we can only use data from one to train
    # different ICs may operate in different domains in state space
    print(f'Training on {fp}')

    ## Load data
    world, params = serialize.load_world(fp)
    history = world.get_full_history_with_indicators()

    ## Break the trajectory into distinct segments
    steps = int(history.shape[0] / world.n_agents / hyperparams['segment_count'])

    ## Try the training data multiple times if necessary
    halt = False
    for epoch in itertools.count(start=0, step=1):

        ## Safety condition to stop infinite loop. Epoch limit. Also necessary for regular halting condition
        if halt == True or epoch > 9: break

        # TODO: Fix IID conditions
        # TODO: Moving window to create segments
        # TODO: Increase window size to learn r more effectively

        ## Loop through all the segments of the trajectory
        for segment_i in range(hyperparams['segment_count']):
            segment_history = history[int(steps*segment_i*world.n_agents):int(steps*(segment_i+1)*world.n_agents),:]

            ## Define input and expected output
            X = segment_history[0:world.n_agents,:]
            y = segment_history

            ## Run sim
            y_pred = model(X, steps)

            ## Compute gradients
            model.zero_grad()
            model.compute_grad(segment_history)

            ## Adjust parameters: learning step
            optimizer.step()

            ## Print and record progress
            loss, l_comps = model.compute_loss(y, y_pred, return_components=True)
            print(f'### Epoch {epoch} | Step {segment_i} | Overall Loss: {loss}')
            print(f'pos L: {l_comps["p_err"]}\tvel L: {l_comps["v_err"]}')
            print(f'ham L: {l_comps["h_err"]}')
            print(f'G guess: {model.G}')
            print(f'r guess: {model.r}')
            losses_over_time['pos'].append(l_comps["p_err"].detach().numpy())
            losses_over_time['vel'].append(l_comps["v_err"].detach().numpy())
            losses_over_time['ham'].append(l_comps["h_err"].detach().numpy())
            losses_over_time['g'].append(float(model.G.detach().numpy()))
            losses_over_time['r'].append(float(model.r.detach().numpy()))

            ## Halting condition
            if model.compute_loss(y, y_pred) < hyperparams['accuracy_threshold']:
                print(f"Accuracy threshold reached in iteration {segment_i} of epoch {epoch} - halting.")
                halt = True
                break

except KeyboardInterrupt as e:
    pass

finally:
    print("Logging run...")
    pandas.DataFrame(losses_over_time).to_csv('./data/results/losses'+str(datetime.datetime.now())+"_segment_train.csv")