import torch
import torch.nn as nn
from multi_agent_kinetics import experiments, forces
from hts.learning import functions
import torch.nn.functional as F

class PhiModel(nn.Module):
    def __init__(self):
        super(PhiModel, self).__init__()

        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class KernelLearner(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(KernelLearner, self).__init__()

        ##self.hidden_size = hidden_size
        '''
        self.i2bl = nn.Linear(input_size + hidden_size, big_layers_size)
        self.big_layers = nn.Linear(big_layers_size, big_layers_size)
        self.bl2o = nn.Linear(big_layers_size, output_size)
        self.bl2h = nn.Linear(big_layers_size, hidden_size)
        self.softmax = nn.LogSoftmax()
        '''
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x
    '''
    def forward(self, input_vector, hidden):
        
        combined = torch.cat((input_vector, hidden))
        c2bl = self.i2bl(combined)
        bl = self.big_layers(c2bl)
        ##hidden = self.bl2h(bl)
        output = self.bl2o(bl)
        output = self.softmax(output)
        return output, hidden
        '''

    
    def init_hidden(self):
        return torch.zeros(self.hidden_size, dtype=torch.float)



class PhysicsSingleStepModel(nn.Module):
    '''
    '''

    def __init__(self, epsilon=2, sigma=25, **kwparams):
        '''
        '''

        super(PhysicsSingleStepModel, self).__init__()

        ## Record world parameters
        self.params = kwparams

        ## Define learnable parameters
        self.epsilon = torch.nn.Parameter(torch.Tensor((float(epsilon),)))
        self.sigma = torch.nn.Parameter(torch.Tensor((float(sigma),)))

        ## Define functions to apply during forward propagation
        self.physics_func = functions.PhysicsStep

    def forward(self, agent, last_state):
        '''
        '''

        last_state = self.physics_func.apply(
            agent,
            last_state,
            self.params,
            self.sigma,
            self.epsilon
        )
        
        return last_state.float()

class PhysicsForwardRunModel(nn.Module):
    '''
    '''

    def __init__(self, epsilon=0, sigma=0, **kwparams):
        '''
        '''

        super(PhysicsForwardRunModel, self).__init__()

        ## Record world parameters
        self.params = kwparams

        ## Define learnable parameters
        self.epsilon = torch.nn.Parameter(torch.Tensor((float(epsilon),)))
        self.sigma = torch.nn.Parameter(torch.Tensor((float(sigma),)))

        ## Define functions to apply during forward propagation
        self.physics_func = functions.PhysicsForwardRun

    def forward(self, agent, initial_state):
        '''
        '''

        trajectories = self.physics_func.apply(
            agent,
            initial_state,
            self.params,
            self.sigma,
            self.epsilon
        )
        
        return torch.Tensor(trajectories).float()

class NeuralNetStateTransitionModel (nn.Module):
    '''
    '''

    def __init__(self, **kwparams):
        '''
        '''

        super(NeuralNetStateTransitionModel, self).__init__()


        self.input_layer = torch.nn.Linear(2 * 4 * 3, 2 * 4 * 3) # input: 2 dims * (2 vel + 2 pos) * 3 agents
        self.fc1 = torch.nn.Linear(2 * 4 * 3, 2 * 4 * 3) # fully connected layer, same size
        self.output_layer = torch.nn.Linear(2 * 4 * 3, 2 * 2 * 3) # output: 2 dims * 2 pos * 3 agents
    
    def forward(self, x):
        '''
        '''

        x = F.relu(self.input_layer)
        x = F.relu(self.fc1)
        x = self.output_layer
        return x