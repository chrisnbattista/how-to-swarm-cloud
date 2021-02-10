import torch
from multi_agent_kinetics import experiments, forces
from hts.learning import functions




class PhysicsSingleStepModel(torch.nn.Module):
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
        
        return last_state

class PhysicsForwardRunModel(torch.nn.Module):
    '''
    '''

    def __init__(self, epsilon=2, sigma=25, **kwparams):
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

        last_state = self.physics_func.apply(
            agent,
            initial_state,
            self.params,
            self.sigma,
            self.epsilon
        )
        
        return last_state