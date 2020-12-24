import torch
from particle_sim import experiments, forces
import functions




class PhysicsModel(torch.nn.Module):
    '''
    '''

    def __init__(self, **kwparams):
        '''
        '''

        super(PhysicsModel, self).__init__()

        ## Record world parameters
        self.params = kwparams

        ## Define learnable parameters
        self.epsilon = torch.nn.Parameter(torch.randn(()))
        self.sigma = torch.nn.Parameter(torch.randn(()))

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

        print((self.epsilon.data, self.sigma.data))
        
        return last_state