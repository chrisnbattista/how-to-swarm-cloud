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

        learned_function = lambda world: forces.pairwise_world_lennard_jones_force(world, **{ \
            'epsilon': self.epsilon.data,
            'sigma': self.sigma.data
        })

        args = {**self.params, **{'forces':self.params['forces'] + [learned_function]}}

        last_state = self.physics_func.apply(
            agent,
            last_state,
            args,
            self.sigma,
            self.epsilon
        )
        
        return last_state