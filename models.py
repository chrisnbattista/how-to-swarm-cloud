import torch
from particle_sim import experiments




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

    def forward(self, last_state):
        '''
        '''

        predicted_next_state, indicators = \
            experiments.advance_timestep(last_state, **self.params)
        
        return torch.from_numpy(predicted_next_state)
    
    def backward(self):
        '''
        '''

        return 1