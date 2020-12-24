





import torch
from particle_sim import experiments






class PhysicsStep (torch.autograd.Function):
    '''
    '''

    @staticmethod
    def forward(ctx, agent, last_state, params={}, sigma=0, epsilon=0):
        '''
        '''

        ##ctx.save_for_backward(agent, last_state, params)

        print(params)

        state, indicators = experiments.advance_timestep(last_state.detach().numpy(), **params)

        state = torch.tensor(state[agent][1:3], requires_grad=True)

        return state
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
        '''

        print(grad_output)

        return None, None, None, grad_output[0], grad_output[1]