





import torch
from particle_sim import experiments, forces






class PhysicsStep (torch.autograd.Function):
    '''
    '''

    @staticmethod
    def forward(ctx, agent, last_state, params={}, sigma=0, epsilon=0):
        '''
        '''

        ctx.agent = agent
        ctx.last_state = last_state
        ctx.params = params
        ctx.sigma = sigma
        ctx.epsilon = epsilon

        learned_function = lambda world: forces.pairwise_world_lennard_jones_force(world, **{ \
            'epsilon': epsilon.data,
            'sigma': sigma.data
        })

        args = {**params, **{'forces':params['forces'] + [learned_function]}}

        state, indicators = experiments.advance_timestep(last_state.detach().numpy(), **args)

        state = torch.tensor(state[agent][1:3], requires_grad=True)
        ctx.predicted_state = state

        return state
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
        '''

        to_diff = ('sigma', 'epsilon')
        diffs = [torch.Tensor([0, 0]) for _ in range(2)]

        delta = 1

        # loop through different parameters to take partials
        # utilizing Central Difference Theorem
        for i in range(len(to_diff)):
            # loop through the positive and negative perturbations (three-point finite difference)
            for d in (delta,):

                func_args = {
                    'epsilon': ctx.epsilon.data,
                    'sigma': ctx.sigma.data
                }
                func_args.update({to_diff[i]: func_args[to_diff[i]] + d}) # add perturbation to specified parameter

                learned_function = lambda world: forces.pairwise_world_lennard_jones_force(world, **func_args)

                advance_args = {**ctx.params, **{'forces':ctx.params['forces'] + [learned_function]}}

                # run the single timestep forward with the perturbed parameter, ceteris paribus
                # does this once in each direction +/- (via loop above), dividing each by the delta
                # accumulates them in the appropriate diff index to get numerical derivative

                different_state = torch.from_numpy(
                    experiments.advance_timestep(
                        ctx.last_state.detach().numpy(),
                        **advance_args
                    )[0][ctx.agent][1:3]
                )

                ##different_state = torch.tensor()##)
                diffs[i] += (different_state - ctx.predicted_state) / d

        ##print(diffs)
        return None, None, None, diffs[0], diffs[1]