





import torch
from multi_agent_kinetics import experiments, forces, sim, worlds, indicators
import hts.learning.tb_logging






class PhysicsForwardRun (torch.autograd.Function):
    '''
    PyTorch-compatible function taking an initial world state and force configuration
    and yielding all particle trajectories over n timesteps.
    '''

    @staticmethod
    def forward(ctx, agent, initial_state, params={}, sigma=0, epsilon=0):
        '''
        '''

        broadcasted_ic = initial_state[None, :]

        ctx.agent = agent
        ctx.initial_state = broadcasted_ic
        full_params = {**params, **{
            'sigma': sigma,
            'epsilon': epsilon
        }}

        ctx.full_params = full_params

        world = worlds.World(
            initial_state=initial_state,
            **full_params
        )

        # Run forward certain number of steps and return all trajectories
        world.advance_state(full_params['n_steps'])
        ctx.history = world.get_history().copy()
        return world.get_history()

    @staticmethod
    def backward(ctx, grad_output):
        '''
        Does finite difference with two terms to find derivative value for each parameter by rerunning sim.
        '''

        d = 0.1
        d_params = {}

        for p in ['sigma', 'epsilon']:

            plus_params = {**ctx.full_params, **{
                p: ctx.full_params[p] + d
            }}

            minus_params = {**ctx.full_params, **{
                p: ctx.full_params[p] + d
            }}
            
            plus_world = worlds.World(
                initial_state=ctx.initial_state,
                **plus_params
            )
            plus_world.advance_state(plus_params['n_steps'])
            plus_d_trajs = plus_world.get_history()

            minus_world = worlds.World(
                initial_state=ctx.initial_state,
                **minus_params
            )
            minus_world.advance_state(minus_params['n_steps'])
            minus_d_trajs = minus_world.get_history()

            plus_d_cost = indicators.mse_trajectories(ctx.history, plus_d_trajs, ctx.full_params['n_particles'])
            minus_d_cost = indicators.mse_trajectories(ctx.history, minus_d_trajs, ctx.full_params['n_particles'])

            d_params[p] = (plus_d_cost - minus_d_cost)
        
        return None, None, None, d_params['sigma'], d_params['epsilon']



class PhysicsEngine:
    '''
    Abstract base class
    '''

    @staticmethod
    def time_step(last_state, params, sigma, epsilon):
        '''
        Utility function to standardize forward step
        Uses strong Markov assumption
        '''

        # Function construction
        learned_function = lambda world: forces.pairwise_world_lennard_jones_force(world, **{ \
            'epsilon': epsilon.data,
            'sigma': sigma.data
        })

        # Setup
        args = {**params, **{'forces':params['forces'] + [learned_function]}}

        world = worlds.World(
            initial_state=last_state,
            **args
        )

        ##print(last_state)
        ##print(world.get_state())

        # The actual forward step
        world.advance_state()
        

        return world.get_state()


class PhysicsStep (torch.autograd.Function, PhysicsEngine):
    '''
    PyTorch-compatible function taking a world state and force configuration and yielding a predicted position for one agent.
    '''

    @staticmethod
    def forward(ctx, agent, last_state, params={}, sigma=0, epsilon=0):
        '''
        '''

        # Context saving
        ctx.agent = agent
        ctx.last_state = last_state
        ctx.params = params
        ctx.sigma = sigma
        ctx.epsilon = epsilon

        # Time step
        next_state = PhysicsEngine.time_step(last_state, params, sigma, epsilon)
        
        # Downselect to requested agent
        state = torch.tensor(next_state[agent][1:3], requires_grad=True)
        ctx.predicted_state = state

        return state
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
        '''

        to_diff = ('sigma', 'epsilon')
        diffs = [torch.Tensor([0, 0]) for _ in range(2)]

        delta = 0.1

        # loop through different parameters to take partials
        # utilizing Central Difference Theorem
        for i in range(len(to_diff)):
            # loop through the positive and negative perturbations (three-point finite difference)
            for d in (delta, -delta):

                p = {
                    'epsilon': ctx.epsilon.data,
                    'sigma': ctx.sigma.data
                }
                p.update({to_diff[i]: p[to_diff[i]] + d}) # add perturbation to specified parameter

                # run the single timestep forward with the perturbed parameter, ceteris paribus
                # does this once in each direction +/- (via loop above), dividing each by the delta
                # accumulates them in the appropriate diff index to get numerical derivative

                different_state = super().time_step(ctx.last_state, ctx.params, p['sigma'], p['epsilon'])

                diffs[i] += (different_state[ctx.agent][1:3] - ctx.predicted_state) / d

        ##tb_logging.writer.add_scalar("Average error", diffs[0].mean(), tb_logging.epoch)
        return None, None, None, diffs[0], diffs[1]