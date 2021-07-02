import numpy as np
import torch
from multi_agent_kinetics import experiments, forces, sim, worlds, indicators, viz
import hts.learning.tb_logging

class PhysicsForwardRun (torch.autograd.Function):
    '''
    PyTorch-compatible function taking an initial world state and force configuration
    and yielding all particle trajectories over n timesteps.
    '''

    @staticmethod
    def forward(ctx, agent, initial_state, params={}, sigma=torch.Tensor([0]), epsilon=torch.Tensor([0])):
        '''
        '''

        ctx.agent = agent
        ctx.initial_state = initial_state
        full_params = {**params, **{
            'sigma': sigma,
            'epsilon': epsilon
        }}
        ctx.full_params = full_params

        # Function construction
        learned_function = lambda world, context: forces.pairwise_world_lennard_jones_force(
            world,
            **{
                'epsilon': epsilon.data,
                'sigma': sigma.data
            },
            context=context
        )

        ##print(f'{epsilon.data} {sigma.data}')

        world = worlds.World(
            initial_state=initial_state,
            forces=[learned_function],
            **params
        )

        # Run forward certain number of steps and return all trajectories
        world.advance_state(full_params['n_timesteps']-1)
        ##viz.trace_trajectories(world, *viz.set_up_figure())
        ctx.history = world.get_history().copy()
        return torch.Tensor(world.get_history()[agent+params['n_agents']::params['n_agents'],3:5])

    @staticmethod
    def backward(ctx, *grad_output):
        '''
        Does finite difference with two terms to find derivative value for each parameter by rerunning sim.
        '''

        d = 1
        d_params = {}

        for p in ['sigma', 'epsilon']:

            plus_params = {**ctx.full_params, **{
                p: ctx.full_params[p] + d
            }}
            learned_plus_function = lambda world, context: forces.pairwise_world_lennard_jones_force(world, plus_params['epsilon'], plus_params['sigma'])

            minus_params = {**ctx.full_params, **{
                p: ctx.full_params[p] + d
            }}
            learned_minus_function = lambda world, context: forces.pairwise_world_lennard_jones_force(world, minus_params['epsilon'], minus_params['sigma'])
            
            plus_world = worlds.World(
                initial_state=ctx.initial_state,
                forces=[learned_plus_function],
                **plus_params
            )
            plus_world.advance_state(plus_params['n_timesteps']-1)
            plus_d_trajs = plus_world.get_history()

            minus_world = worlds.World(
                initial_state=ctx.initial_state,
                forces=[learned_minus_function],
                **minus_params
            )
            minus_world.advance_state(minus_params['n_timesteps']-1)
            minus_d_trajs = minus_world.get_history()

            print(plus_d_trajs)

            viz.trace_predicted_vs_real_trajectories(ctx.history, plus_d_trajs, '', *viz.set_up_figure())

            plus_d_cost = indicators.mse_trajectories(ctx.history, plus_d_trajs, ctx.full_params['n_agents'])
            minus_d_cost = indicators.mse_trajectories(ctx.history, minus_d_trajs, ctx.full_params['n_agents'])

            d_params[p] = torch.Tensor([plus_d_cost - minus_d_cost])

            print(plus_d_cost)
            print(minus_d_cost)
        
        ##print(d_params['sigma'])
        ##print(d_params['epsilon'])
        
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
        learned_function = lambda world, context: forces.pairwise_world_lennard_jones_force(
            world,
            **{ \
                'epsilon': epsilon.data,
                'sigma': sigma.data
            },
            context=context
        )

        # Setup
        args = {**params, **{'forces':[learned_function]}}

        world = worlds.World(
            initial_state=last_state,
            **args
        )

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
        state = torch.tensor(next_state[agent][3:5], requires_grad=True)
        ctx.predicted_state = state.float().clone().detach()

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

                different_state = torch.tensor(PhysicsEngine.time_step(ctx.last_state, ctx.params, p['sigma'], p['epsilon']))

                diffs[i] += (different_state[ctx.agent][3:5] - ctx.predicted_state) / d

        ##tb_logging.writer.add_scalar("Average error", diffs[0].mean(), tb_logging.epoch)
        return None, None, None, diffs[0], diffs[1]