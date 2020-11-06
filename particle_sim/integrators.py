import numpy as np
import pandas as pd

## Simple scalar integration functions

def integrate_rect(delta_t, initial_value):
    return initial_value * delta_t

def double_integrate_rect(delta_t, initial_value):
    return initial_value * delta_t**2


## World level integration functions

def integrate_rect_world(world, force_matrix, timestep):
    '''
    '''

    velocity_matrix = integrate_rect(timestep, force_matrix)
    world['v_1'] += velocity_matrix['b_1']
    world['v_2'] += velocity_matrix['b_2']

    displacement_matrix = integrate_rect(timestep, world[['v_1', 'v_2']])
    displacement_matrix.rename(
        columns={
            'v_1': 'b_1',
            'v_2': 'b_2'
        },
        inplace=True
    )

    displacement_matrix['m'] = world['m']
    displacement_matrix[['b_1', 'b_2']] = displacement_matrix[['b_1', 'b_2']].div(displacement_matrix['m'], axis=0)
    displacement_matrix.drop(columns=['m'], inplace=True)

    return (world + displacement_matrix).fillna(world)