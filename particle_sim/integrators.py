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

    displacement_matrix = double_integrate_rect(timestep, force_matrix)
    displacement_matrix['m'] = world['m']
    displacement_matrix[['b_1', 'b_2']] = displacement_matrix[['b_1', 'b_2']].div(displacement_matrix['m'], axis=0)
    displacement_matrix.drop(columns=['m'], inplace=True)

    ##print(world)
    ##print(displacement_matrix)

    return (world + displacement_matrix).fillna(world)