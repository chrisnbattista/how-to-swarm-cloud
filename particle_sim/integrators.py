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

    print( (world + displacement_matrix).fillna(world) )

    return (world + displacement_matrix).fillna(world)