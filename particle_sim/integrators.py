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

    # Velocity from force
    # F = ma = m * derivative(v)
    # v = integral(F) / m
    velocity_matrix = integrate_rect(timestep, force_matrix)  / 10 # MUST FIX - constant mass
    world[:, 4:6] += velocity_matrix

    # Displacement from velocity
    world[:, 1:3] += integrate_rect(timestep, world[:, 4:6])

    return world
