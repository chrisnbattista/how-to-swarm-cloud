import numpy as np

def lennard_jones_potential(epsilon, omega, r):
    return 4 * epsilon * ( (omega / r)**12 - (omega / r)**6 )