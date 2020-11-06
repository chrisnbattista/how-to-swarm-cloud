import numpy as np

def lennard_jones_potential(epsilon, omega, r):
    return (4 * epsilon * omega**12)/r**12 - (4 * epsilon * omega**6)/r**6

def pairwise_world_lennard_jones_potential(world, epsilon, omega):
    '''
    '''
    potential_matrix = np.zeros( (world.shape[0], 2) )
    for i in range(world.shape[0]):
        for j in range(world.shape[0]):
            if i == j: continue
            norm = np.linalg.norm(world.iloc[j][['b_1', 'b_2']] - world.iloc[i][['b_1', 'b_2']])
            magnitude = lennard_jones_potential(epsilon, omega, norm)
            direction = ((world.iloc[j] - world.iloc[i])/norm)[['b_1', 'b_2']]
            ##print(f"norm: {norm} magnitude: {magnitude} direction: {direction}")
            ##print(direction * magnitude)
            potential_matrix[j] += direction * magnitude
    
    ##print(potential_matrix)
    return potential_matrix