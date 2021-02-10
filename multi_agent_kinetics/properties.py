





import numpy as np
import scipy
from . import forces, kernels






def pressure(state, particle_i, h):
    '''
    '''

    rho = 0

    for i in range(state.shape[0]):
        if i == particle_i: continue
        rho = rho + 10 * kernels.quadratic_kernel(
                                np.linalg.norm(
                                    state[i, 1:3],
                                    state[particle_i, 1:3]
                                ),
                            h)
    
    return rho


def density_all(state):
    '''
    Calculates the density for a particle given world state.
    '''

    # isolate particle position data
    coords = state[:,1:3]
    p_dists = scipy.spatial.distance.pdist(coords)
    kernel = scipy.spatial.distance.squareform(
        [kernels.cubic_spline(d) for d in p_dists]
    )
    kernel *= 10 # HARD CODED MASS

    return np.sum(kernel, axis=1)

