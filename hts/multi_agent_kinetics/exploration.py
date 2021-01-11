





import torch
import matplotlib.pyplot as plt
import numpy as np

import itertools





def generate_cost_plot(model,
                        data,
                        criterion,
                        param_ranges):
    '''

    '''

    selected_sample = 0 ##int(len(data)/2)
    x, y = data[selected_sample]
    print(x)
    print(y)

    ordered_keys = list(param_ranges.keys())
    ordered_ranges = [param_ranges[k] for k in ordered_keys]
    combinations = list(itertools.product(*ordered_ranges))

    cost_data = np.empty( (len(combinations), len(ordered_keys)+1) )

    for j in range(len(combinations)):
        for i in range(len(ordered_keys)):
            model.__getattr__(ordered_keys[i]).data = torch.Tensor([combinations[j][i]])
        y_pred = model(0, x)
        y_pred = y_pred.detach()
        cost_data[j] = (criterion(y_pred, y), *combinations[j])

    ##print(cost_data)
    ##plt.contour(cost_data)

    if (len(ordered_keys) == 2): # can be plotted for mortals' understanding, no hypercubes...
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(cost_data[:,1], cost_data[:,2], cost_data[:,0])
        ax.set_xlabel(ordered_keys[0])
        ax.set_ylabel(ordered_keys[1])
        ax.set_zlabel("cost")

        plt.show()