





import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from tqdm import tqdm

import itertools





def set_up_figure(title='Plot'):
    plt.ion()
    plt.show()
    sns.set_theme()
    fig, ax = plt.subplots(2,1,
                        gridspec_kw={'height_ratios': [4, 1]},
                        figsize=(7.5, 9)
    )
    fig.canvas.set_window_title(title)
    return fig, ax

def trace_trajectories(world, fig, ax, fig_title=''):
    '''
    Performs colored line plots of all particle trajectories in system.
    '''

    ax[0].clear()

    for i in range(world.n_agents):
        sns.scatterplot(
            x=world.history[i::world.n_agents,1],
            y=world.history[i::world.n_agents,2],
            ax=ax[0],
            ci=None
        )
        '''sns.scatterplot(
            x=(world.history[i,1],),
            y=(world.history[i,2],),
            ax=ax[0],

        )'''
    
    fig.canvas.set_window_title(fig_title)
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.01)


def render_state(world,
                    fig,
                    ax,
                    show_indicators=False,
                    indicators=None,
                    indicator_labels=None,
                    fig_title=None,
                    agent_colors=None,
                    agent_sizes=None):
    '''
    Display the particles described in the world array onto the figure and axis provided.
    Activates the plt event loop so that the figure is displayed.
    Also, display the indicator timeseries below if show_indicators=True.

    Expects:
        world state numpy array with only one timestep of data
        plt figure
        axes object with one axis if show_indicators=False, two axes otherwise
    '''

    ax[0].clear()

    p = sns.scatterplot(
            x=world[:,1],
            y=world[:,2],
            c=agent_colors,
            s=agent_sizes,
            ax=ax[0]
    )

    if show_indicators and indicators:
        n_indicators = indicators.shape[1]
        ax[1].clear()
        for ind in range(1, n_indicators+1):
            sns.lineplot(
                x=indicators[:i, 0],
                y=indicators[:i, ind],
                ax=ax[1],
                legend=False
            )
        
        if indicator_labels:
            plt.legend(loc='lower right', labels=indicator_labels)
    
    if fig_title != None:
        fig.canvas.set_window_title(fig_title)
    # if not len(fig.texts):
    #     fig.text(0.01, 0.01, note)
    # else:
    #     fig.texts[0].set_text(note)

    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.01)

def generate_cost_plot(model,
                        data,
                        criterion,
                        param_ranges,
                        index_range):
    '''

    '''

    ##selected_sample = 0 ##int(len(data)/2)
    ##x, y = data[selected_sample]

    ordered_keys = list(param_ranges.keys())
    ordered_ranges = [param_ranges[k] for k in ordered_keys]
    combinations = list(itertools.product(*ordered_ranges))

    cost_data = np.empty( (len(combinations), len(ordered_keys)+1) )

    for k in tqdm(index_range):
        x, y = data[k]
        for j in range(len(combinations)):
            for i in range(len(ordered_keys)):
                model.__getattr__(ordered_keys[i]).data = torch.Tensor([combinations[j][i]])

            ##print(x)
            
            y_pred = model(0, x)
            y_pred = y_pred.detach()
            cost_data[j] = np.add(cost_data[j], (criterion(y_pred, y), *combinations[j]))

    cost_data = cost_data / len(index_range)

    ##print(cost_data)
    ##plt.contour(cost_data)

    ##print(cost_data)

    if (len(ordered_keys) == 2): # can be plotted for mortals' understanding, no hypercubes...
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(cost_data[:,1], cost_data[:,2], cost_data[:,0])
        ax.set_xlabel(ordered_keys[0])
        ax.set_ylabel(ordered_keys[1])
        ax.set_zlabel("cost")
        plt.show()