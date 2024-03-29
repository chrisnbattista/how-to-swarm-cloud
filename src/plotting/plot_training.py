import pandas as pd
import matplotlib.pyplot as plt
import sys


def plot_training_graph(data_df):
    df = data_df.loc[:, ~data_df.columns.str.contains('^Unnamed')]
    df.rename(columns={
        'pos': 'Position Reconstruction Error',
        'vel': 'Velocity Reconstruction Error',
        'ham': 'Hamiltonian Reconstruction Error',
        'g': 'Learned G Value',
        'r': 'Learned r Value',
    },
              inplace=True)
    fig, axs = plt.subplots(2)
    fig.suptitle('Parameter Values and Training Loss versus Samples Seen')
    axs[0].xaxis.get_major_locator().set_params(integer=True)
    axs[0].plot(df.index, df['Position Reconstruction Error'])
    axs[0].plot(df.index, df['Velocity Reconstruction Error'])
    axs[1].plot(df.index, df['Learned G Value'])
    axs[1].plot(df.index, df['Learned r Value'])
    axs[0].set(ylabel="Loss Component",
               xlabel="Number of Training Examples Seen")
    axs[1].set(ylabel="Parameter Value",
               xlabel="Number of Training Examples Seen")
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    plot_training_graph(df)

## TODO: Log scale error plots
## TODO: plot total loss function value

## TODO: use different line types and markers. zoom in view at beginning
## TODO: make sure terms are the same in the figures and the paper copy (LaTEX)
## TODO: possibly separate loss and parameter value plots into subplot arrangement. yes, make G_error/G_actual and r_error/r_actual plot and separate reconstruction errors into own plot
# Plot cost function?
# show performance with different tuning parameters
# consider 2D plot showing loss landscape across multiple loss measures. how to show many cases for holistic algorithm performance / dependence characterization.
## TODO ^ this shows tradeoff between learning rate and accuracy
## Reasoning : argument behind error plot. Binary convergence or speed to convergence. particularly segment length

## WHAT IS THE PUNCHLINE?
# Error plot: we can achieve high accuracy
# Tradeoff: can learn quickly, Hamiltonian helps

## TODO: no magic numbers! (unless evidence!)