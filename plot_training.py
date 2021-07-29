import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_training_graph(data_df):
    df = data_df.loc[:, ~data_df.columns.str.contains('^Unnamed')]
    df.rename(
        columns={
            'pos': 'Position Reconstruction Error',
            'vel': 'Velocity Reconstruction Error',
            'ham': 'Hamiltonian Reconstruction Error',
            'guess': 'Learned G Value',
        },
        inplace=True)
    fig = plt.figure()
    ax = fig.gca()
    df.plot(ax=ax)
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.ylabel("Loss / Parameter Value")
    plt.xlabel("Number of Training Examples Seen")
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    plot_training_graph(df)