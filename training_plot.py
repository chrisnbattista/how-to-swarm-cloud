import pandas as pd
import matplotlib.pyplot as plt
import sys

df = pd.read_csv(sys.argv[1])
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
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