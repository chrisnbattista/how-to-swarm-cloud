import matplotlib.pyplot as plt
import random, glob
import pandas as pd

## Select dataset
data_choice = input("which dataset>").strip()

## Get data
path = random.choice(
    glob.glob(f'./data/two_particle/{data_choice}/*.csv')
)
print('Loading random data file from dataset...')
data = pd.read_csv(
                path,
                delimiter=',', 
                index_col=False
            )

## Plot trace
fig = plt.figure()
plt.scatter(
                    x=data['b_1'],
                    y=data['b_2'],
                    s=1
                )
plt.title('Parametric plot of x(t)')
fig.canvas.set_window_title('Trace')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()