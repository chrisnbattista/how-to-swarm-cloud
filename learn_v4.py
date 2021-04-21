import numpy as np
import pandas as pd
import random, glob
from multi_agent_kinetics import serialize
from tqdm import tqdm

## Get data
paths = glob.glob('./data/two_particle/*/*.csv')
dataset = []
print('Loading {} data files...'.format(len(paths)))
for p in tqdm(range(len(paths))):
    d = pd.read_csv(
                    paths[p],
                    delimiter=',', 
                    index_col=False
                )
    particle_1_data = d[d['id'].astype(int) == 0].rename(
        columns=\
            {
                'b_1': 'x_11',
                'b_2': 'x_12',
                'v_1': 'x_dot_11',
                'v_2': 'x_dot_12',
            }
    ).set_index('t').drop(columns=['id', 'm'])
    particle_2_data = d[d['id'].astype(int) == 1].rename(
        columns=\
            {
                'b_1': 'x_21',
                'b_2': 'x_22',
                'v_1': 'x_dot_21',
                'v_2': 'x_dot_22',
            }
    ).set_index('t').drop(columns=['id', 'm'])
    d_2 = pd.concat([particle_1_data, particle_2_data], axis=1)
    dataset.append(d_2)
data = pd.concat(dataset, axis=0, ignore_index=True)
print('Data columns: {}'.format(data.columns))

## Preprocess data
data['iad'] = np.sqrt(
                (data['x_11'] - data['x_21'])**2 \
                + (data['x_12'] - data['x_22'])**2
)
data['speed_1'] = np.sqrt(
                data['x_dot_11']**2 \
                + data['x_dot_12']**2
)
data['speed_2'] = np.sqrt(
                data['x_dot_21']**2 \
                + data['x_dot_22']**2
)
data['phi'] = data['speed_1'] / data['iad']
print(data)