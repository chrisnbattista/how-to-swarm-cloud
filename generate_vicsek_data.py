import numpy as np
from multi_agent_kinetics import single_integrator
import random, os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import normalize
import pandas as pd

do_plot = False

dyn = 'vicsek'
scaled = input("Scaled>?")
if scaled:
    dyn = 'lin'

def vicsek(row_i, timestep, data, v=0.001):
    row_j = data[data['id'] != row_i['id']]
    iadis = row_j[['b_1', 'b_2']].values - row_i[['b_1', 'b_2']].values # inter-agent displacement
    x_dot = normalize(iadis) * v
    return x_dot

def lin(row_i, timestep, data, v=0.001):
    row_j = data[data['id'] != row_i['id']]
    iadis = row_j[['b_1', 'b_2']].values - row_i[['b_1', 'b_2']].values # inter-agent displacement
    x_dot = iadis * v
    return x_dot

try: os.mkdir('./data/two_particle/'+dyn)
except FileExistsError: pass
for i in tqdm(range(1000)):
    data = single_integrator.run_sim(dynamics=eval(dyn), radius=3)
    np.savetxt(f"./data/two_particle/{dyn}/{random.randint(15345, 2524323254)}.csv",
            data,
            comments='',
            delimiter=',',
            fmt='%10.6f',
            header=','.join(['t', 'id', 'm', 'b_1', 'b_2', 'v_1', 'v_2'])
        )
    if do_plot:
        plt.scatter(
                    x=data['b_1'],
                    y=data['b_2']
                )
        plt.show()