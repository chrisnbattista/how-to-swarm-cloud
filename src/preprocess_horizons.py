import pandas as pd
import torch
from tqdm import tqdm
from multi_agent_kinetics import worlds
from multi_agent_kinetics.indicators import hamiltonian
from multi_agent_kinetics import potentials

df = pd.read_csv('../data/3D/horizons/data.notcsv').drop('time', axis=1)
print('Raw DF:')
print(df)
##df = df.groupby('t')
##print(df)

bodies = ['sun', 'jupiter', 'saturn']
masses = {
    'sun': 1.989 * (10**30),
    'jupiter': 1.898 * (10**27),
    'saturn': 5.683 * (10**26)
}
data = []

D_SCALE = 1.489600e+8
M_SCALE = 4.959563071040455e+27
T_SCALE = 31.6 * 10**2

V_SCALE = D_SCALE / T_SCALE
E_SCALE = M_SCALE * D_SCALE**2 / (T_SCALE**2)

G = 6.67408 * (10**(-11))

# '3d': ('t', 'id', 'm', 'b_1', 'b_2', 'b_3', 'v_1', 'v_2', 'v_3')

for i in range(0, len(df['t'])):
    for j in range(len(bodies)):
        k = i * j
        data.append([
            df['t'][i] * 24.0 * 3600 / T_SCALE, j, masses[bodies[j]] / M_SCALE,
            df['x_' + bodies[j]][i] * 1000 / D_SCALE,
            df['y_' + bodies[j]][i] * 1000 / D_SCALE,
            df['z_' + bodies[j]][i] * 1000 / D_SCALE,
            df['vx_' + bodies[j]][i] * 1000 / V_SCALE,
            df['vy_' + bodies[j]][i] * 1000 / V_SCALE,
            df['vz_' + bodies[j]][i] * 1000 / V_SCALE, 0
        ])

processed_df = pd.DataFrame(data,
                            columns=list(worlds.schemas['3d']) +
                            ['Hamiltonian']).set_index('t')

print("Converted DF:")
print(processed_df)

# for i in tqdm(range(0, len(df['t']))):
#     processed_df['Hamiltonian'][i * 3:(i + 1) * 3] = (
#         1.0 / E_SCALE) * potentials.gravitational_potential_energy_from_state(
#             torch.tensor(processed_df[i * 3:(i + 1) * 3].to_numpy()),
#             spatial_dims=3,
#             G=G) + potentials.kinetic_energy_from_state(torch.tensor(
#                 processed_df[i * 3:(i + 1) * 3].to_numpy()),
#                                                         spatial_dims=3)

# print("Instrumented DF:")
# print(processed_df)
processed_df.to_csv('../data/3D/horizons/data_processed.csv')