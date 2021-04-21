import numpy as np
import pandas as pd
import random, glob, warnings
from multi_agent_kinetics import serialize
from tqdm import tqdm
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

## Select dataset
choice = input("which dataset>").strip()

## Get data
paths = glob.glob(f'./data/two_particle/{choice}/*.csv')
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
msk = np.random.rand(len(data)) < 0.8
train = data[msk]
test = data[~msk]

errors = []

## Train linear model
#poly_transformer = PolynomialFeatures(degree=3)
regr = linear_model.LinearRegression()
regr.fit(train['iad'].values.reshape(-1, 1), train['phi'])

## Test linear model
y_pred = regr.predict(test['iad'].values.reshape(-1, 1))
phi = np.array(test['phi'])
error = np.divide(
    np.subtract(y_pred, phi),
    phi
)
error = error[np.isfinite(error)]
errors.append(np.abs(np.average(error)))
print(f'---\nLinear model\nMean error: {np.abs(np.average(error)) * 100:.2f}%')

with warnings.catch_warnings():

    ## Try multiple degrees of polynomial models
    for degree in range(2, 10):

        ## Train polynomial model
        poly_transformer = PolynomialFeatures(degree=degree)
        transformed_features = poly_transformer.fit_transform(train['iad'].values.reshape(-1, 1))
        regr2 = linear_model.LinearRegression()
        regr2.fit(transformed_features, train['phi'])

        ## Test polynomial model
        transformed_features_test = poly_transformer.fit_transform(test['iad'].values.reshape(-1, 1))
        y_pred2 = regr2.predict(transformed_features_test)
        phi = np.array(test['phi'])
        error2 = np.divide(
            np.subtract(y_pred2, phi),
            phi
        )
        error2 = error2[np.isfinite(error2)]
        errors.append(np.abs(np.average(error2)))
        print(f'---\nPolynomial (d={degree}) model\nMean error: {np.abs(np.average(error2)) * 100:.2f}%')

## Plot error
plt.plot(range(1, 10), [e*100 for e in errors])
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.title('Naive fits of kernel function')
plt.xlabel('Degree of fit')
plt.ylabel('Percentage Error')
plt.show()