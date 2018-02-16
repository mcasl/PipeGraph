"""
.. _example5:

Fifth Example: Demultiplexor - multiplexor
-----------------------------------------------

An imaginative layout using a classifier to predict the cluster labels and fitting a separate model for each cluster.

Steps of the PipeGraph:

- **scaler**: A :class:`MinMaxScaler` data preprocessor
- **classifier**: A :class:`GaussianMixture` classifier
- **demux**: A custom :class:`Demultiplexer` class in charge of splitting the input arrays accordingly to the selection input vector
- **lm_0**: A :class:`LinearRegression` model
- **lm_1**: A :class:`LinearRegression` model
- **lm_2**: A :class:`LinearRegression` model
- **mux**: A custom :class:`Multiplexer` class in charge of combining different input arrays into a single one accordingly to the selection input vector

.. figure:: https://raw.githubusercontent.com/mcasl/PipeGraph/master/examples/images/Diapositiva5.png

    Figure 1. PipeGraph diagram showing the steps and their connections
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from pipegraph.pipeGraph import PipeGraph, Multiplexer, Demultiplexer
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

X_first = pd.Series(np.random.rand(100,))
y_first = pd.Series(4 * X_first + 0.5*np.random.randn(100,))
X_second = pd.Series(np.random.rand(100,) + 3)
y_second = pd.Series(-4 * X_second + 0.5*np.random.randn(100,))
X_third = pd.Series(np.random.rand(100,) + 6)
y_third = pd.Series(2 * X_third + 0.5*np.random.randn(100,))

X = pd.concat([X_first, X_second, X_third], axis=0).to_frame()
y = pd.concat([y_first, y_second, y_third], axis=0).to_frame()

scaler = MinMaxScaler()
gaussian_mixture = GaussianMixture(n_components=3)
demux = Demultiplexer()
lm_0 = LinearRegression()
lm_1 = LinearRegression()
lm_2 = LinearRegression()
mux = Multiplexer()


steps = [('scaler', scaler),
         ('classifier', gaussian_mixture),
         ('demux', demux),
         ('lm_0', lm_0),
         ('lm_1', lm_1),
         ('lm_2', lm_2),
         ('mux', mux), ]

connections = { 'scaler': {'X': 'X'},

                'classifier': {'X': ('scaler', 'predict')},

                'demux': {'X': ('scaler', 'predict'),
                          'y': 'y',
                          'selection': ('classifier', 'predict')},

                'lm_0': {'X': ('demux', 'X_0'),
                         'y': ('demux', 'y_0')},

                'lm_1': {'X': ('demux', 'X_1'),
                         'y': ('demux', 'y_1')},

                'lm_2': {'X': ('demux', 'X_2'),
                         'y': ('demux', 'y_2')},

                'mux': {'0': ('lm_0', 'predict'),
                        '1': ('lm_1', 'predict'),
                        '2': ('lm_2', 'predict'),
                        'selection': ('classifier', 'predict')}}

###############################################################################
# Use PipeGraph when predict needs (``X``,``y``)

pgraph = PipeGraph(steps=steps, connections=connections)
pgraph.fit(X, y)
#%%
y_pred = pgraph.predict(X, y=None)['predict']
plt.scatter(X, y)
plt.scatter(X, y_pred)

