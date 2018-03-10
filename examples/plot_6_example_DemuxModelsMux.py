"""
.. _example6:

Sixth Example: Demultiplexor - multiplexor
-----------------------------------------------

A demonstration of serveral interesting features:
 1. How the user can choose to encapsulate several blocks into a PipeGraph and use it as a single unit in another PipeGraph
 2. How these components can be dynamically built on runtime depending on initialization parameters
 3. How these components can be dynamically built on runtime depending on input signal values during fit


Encapsulating several blocks into a PipeGraph and using them as a single unit in another PipeGraph
==================================================================================================

We consider Example number five in which we had the following steps:


- **scaler**: A :class:`MinMaxScaler` data preprocessor
- **classifier**: A :class:`GaussianMixture` classifier
- **demux** in charge of splitting the input arrays accordingly to the selection input vector
- **lm_0**: A :class:`LinearRegression` model
- **lm_1**: A :class:`LinearRegression` model
- **lm_2**: A :class:`LinearRegression` model
- **mux**: A custom :class:`Multiplexer` class in charge of combining different input arrays into a single one accordingly to the selection input vector

For instance, we can find interesting to encapsulate the Demultiplexer, the linear model collection, and the Multiplexer into a single unit:
We build a PipeGraph with these steps alone:
"""
from sklearn.linear_model import LinearRegression
from pipegraph.standard_blocks import Demultiplexer, Multiplexer
from pipegraph.base import PipeGraph

demux = Demultiplexer()
lm_0 = LinearRegression()
lm_1 = LinearRegression()
lm_2 = LinearRegression()
mux = Multiplexer()

three_multiplexed_models_steps = [
         ('demux', demux),
         ('lm_0', lm_0),
         ('lm_1', lm_1),
         ('lm_2', lm_2),
         ('mux', mux), ]

three_multiplexed_models_connections = {
                'demux': {'X': 'X',
                          'y': 'y',
                          'selection': 'selection'},
                'lm_0': {'X': ('demux', 'X_0'),
                         'y': ('demux', 'y_0')},
                'lm_1': {'X': ('demux', 'X_1'),
                         'y': ('demux', 'y_1')},
                'lm_2': {'X': ('demux', 'X_2'),
                         'y': ('demux', 'y_2')},
                'mux': {'0': 'lm_0',
                        '1': 'lm_1',
                        '2': 'lm_2',
                        'selection': 'selection'}}

three_multiplexed_models = PipeGraph(steps=three_multiplexed_models_steps,
                                     fit_connections=three_multiplexed_models_connections )

#########################################################################################################
#  Now we can use this PipeGraph as a reusable component in another PipeGraph:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from pipegraph.base import PipeGraphRegressor

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
models = three_multiplexed_models

steps = [('scaler', scaler),
         ('classifier', gaussian_mixture),
         ('models', three_multiplexed_models), ]

connections = {'scaler': {'X': 'X'},
               'classifier': {'X': 'scaler'},
               'models': {'X': 'scaler',
                          'y': 'y',
                          'selection': 'classifier'},
               }

pgraph = PipeGraphRegressor(steps=steps, fit_connections=connections)
pgraph.fit(X, y)

y_pred = pgraph.predict(X)

plt.scatter(X, y)
plt.scatter(X, y_pred)

################################################################################################
# Now we can think of programatically change the number of models inside this component.
# First we do it by using initialization parameters in a PipeGraph subclass
from pipegraph.standard_blocks import DemuxModelsMux

models = DemuxModelsMux(number_of_replicas=3,
                        model_class=LinearRegression,
                        model_parameters={})

steps = [('scaler', scaler),
         ('classifier', gaussian_mixture),
         ('models', models), ]

connections = {'scaler': {'X': 'X'},
               'classifier': {'X': 'scaler'},
               'models': {'X': 'scaler',
                          'y': 'y',
                          'selection': 'classifier'},
               }

pgraph = PipeGraphRegressor(steps=steps, fit_connections=connections)
pgraph.fit(X, y)

y_pred = pgraph.predict(X)

plt.scatter(X, y)
plt.scatter(X, y_pred)

