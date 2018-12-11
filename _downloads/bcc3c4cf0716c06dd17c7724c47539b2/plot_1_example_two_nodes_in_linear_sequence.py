"""
.. _example1:

First example: A simple linear workflow
-----------------------------------------

Let's start with a simple example that could be perfectly expressed using a :class:`Pipeline`. The data is transformed using a :class:`MinMaxScaler` step and the preprocessed data is fed to a linear model.

Steps of the **PipeGraph**:

- **scaler**: preprocesses the data using a :class:`MinMaxScaler` object
- **linear_model**: fits and predicts a :class:`LinearRegression` model

.. figure:: https://raw.githubusercontent.com/mcasl/PipeGraph/master/examples/images/Diapositiva1.png

    Figure 1. PipeGraph diagram showing the steps and their connections

.. _fig2:

.. figure:: https://raw.githubusercontent.com/mcasl/PipeGraph/master/examples/images/Diapositiva1-2.png

    Figure 2. Illustration of the connections of the PipeGraph
"""

###############################################################################
# Firstly, we import the necessary libraries and create some artificial data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from pipegraph import PipeGraph

X = np.random.rand(100, 1)
y = 4 * X + 0.5*np.random.randn(100, 1)


###############################################################################
# Secondly, we define the steps of the **PipeGraph** as a list of tuples (label, sklearn object) as if we were defining a standard :class:`Pipeline`.

scaler = MinMaxScaler()
linear_model = LinearRegression()
steps = [('scaler', scaler),
        ('linear_model', linear_model)]

###############################################################################
# As the last step is a regressor, a :class:`PipeGraphRegressor` is instantiated.
#  The results from applying ``fit`` and ``predict`` and predict data are shown.

pgraph = PipeGraph(steps=steps)
pgraph.fit(X, y)
y_pred = pgraph.predict(X)

plt.scatter(X, y, label='Original Data')
plt.scatter(X, y_pred, label='Predicted Data')
plt.title('Plots of original and predicted data')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Index')
plt.ylabel('Value of Data')
plt.show()

###############################################################################
# This example described the basic configuration of a :class:`PipeGraphRegressor`. The :ref:`following examples <example2>` show more elaborated cases in order of increasing complexity.


