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
from pipegraph import PipeGraphRegressor

X = np.random.rand(100, 1)
y = 4 * X + 0.5*np.random.randn(100, 1)


###############################################################################
# Secondly, we define the steps of the **PipeGraph** as a list of tuples (label, sklearn object) as if we were defining a standard :class:`Pipeline`.

scaler = MinMaxScaler()
linear_model = LinearRegression()
steps = [('scaler', scaler),
        ('linear_model', linear_model)]

###############################################################################
# Then, we need to provide the connections amongst steps as a dictionary. Please, refer to :ref:`Figure 2 <fig2>` for a faster understanding.
#
# - The keys of the top level entries of the dictionary must be the same as those of the previously defined steps.
# - The values assocciated to these keys define the variables from other steps that are going to be considered as inputs for the current step. They are dictionaries themselves, where:
#
#   - The keys of the nested dictionary represent the input variables as named at the current step.
#   - The values assocciated to these keys define the steps that hold the desired information and the variables as named at that step. This information can be written as:
#
#     - A tuple with the label of the step in position 0 followed by the name of the output variable in position 1.
#     - A string representing a variable from an external source to the :class:`PipeGraphRegressor` object, such as those provided by the user while invoking the ``fit``, ``predict`` or ``fit_predict`` methods.
#
# For instance, the linear model accepts as input ``X`` the output named ``predict`` at the ``scaler`` step, and as input ``y`` the value of ``y`` passed to ``fit``, ``predict`` or ``fit_predict`` methods.

connections = { 'scaler': { 'X': 'X'},
                'linear_model': {'X': ('scaler', 'predict'),
                                 'y': 'y'}}

###############################################################################
# Once the steps and the connections are defined, we instantiate a :class:`PipeGraphRegressor` and show the results from applying ``fit`` and ``predict`` and predict data. Use PipeGraphRegressor when the result is a regression.

pgraph = PipeGraphRegressor(steps=steps, connections=connections)
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


