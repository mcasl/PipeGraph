"""
.. _example2:

Second example: :class:`GridSearchCV` demonstration
----------------------------------------------------

This example shows how to use :class:`GridSearchCv` with PipeGraph to effectively fit the best model across a number of hyperparameters.
It is equivalent to use :class:`GridSearchCv` with :class:`Pipeline`. More complicated cases are shown in the following examples. In this second example we wanted to show how to fit a :class:`GridSearchCV` in a yet simple scenario.

Steps of the PipeGraph:

- **scaler**: a preprocessing step using a :class:`MinMaxScaler` object
- **polynomial_features**: a transformer step
- **linear_model**: the :class:`LinearRegression` object we want to fit and use for predict.

.. figure:: https://raw.githubusercontent.com/mcasl/PipeGraph/master/examples/images/Diapositiva2.png

    Figure 1. PipeGraph diagram showing the steps and their connections
"""

###############################################################################
# Firstly, we import the necessary libraries and create some artificial data.

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from pipegraph.pipeGraph import PipeGraphRegressor

import matplotlib.pyplot as plt

X = 2*np.random.rand(100,1)-1
y = 40 * X**5 + 3*X*2 +  3*X + 3*np.random.randn(100,1)

scaler = MinMaxScaler()
polynomial_features = PolynomialFeatures()
linear_model = LinearRegression()

###############################################################################
# Secondly, we define the steps and connections in a similar manner as we described in the :ref:`first example <example1>`. A ``param_grid`` dictionary is defined as specified by :class:`GridSearchCV`.
# In this case we just want to explore a few possibilities varying the degree of the polynomials and whether to use or not an intercept at the linear model.

steps = [('scaler', scaler),
         ('polynomial_features', polynomial_features),
         ('linear_model', linear_model)]

connections = {'scaler': { 'X': 'X'},
               'polynomial_features': {'X': ('scaler', 'predict')},
               'linear_model': {'X': ('polynomial_features', 'predict'),
                                'y': 'y' }  }

param_grid = {'polynomial_features__degree': range(1, 11),
              'linear_model__fit_intercept': [True, False]}

###############################################################################
# Now, we use ``PipeGraphRegressor`` as estimator for :class:`GridSearchCV` and perform the ``fit`` and ``predict`` operations.

pgraph = PipeGraphRegressor(steps=steps, connections=connections)
grid_search_regressor  = GridSearchCV(estimator=pgraph, param_grid=param_grid, refit=True)
grid_search_regressor.fit(X, y)
y_pred = grid_search_regressor.predict(X)

plt.scatter(X, y)
plt.scatter(X, y_pred)
plt.show()


coef = grid_search_regressor.best_estimator_.get_params()['linear_model'].coef_
degree = grid_search_regressor.best_estimator_.get_params()['polynomial_features'].degree

print('Information about the parameters of the best estimator: \n degree: {} \n coefficients: {} '.format(degree, coef))

###############################################################################
# This example showed how to use :class:`GridSearchCV` with :class:`PipeGraphRegressor` in a simple linear workflow.
# :ref:`Next example <example3>` provides detail on how to proceed with a non linear case.

