"""
Second example: GridSearchCV demonstration
---------------------------------------------

This example shows how to use GridSearchCv with PipeGraph to effectively fit the best model across a number of hyperparameters.
It is equivalent to use GridSearchCv with Pipeline.

Steps of the PipeGraph:

- scaler: implements MinMaxScaler() class
- polynomial_features: implements PolynomialFeatures() class
- linear_model: implements LinearRegression() class
"""
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
# Use PipeGraphRegressor when the result is a regression

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