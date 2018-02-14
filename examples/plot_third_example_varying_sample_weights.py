"""
Third Example: Injecting varying sample_weights
---------------------------------------------------

This example shows how to inject sample weight after arbitrary transformations.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from pipegraph.pipeGraph import PipeGraphRegressor, CustomPower, ColumnSelector, Reshape
import matplotlib.pyplot as plt


X = pd.DataFrame(dict(X=np.array([   1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11]),
          sample_weight=np.array([0.01, 0.95, 0.10, 0.95, 0.95, 0.10, 0.10, 0.95, 0.95, 0.95, 0.01])))
y = np.array(                    [  10,    4,   20,   16,   25 , -60,   85,   64,   81,  100,  150])
scaler = MinMaxScaler()
polynomial_features = PolynomialFeatures()
linear_model = LinearRegression()
custom_power = CustomPower()
selector = ColumnSelector(mapping={'X': slice(0, 1),
                                   'sample_weight': slice(1,2)})

steps = [('selector', selector),
         ('custom_power', custom_power),
         ('scaler', scaler),
         ('polynomial_features', polynomial_features),
         ('linear_model', linear_model)]

connections = { 'selector':     {'X':'X'},
                'custom_power': {'X': ('selector', 'sample_weight')},
                'scaler':       {'X': ('selector', 'X')},
                'polynomial_features': {'X': ('scaler', 'predict')},
                'linear_model': {'X': ('polynomial_features', 'predict'),
                                 'y': 'y',
                                 'sample_weight': ('custom_power', 'predict')}  }

param_grid = {'polynomial_features__degree': range(1, 3),
              'linear_model__fit_intercept': [True, False],
              'custom_power__power': [1, 5, 10, 20, 30]}

pgraph = PipeGraphRegressor(steps=steps, connections=connections)
grid_search_regressor  = GridSearchCV(estimator=pgraph, param_grid=param_grid, refit=True)
grid_search_regressor.fit(X, y)
y_pred = grid_search_regressor.predict(X)
plt.scatter(X.loc[:,'X'], y)
plt.scatter(X.loc[:,'X'], y_pred)
plt.show()
grid_search_regressor.best_estimator_.get_params()['custom_power']

