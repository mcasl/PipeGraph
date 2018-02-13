
#from sklearn.datasets import load_boston

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from pipegraph.pipeGraph import PipeGraphRegressor
import matplotlib.pyplot as plt

##
# Ejemplo 1: sc + lm
X = np.random.rand(100,1)
y = 4 * X + 0.5*np.random.randn(100,1)
scaler = MinMaxScaler()
linear_model = LinearRegression()
steps = [('scaler', scaler), ('linear_model', linear_model)]
connections = {'scaler': { 'X': 'X'}, 'linear_model': {'X': ('scaler', 'predict'),  'y': 'y' }  }
pgraph = PipeGraphRegressor(steps=steps, connections=connections)
pgraph.fit(X, y)
y_pred = pgraph.predict(X)
plt.scatter(X, y)
plt.scatter(X, y_pred)
plt.show()
linear_model.coef_

##
# Ejemplo 2: sc + feature + lm
#   usando gridsearchcv

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
pgraph = PipeGraphRegressor(steps=steps, connections=connections)
grid_search_regressor  = GridSearchCV(estimator=pgraph, param_grid=param_grid, refit=True)
grid_search_regressor.fit(X, y)
y_pred = grid_search_regressor.predict(X)
plt.scatter(X, y)
plt.scatter(X, y_pred)
plt.show()
grid_search_regressor.best_estimator_.get_params()['linear_model'].coef_
grid_search_regressor.best_estimator_.get_params()['polynomial_features'].degree

##
# Ejemplo 3. con potencias
from sklearn.base import BaseEstimator


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


##
# Ejemplo 4. combinación de clasificadores

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from pipegraph.pipeGraph import PipeGraphClassifier, Concatenator
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

iris = load_iris()
X = iris.data
y = iris.target

scaler = MinMaxScaler()
gaussian_nb = GaussianNB()
svc = SVC()
mlp = MLPClassifier()
concatenator = Concatenator()

steps = [('scaler', scaler),
         ('gaussian_nb', gaussian_nb),
         ('svc', svc),
         ('concat', concatenator),
         ('mlp', mlp)]

connections = { 'scaler': {'X': 'X'},
                'gaussian_nb': {'X': ('scaler', 'predict'),
                                'y': 'y'},
                'svc':  {'X': ('scaler', 'predict'),
                         'y': 'y'},
                'concat': {'X1': ('scaler', 'predict'),
                           'X2': ('gaussian_nb', 'predict'),
                           'X3': ('svc', 'predict')},
                'mlp': {'X': ('concat', 'predict'),
                        'y': 'y'}
            }

param_grid = {'svc__C': [0.1, 0.5, 1.0],
              'mlp__hidden_layer_sizes': [(3,), (6,), (9,),],
              'mlp__max_iter': [5000, 10000]}

pgraph = PipeGraphClassifier(steps=steps, connections=connections)
grid_search_classifier  = GridSearchCV(estimator=pgraph, param_grid=param_grid, refit=True)
grid_search_classifier.fit(X, y)
y_pred = grid_search_classifier.predict(X)
grid_search_regressor.best_estimator_.get_params()




