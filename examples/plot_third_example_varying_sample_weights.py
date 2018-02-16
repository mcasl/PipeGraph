"""
Third Example: Injecting varying sample_weight vectors to a linear regression model for GridSearchCV
---------------------------------------------------
This example illustrates a case in which different vectors of sample_weight are injected to a linear regression model
in order to evaluate them and obtain the sample_weight that generate the best results. Let's imagine we have a
sample_weight vector and different powers of the vector are needed to be evaluated. To perform such experiment,
the following constrains appear:

- the shape of the graph is not a pipe
- More than two variables (typically: X and y) need to be accordingly split in order to perform the cross validation with GridSearchCV, in this case: X, y and sample_weight
- sample_weight of the LinearRegression function varies for different searchs of GridSearchCV. In a GridSearchCV with Pipeline, sample_weight can't vary because it is not an attribute of LinearRegression but an attribute of its fit method.

Steps of the PipeGraph:

- selector: implements ColumnSelector() class. X augmented data is column-wise divided as specified in a mapping dictionary. We create an augmented X in which all data but ``y`` is concatenated and it will be used by GridSearchCV to make the cross validation splits. selector step de-concatenates such data.
- custom_power: implements CustomPower() class. Input data is powered to a given power.
- scaler: implements MinMaxScaler() class
- polynomial_features: implements PolynomialFeatures() class
- linear_model: implements LinearRegression() class

"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from pipegraph.pipeGraph import PipeGraphRegressor, CustomPower, ColumnSelector, Reshape
import matplotlib.pyplot as plt

###############################################################################
# We create an augmented ``X`` in which all data but ``y`` is concatenated. In this case, we concatenate ``X`` and ``sample_weight`` vectors.

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

###############################################################################
# Use PipeGraphRegressor when the result is a regression

pgraph = PipeGraphRegressor(steps=steps, connections=connections)
grid_search_regressor = GridSearchCV(estimator=pgraph, param_grid=param_grid, refit=True)
grid_search_regressor.fit(X, y)
y_pred = grid_search_regressor.predict(X)

plt.scatter(X.loc[:,'X'], y)
plt.scatter(X.loc[:,'X'], y_pred)
plt.show()

power = grid_search_regressor.best_estimator_.get_params()['custom_power']
print('Power that obtains the best results in the linear model: \n {}'.format(power))
