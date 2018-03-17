"""
.. _example10:

Tenth Example: # Alternative solution to example number 9
------------------------------------------------------------------------------------

We continue demonstrating several interesting features:
 1. How the user can choose to encapsulate several blocks into a PipeGraph and use it as a single unit in another PipeGraph
 2. How these components can be dynamically built on runtime depending on initialization parameters
 3. How these components can be dynamically built on runtime depending on input signal values during fit
 4. Using GridSearchCV to explore the best combination of hyperparameters

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from pipegraph.base import ( PipeGraphRegressor,
                             ClassifierAndRegressorsBundle,
                             NeutralRegressor
                            )

X_first = pd.Series(np.random.rand(1000,))
y_first = pd.Series(4 * X_first + 0.5*np.random.randn(1000,))
X_second = pd.Series(np.random.rand(1000,) + 3)
y_second = pd.Series(-4 * X_second + 0.5*np.random.randn(1000,))
X_third = pd.Series(np.random.rand(1000,) + 6)
y_third = pd.Series(2 * X_third + 0.5*np.random.randn(1000,))

X = pd.concat([X_first, X_second, X_third], axis=0).to_frame()
y = pd.concat([y_first, y_second, y_third], axis=0).to_frame()

X_train, X_test, y_train, y_test = train_test_split(X, y)

#######################################################################################################
# Now we consider an alternative solution to example number 9. The previous solution showed the potential
# of being able to morph the graph during fitting. A simpler approach is considered in this example by reusing
# components and combining the classifier with the demultiplexed models:
import inspect
print(inspect.getsource(ClassifierAndRegressorsBundle))

################################################################################################
# Using this new component we can build a simplified PipeGraph:

scaler = MinMaxScaler()
classifier_and_models = ClassifierAndRegressorsBundle(number_of_replicas=6)
neutral_regressor = NeutralRegressor()

steps = [('scaler', scaler),
         ('bundle', classifier_and_models),
         ('neutral', neutral_regressor)]

connections = {'scaler': {'X': 'X'},
               'bundle': {'X': 'scaler', 'y': 'y'},
               'neutral': {'X': 'bundle'}}

pgraph = PipeGraphRegressor(steps=steps, fit_connections=connections)


##############################################################################################################
# Using GridSearchCV to find the best number of clusters and the best regressors

from sklearn.model_selection import GridSearchCV

param_grid = {'bundle__number_of_replicas': range(3,10)}
gs = GridSearchCV(estimator=pgraph, param_grid=param_grid, refit=True)
gs.fit(X_train, y_train)
y_pred = gs.predict(X_train)
plt.scatter(X_train, y_train)
plt.scatter(X_train, y_pred)
print("Score:" , gs.score(X_test, y_test))
print("bundle__number_of_replicas:", gs.best_estimator_.get_params()['bundle__number_of_replicas'])


