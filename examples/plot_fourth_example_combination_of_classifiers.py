"""
Fourth Example: Combination of classifiers
-------------------------------------------------

A set of classifiers is combined as input to a neural network. Additionally, the scaled inputs are injected as well to the neural network.
"""
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
grid_search_classifier.best_estimator_.get_params()


# Code for plotting the confusion matrix taken
#  from 'Python Data Science Handbook' by Jake VanderPlas
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()  # for plot styling

mat = confusion_matrix(y_pred, y)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');

plt.show()


