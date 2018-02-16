"""
First example: A simple linear workflow
-----------------------------------------

Let's start with a simple example that could be perfectly expressed using a Pipeline. The data is transformed using a MinMaxScaler step and the preprocessed data is fed to a linear model.

Steps of the PipeGraph:

- scaler: implements MinMaxScaler() class
- linear_model: implements LinearRegression() class

.. image:: https://raw.githubusercontent.com/mcasl/PipeGraph/master/examples/images/Diapositiva1.png
"""



import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from pipegraph import PipeGraphRegressor

X = np.random.rand(100, 1)
y = 4 * X + 0.5*np.random.randn(100, 1)

scaler = MinMaxScaler()
linear_model = LinearRegression()


###############################################################################
# First we define the steps of the PipeGraph as a dictionary: we assign labels as the keys and sklearn classes as the
# values to each entry of the dictionary.

steps = [('scaler', scaler),
        ('linear_model', linear_model)]

###############################################################################
# Then, we need to provide the connections amongst steps as a dictionary. The keys of the top level entries of the
# dictionary are the labels of the steps and the values are dictionaries themselves. The keys of the latter dictionary
# are the names of the input variables of the current step and the values are either external variables to the PipeGraphRegressor object
# or the output variables taken from previous steps, which can be written as either:
#
# - A string representing a variable from an external source to the PipeGraphRegressor object, (can we say???? A string representing an input variable of the fit, predict or fit_predict methods of the PipeGraph??? - también arriba)
# - A tuple with the label of the step in position 0 followed by the name of the output variable in position 1.
# i.e. they represent the values passed to fit and predict methods.
#
# For instance, the linear model accepts as input `X` the output named 'predict' from the 'scaler' step,
# and as input 'y' the value of 'y' passed as an input variable by fit, predict or fit_predict methods of the PipeGraph.

connections = {'scaler': { 'X': 'X'},
                'linear_model': {'X': ('scaler', 'predict'),
                                'y': 'y'}}

###############################################################################
# Here we show how to construct the PipeGraphRegressor with the steps and connections previously defined and how to fit
# and predict data. Use PipeGraphRegressor when the result is a regression

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




