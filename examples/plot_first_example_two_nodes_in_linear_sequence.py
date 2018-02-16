"""
First example: A simple linear workflow
-----------------------------------------

Let's start with a simple example that could be perfectly expressed using a Pipeline. The data is transformed
using a MinMaxScaler step and the preprocessed data is fed to a linear model.

First, let's implement it using a Pipeline and then expressed as
an :class:`pipegraph.PipeGraphRegressor` object.

For that purpose we need to provide the connections amongst steps as a dictionary
whose top level entries are the steps labels and  whose values are dictionaries themselves expressing,
 for each node, the relationship between the output from a previous node and the input variables of the current node.

This is written as either:
- A tuple with the label of the step in position 0 followed by the name of the output variable in position 1.
- A string representing a variable from  a source external to the PipeGraphRegressor object,
i.e. they represent the values passed to fit and predict methods.

For instance, the linear model accepts as input `X` the output named 'predict' from the 'scaler' step,
and as input 'y' the value of 'y' passed by fit or predict methods.

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

steps = [('scaler', scaler),
         ('linear_model', linear_model)]

connections = {'scaler': { 'X': 'X'},
               'linear_model': {'X': ('scaler', 'predict'),
                                'y': 'y'}}


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




