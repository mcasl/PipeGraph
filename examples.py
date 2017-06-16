############################################
#classification : Numeric vector with cluster indicators. The 0 is assigned to noise samples.
#dist           : Confidence level for mahalanobis ellipsoid to consider points per model from seed point.
#form           : Proposed formula for analysis. ie V3~V2+V1
#v.pred         : Predicted variable, (number of column).
#width          : Significancy (P[Z<=z]=width) for residuals analysis. Normal from -\infty to z.
#max.it         : Maximum number of iterations
#sigmas          : Optional Variance-Covariance matrices, such as from mclust library. In case that they are not accurately calculated from the samples, such as in the model based clustering, where they are calculated according to the relative belongings.
#%%

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm
import numpy as np
from scipy.spatial.distance import mahalanobis, cdist
import patsy
import scipy 
import paella
import random
from math import pi


random.seed(2017)

N = 1000
x = np.random.uniform(-pi, pi, N) 
y = np.sin(x) + np.random.normal(0,0.2, N)
data = pd.DataFrame({"A":x, "B":y})
plt.scatter(x=data["A"], y=data["B"] )


from sklearn.cluster import KMeans
myKMeans = KMeans(n_clusters=3)
myKMeans.fit(data)
classification = pd.DataFrame({"Classification":myKMeans.labels_}, index=data.index)

plt.scatter(x=data["A"], y=data["B"], c=classification);


paellaRun = Paella(formula="B ~ A", data=data, classification=classification, max_it=1000)
paellaRun.fit()
umbral = 0.995
plt.scatter(x=data["A"], y=data["B"], c= (paellaRun.occurrences/paellaRun.iterations ) > umbral)


plt.scatter(x=data["A"], y=data["B"] )


plt.scatter(x=data["A"], y=data["B"] , c=classification, marker=".")
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"], marker="+")
plt.scatter(x=data.loc[seed_point,"A"], y=data.loc[seed_point,"B"],s=500, c="black", marker="x")


plt.scatter(x=data.loc[G_i_rows,"A"], y=data.loc[G_i_rows,"B"],s=100, c="pink")
plt.scatter(x=data.loc[seed_point,"A"], y=data.loc[seed_point,"B"],s=500, c="black", marker="x")


plt.scatter(x=data.loc[seed_point,"A"], y=data.loc[seed_point,"B"],s=100, c="red", picker=5)
plt.scatter(x=data.loc[rows,"A"],       y=data.loc[rows,"B"]      ,s=100, c="red")


plt.scatter(x=data.loc[G_i_rows, "A"], y=M_i_model.fittedvalues,s=100, c="green")


malas = np.abs(r_x_j) > r_threshold
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c="black" )
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c=malas)


plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c="red" )

#########
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"], marker="+", c="blue")
plt.scatter(x=data.loc[seed_point,"A"], y=data.loc[seed_point,"B"],s=500, c="red", marker="x")

plt.scatter(x=data.loc[rowIndexes,"A"],       y=data.loc[rowIndexes,"B"]      ,s=100, c="black")
plt.scatter(x=data.loc[G_i_rows,"A"], y=data.loc[G_i_rows,"B"],s=1, c="pink")
plt.scatter(x=data.loc[rows,"A"],       y=data.loc[rows,"B"]      ,s=1, c="red")
plt.scatter(x=data.loc[seed_point,"A"], y=data.loc[seed_point,"B"],s=500, c="pink", marker="x")


plt.scatter(x=data.loc[G_i_rows, "A"], y=M_i_model.fittedvalues,s=100, c="green")


malas = np.abs(r_x_j) > r_threshold
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c="black" )
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c=malas)


plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c="red" )
#####77
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"], c="blue", s=100)
plt.scatter(x=data.loc[seed_point,"A"], y=data.loc[seed_point,"B"],s=500, c="yellow", marker="x")

plt.scatter(x=data.loc[G_i_rows,"A"], y=data.loc[G_i_rows,"B"],s=10, c="pink")
plt.scatter(x=data.loc[rows,"A"],       y=data.loc[rows,"B"]      ,s=100, c="red")
plt.scatter(x=data.loc[seed_point,"A"], y=data.loc[seed_point,"B"],s=500, c="pink", marker="x")


plt.scatter(x=data.loc[G_i_rows, "A"], y=M_i_model.fittedvalues,s=100, c="green")


malas = np.abs(r_x_j) > r_threshold
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c="black" )
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c=malas)


plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c="red" )

#############################3
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"], marker="+", c="blue")
plt.scatter(x=data.loc[seed_point,"A"], y=data.loc[seed_point,"B"],s=500, c="red", marker="x")

plt.scatter(x=data.loc[rowIndexes,"A"],       y=data.loc[rowIndexes,"B"]      ,s=100, c="black")
plt.scatter(x=data.loc[G_i_rows,"A"], y=data.loc[G_i_rows,"B"],s=1, c="pink")
plt.scatter(x=data.loc[rows,"A"],       y=data.loc[rows,"B"]      ,s=1, c="red")
plt.scatter(x=data.loc[seed_point,"A"], y=data.loc[seed_point,"B"],s=500, c="pink", marker="x")


plt.scatter(x=data.loc[G_i_rows, "A"], y=M_i_model.fittedvalues,s=100, c="green")


malas = np.abs(r_x_j) > r_threshold
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c="black" )
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c=malas)


plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c="red" )
##### PHASE 2
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c=np.abs(r_x_j), cmap="Greys" )

##### TOTAL
plt.scatter(x=data["A"], y=data["B"],s=100, c=paellaRun.occurrences, cmap="cool" )

