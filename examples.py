############################################
#classification : Numeric vector with cluster indicators. The 0 is assigned to noise samples.
#dist           : Confidence level for mahalanobis ellipsoid to consider points per model from seed point.
#form           : Proposed formula for analysis. ie V3~V2+V1
#v.pred         : Predicted variable, (number of column).
#width          : Significancy (P[Z<=z]=width) for residuals analysis. Normal from -\infty to z.
#max.it         : Maximum number of iterations
#sigmas          : Optional Variance-Covariance matrices, such as from mclust library. In case that they are not accurately calculated from the samples, such as in the model based clustering, where they are calculated according to the relative belongings.
#%%

import pandas as pd
import statsmodels.formula.api as sm
import numpy as np
from scipy.spatial.distance import mahalanobis, cdist
import patsy
import scipy 
import paella
import random
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs, make_s_curve
from sklearn import manifold

random.seed(2017)
data, classification = make_s_curve(n_samples=1000, noise=0.0, random_state=0)
data = pd.DataFrame(data, columns= ["A","B","C"])



mds = manifold.MDS(n_components=2, max_iter=100, n_init=1)
Y = mds.fit_transform(data)
data = pd.DataFrame(Y)
data.columns = ["A","B"]
C1 = data.query("A>0 & B>0")
C2 = data.query("A>0 & B<0")

data = pd.concat([C1, C2], axis=0)
classification = data.B > -0.5
classification = pd.DataFrame({"Classification":classification}, index=data.index)


paellaRun = Paella(formula="B ~ A", data=data, classification=classification, max_it=1000)
paellaRun.fit()
umbral = 0.99
plt.scatter(x=data["A"], y=data["B"], c= (paellaRun.occurrences/paellaRun.iterations ) > umbral)


plt.scatter(x=data["A"], y=data["B"] )





plt.scatter(x=data["A"], y=data["B"] , c=classification)

plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"])

plt.scatter(x=data.loc[seed_point,"A"], y=data.loc[seed_point,"B"],s=100, c="pink")

plt.scatter(x=data.loc[G_i_rows,"A"], y=data.loc[G_i_rows,"B"],s=100, c="pink")
plt.scatter(x=data.loc[seed_point,"A"], y=data.loc[seed_point,"B"],s=100, c="red", picker=5)
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c="red")

M_i_model
plt.scatter(x=data.loc[G_i_rows, "A"], y=M_i_model.fittedvalues,s=100, c="green")
malas = np.abs(r_x_j) > r_threshold
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c=malas)
plt.scatter(x=data.loc[rows,"A"], y=data.loc[rows,"B"],s=100, c="black" )

plt.scatter(x=data.loc[:,"A"], y=data.loc[:,"B"],s=100, c=np.abs(r_x_j) > alpha_threshold )

malas = np.abs(r_x_j) > alpha_threshold
plt.scatter(x=data.loc[:, "A"], y=data.loc[:,"B"],s=100, c=malas)


#

def onpick(event):
    origin = data.index[event.ind[0]]
    plt.gca().set_title('Selected item came from {}'.format(origin))

# tell mpl_connect we want to pass a 'pick_event' into onpick when the event is detected
plt.gcf().canvas.mpl_connect('pick_event', onpick)
