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


from sklearn.datasets.samples_generator import make_blobs
data, classification = make_blobs(n_samples=100, centers=3, cluster_std=1.0, n_features=2, random_state=0)
data = pd.DataFrame(data, columns= ["A","B"])
classification = pd.DataFrame({"Classification":classification}, index=data.index)

paellaRun = Paella(formula="B ~ A", data=data, classification=classification, max_it=100)
paellaRun.fit()
umbral = 0.8
plt.scatter(x=data["A"], y=data["B"], c= ( 1- (paellaRun.occurrences / paellaRun.iterations) ) > umbral)


