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
%matplotlib inline
import pandas as pd
import numpy as np
import random
from math import pi


%cd gitlabProjects/Paella
import paella 

random.seed(2017)

N = 500
x1 = np.random.uniform(-pi, pi, N) 
y1 = np.sin(x1) + np.random.normal(0,0.1, N)
data1 = pd.DataFrame({"A":x1, "B":y1}, index=pd.RangeIndex(N))

data = data1
plt.scatter(x=data["A"], y=data["B"] )


from sklearn.cluster import KMeans
myKMeans = KMeans(n_clusters=3)
myKMeans.fit(data)
classification1 = pd.DataFrame({"Classification":myKMeans.labels_}, index=data.index)

plt.scatter(x=data["A"], y=data["B"], c=classification1);

x2 = np.random.uniform(x1.min(), x1.max(), N) 
y2 = np.random.uniform(y1.min(), y1.max(), N) 
data2 = pd.DataFrame({"A":x2, "B":y2}, index=pd.RangeIndex(N)+N)
classification2 = pd.DataFrame({"Classification":4}, index=data2.index)

data = pd.concat([data1, data2],axis=0)
classification =  pd.concat([classification1, classification2],axis=0)
plt.scatter(x=data["A"], y=data["B"], c=classification, cmap="Set1");


paellaRun = paella.Paella(formula="B ~ A",
                          data=data,
                          classification=classification, 
                          noise_label=4,
                          max_it=10000,
                          beta_rate = 0.1, 
                          betaMin_rate = 0.05, 
                          width_r = 0.6,
                          width_alpha = 0.6)
        

paellaRun.fit()
umbral = 1
 plt.scatter(x=data["A"], y=data["B"], c= (1-paellaRun.occurrences/paellaRun.iterations ) >= umbral);


##### TOTAL
plt.scatter(x=data["A"], y=data["B"],s=100, c=paellaRun.occurrences, cmap="cool" )

plt.scatter(x=data.index, y=paellaRun.occurrences )

