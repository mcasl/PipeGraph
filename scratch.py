import pandas as pd
from pipeGraph import *
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal
from sklearn.cluster import DBSCAN
# from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import MinMaxScaler

from paella import *
from sklearn.linear_model import LinearRegression



URL = "https://raw.githubusercontent.com/mcasl/PAELLA/master/data/sin_60_percent_noise.csv"
data = pd.read_csv(URL, usecols=['V1', 'V2'])
X, y = data[['V1']], data[['V2']]
concatenator = CustomConcatenation()
gaussian_clustering = GaussianMixture(n_components=3)
dbscan = DBSCAN(eps=0.05)
mixer = CustomCombination()
paellaModel = CustomPaella(regressor=LinearRegression,
                           noise_label=None,
                           max_it=1,
                           regular_size=100,
                           minimum_size=30,
                           width_r=0.95,
                           n_neighbors=1,
                           power=10,
                           random_state=42)

linearModel = LinearRegression()

steps = [('Concatenate_Xy', concatenator),
              ('Gaussian_Mixture', gaussian_clustering),
              ('Dbscan', dbscan),
              ('Combine_Clustering', mixer),
              ('Paella', paellaModel),
              ('Regressor', linearModel),
              ]

connections = {
    'Concatenate_Xy': dict(df1='X',
                           df2='y'),

    'Gaussian_Mixture': dict(X=('Concatenate_Xy', 'predict')),

    'Dbscan': dict(X=('Concatenate_Xy', 'predict')),

    'Combine_Clustering': dict(
        dominant=('Dbscan', 'predict'),
        other=('Gaussian_Mixture', 'predict')),

    'Paella': dict(X='X', y='y', classification=('Combine_Clustering', 'predict')),

    'Regressor': dict(X='X', y='y', sample_weight=('Paella', 'predict'))
}


pgraph = PipeGraph(steps=steps,
                       connections=connections,
                       use_for_fit='all',
                       use_for_predict=['Regressor'])

pgraph.data = {('_External', 'X'): X,
               ('_External', 'y'): y,
               }
pgraph._fit('Concatenate_Xy')
pgraph._fit('Gaussian_Mixture')
pgraph._fit('Dbscan')
pgraph._fit('Combine_Clustering')
pgraph._fit('Paella')
