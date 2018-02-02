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
    'First': {'X': x,
              'y': y},

    'Concatenate_Xy': dict(
        df1=('First', 'X'),
        df2=('First', 'y')),

    'Gaussian_Mixture': dict(
        X=('Concatenate_Xy', 'Xy')),

    'Dbscan': dict(
        X=('Concatenate_Xy', 'Xy')),

    'Combine_Clustering': dict(
        dominant=('Dbscan', 'prediction'),
        other=('Gaussian_Mixture', 'prediction')),

    'Paella': dict(
        X=('First', 'X'),
        y=('First', 'y'),
        classification=('Combine_Clustering', 'classification')),

    'Regressor': dict(
        X=('First', 'X'),
        y=('First', 'y'),
        sample_weight=('Paella', 'prediction')
    )
}

graph = PipeGraph(steps=steps,
                       connections=connections,
                       use_for_fit='all',
                       use_for_predict=['Regressor'])
