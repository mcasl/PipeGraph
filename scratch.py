import pandas as pd
from importlib import reload
import pipeGraph
reload(pipeGraph)
from pipeGraph import *
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal
from sklearn.cluster import DBSCAN
# from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import MinMaxScaler

from paella import *
from sklearn.linear_model import LinearRegression
#
#
#
# URL = "https://raw.githubusercontent.com/mcasl/PAELLA/master/data/sin_60_percent_noise.csv"
# data = pd.read_csv(URL, usecols=['V1', 'V2'])
# X, y = data[['V1']], data[['V2']]
# concatenator = CustomConcatenation()
# gaussian_clustering = GaussianMixture(n_components=3)
# dbscan = DBSCAN(eps=0.05)
# mixer = CustomCombination()
# paellaModel = CustomPaella(regressor=LinearRegression,
#                            noise_label=None,
#                            max_it=1,
#                            regular_size=100,
#                            minimum_size=30,
#                            width_r=0.95,
#                            n_neighbors=1,
#                            power=10,
#                            random_state=42)
#
# linearModel = LinearRegression()
#
# steps = [('Concatenate_Xy', concatenator),
#               ('Gaussian_Mixture', gaussian_clustering),
#               ('Dbscan', dbscan),
#               ('Combine_Clustering', mixer),
#               ('Paella', paellaModel),
#               ('Regressor', linearModel),
#               ]
#
# connections = {
#     'Concatenate_Xy': dict(df1='X',
#                            df2='y'),
#
#     'Gaussian_Mixture': dict(X=('Concatenate_Xy', 'predict')),
#
#     'Dbscan': dict(X=('Concatenate_Xy', 'predict')),
#
#     'Combine_Clustering': dict(
#         dominant=('Dbscan', 'predict'),
#         other=('Gaussian_Mixture', 'predict')),
#
#     'Paella': dict(X='X', y='y', classification=('Combine_Clustering', 'predict')),
#
#     'Regressor': dict(X='X', y='y', sample_weight=('Paella', 'predict'))
# }
#
#
# pgraph = PipeGraph(steps=steps,
#                        connections=connections,
#                        use_for_fit='all',
#                        use_for_predict=['Regressor'])
#
# pgraph.data = {('_External', 'X'): X,
#                ('_External', 'y'): y,
#                }
# pgraph._fit('Concatenate_Xy')
# pgraph._fit('Gaussian_Mixture')
# pgraph._fit('Dbscan')
# pgraph._fit('Combine_Clustering')
# pgraph._fit('Paella')

URL = "https://raw.githubusercontent.com/mcasl/PAELLA/master/data/sin_60_percent_noise.csv"
data = pd.read_csv(URL, usecols=['V1', 'V2'])
X = X = data[['V1']]
y = y = data[['V2']]
lm = LinearRegression()
steps = [('linear_model', lm)]
connections = {'linear_model': dict(X='X', y='y')}
pgraph = PipeGraph(steps=steps,
                        connections=connections,
                        use_for_fit='all',
                        use_for_predict='all')
param_grid = dict(linear_model__fit_intercept=[False, True],
                  linear_model__normalize=[True, False],
                       )
from sklearn.pipeline import Pipeline
pipa = Pipeline([('linear_model', lm)])

pipa.fit(X,y)

from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(pipa, param_grid=param_grid, return_train_score=True)
gs.fit(X, y)
gs.cv_results_
pgraph.get_params()


gp = GridSearchCV(pgraph, param_grid=param_grid, return_train_score=True, scoring=lm.score)
gp.fit(X, y)
gp.predict(X)


pgraph.fit(X, y)
pgraph.predict(X)
pgraph.score()
lm.score??



X = pd.DataFrame(dict(X=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
y = pd.DataFrame(dict(X=[0, 2.2, 3.9, 6.1, 7.9, 9.9, 12.2, 13.8, 16.2, 17.5, 20.4]))
lm = LinearRegression()
steps = [('linear_model', lm)]
connections = {'linear_model': dict(X='X', y='y')}
pgraph = PipeGraph(steps=steps,
                        connections=connections,
                        use_for_fit='all',
                        use_for_predict='all')
param_grid = dict(linear_model__fit_intercept=[False, True],
                       linear_model__normalize=[True, False],
                       )


pgraph.fit(X=X, y=y)
pgraph.predict(X=X)
pd.concat([X, pd.DataFrame(pgraph.predict(X=X) )], axis=1)
pgraph.steps[0][1].coef_
pgraph.r2_score(X,y)

gp = GridSearchCV(pgraph, param_grid=param_grid, return_train_score=True, scoring=pgraph.r2_score)
gp.fit(X, y)
gp.predict(X)
