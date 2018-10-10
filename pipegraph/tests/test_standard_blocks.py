# -*- coding: utf-8 -*-
# The MIT License (MIT)
#
# Copyright (c) 2018 Laura Fernandez Robles,
#                    Hector Alaiz Moreton,
#                    Jaime Cifuentes-Rodriguez,
#                    Javier Alfonso-Cendón,
#                    Camino Fernández-Llamas,
#                    Manuel Castejón-Limas
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import unittest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

from pipegraph.base import (PipeGraph,
                            add_mixins_to_step,
                            Concatenator, ColumnSelector, RegressorsWithParametrizedNumberOfReplicas,
                            RegressorsWithDataDependentNumberOfReplicas,
                            NeutralRegressor)


from pipegraph.demo_blocks import (CustomCombination,
                                   )

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


class TestModelsWithParameterizedNumberOfReplicas(unittest.TestCase):
    def setUp(self):
        X_first = pd.Series(np.random.rand(100, ))
        y_first = pd.Series(4 * X_first + 0.5 * np.random.randn(100, ))

        X_second = pd.Series(np.random.rand(100, ) + 3)
        y_second = pd.Series(-4 * X_second + 0.5 * np.random.randn(100, ))

        X_third = pd.Series(np.random.rand(100, ) + 6)
        y_third = pd.Series(2 * X_third + 0.5 * np.random.randn(100, ))

        self.X = pd.concat([X_first, X_second, X_third], axis=0).to_frame()
        self.y = pd.concat([y_first, y_second, y_third], axis=0).to_frame()

        scaler = MinMaxScaler()
        gaussian_mixture = GaussianMixture(n_components=3)
        models = RegressorsWithParametrizedNumberOfReplicas(number_of_replicas=3, regressor=LinearRegression())
        neutral_regressor = NeutralRegressor()

        steps = [('scaler', scaler),
                 ('classifier', gaussian_mixture),
                 ('models', models),
                 ('neutral', neutral_regressor)]

        connections = {'scaler': {'X': 'X'},
                       'classifier': {'X': 'scaler'},
                       'models': {'X': 'scaler',
                                  'y': 'y',
                                  'selection': 'classifier'},
                       'neutral': {'X': 'models'}
                       }
        self.pgraph = PipeGraph(steps=steps, fit_connections=connections)

    def test_ModelsWithParameterizedNumberOfReplicas__connections(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph

        self.assertTrue(isinstance(pgraph.named_steps['models'], RegressorsWithParametrizedNumberOfReplicas))
        result_connections = pgraph.named_steps['models'].fit_connections
        expected_connections = {'demux': {'X': 'X', 'selection': 'selection', 'y': 'y'},
                                'regressor_0': {'X': ('demux', 'X_0'), 'y': ('demux', 'y_0')},
                                'regressor_1': {'X': ('demux', 'X_1'), 'y': ('demux', 'y_1')},
                                'regressor_2': {'X': ('demux', 'X_2'), 'y': ('demux', 'y_2')},
                                'mux': {'0': 'regressor_0',
                                        '1': 'regressor_1',
                                        '2': 'regressor_2',
                                        'selection': 'selection'}}
        self.assertEqual(result_connections, expected_connections)
        result_steps = sorted(list(pgraph.named_steps.keys()))
        expected_steps = sorted(['scaler', 'classifier', 'models', 'neutral'])
        self.assertEqual(result_steps, expected_steps)

    def test_ModelsWithParameterizedNumberOfReplicas__predict(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph

        pgraph.fit(X, y)
        y_pred = pgraph.predict(X)
        self.assertEqual(y_pred.shape[0], y.shape[0])

    def test_ModelsWithParameterizedNumberOfReplicas__score(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph

        pgraph.fit(X, y)
        result = pgraph.score(X, y)
        self.assertTrue(result > -42 )


class TestModelsWithDataDependentNumberOfReplicas(unittest.TestCase):
    def setUp(self):
        X_first = pd.Series(np.random.rand(1000, ))
        y_first = pd.Series(4 * X_first + 0.5 * np.random.randn(1000, ))

        X_second = pd.Series(np.random.rand(1000, ) + 3)
        y_second = pd.Series(-4 * X_second + 0.5 * np.random.randn(1000, ))

        X_third = pd.Series(np.random.rand(1000, ) + 6)
        y_third = pd.Series(2 * X_third + 0.5 * np.random.randn(1000, ))

        self.X = pd.concat([X_first, X_second, X_third], axis=0).to_frame()
        self.y = pd.concat([y_first, y_second, y_third], axis=0).to_frame()
        scaler = MinMaxScaler()
        gaussian_mixture = GaussianMixture(n_components=3)
        models = RegressorsWithDataDependentNumberOfReplicas(steps=[('regressor', LinearRegression())])
        neutral_regressor = NeutralRegressor()

        steps = [('scaler', scaler),
                 ('classifier', gaussian_mixture),
                 ('models', models),
                 ('neutral', neutral_regressor)]

        connections = {'scaler': {'X': 'X'},
                       'classifier': {'X': 'scaler'},
                       'models': {'X': 'scaler',
                                  'y': 'y',
                                  'selection': 'classifier'},
                       'neutral': {'X': 'models'},
                       }

        self.pgraph = PipeGraph(steps=steps, fit_connections=connections)
        self.pgraph.fit(self.X, self.y)

    def test_ModelsWithDataDependentNumberOfReplicas__connections(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph

        pgraph.fit(X, y)
        y_pred = pgraph.predict(X)

        self.assertTrue(isinstance(pgraph.named_steps['models'], RegressorsWithDataDependentNumberOfReplicas))
        result_connections = pgraph.named_steps['models']._pipegraph.fit_connections
        expected_connections = {'regressorsBundle': {'X': 'X', 'selection': 'selection', 'y': 'y'}}
        self.assertEqual(result_connections, expected_connections)
        result_steps = sorted(list(pgraph.named_steps.keys()))
        expected_steps = sorted(['scaler', 'classifier', 'models', 'neutral'])
        self.assertEqual(result_steps, expected_steps)
        self.assertEqual(y_pred.shape[0], y.shape[0])

    def test_ModelsWithDataDependentNumberOfReplicas__predict(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph

        pgraph.fit(X, y)
        y_pred = pgraph.predict(X)
        self.assertEqual(y_pred.shape[0], y.shape[0])

    def test_ModelsWithDataDependentNumberOfReplicas__score(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph

        pgraph.fit(X, y)
        result = pgraph.score(X, y)
        self.assertTrue(result > -42 )

    def test_ModelsWithDataDependentNumberOfReplicas__GridSearchCV(self):
        X = self.X
        y = self.y

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        pgraph = self.pgraph
        param_grid = {'classifier__n_components': range(2, 10)}
        gs = GridSearchCV(estimator = pgraph, param_grid=param_grid, refit=True)
        gs.fit(X_train, y_train)
        result = gs.score(X_test, y_test)
        self.assertTrue(result > -42 )


class TestColumnSelector(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame.from_dict({'V1': [1, 2, 3, 4, 5, 6, 7, 8],
                                         'V2': [10, 20, 30, 40, 50, 60, 70, 80],
                                         'V3': [100, 200, 300, 400, 500, 600, 700, 800]})

    def test_ColumnSelector__mapping_is_None(self):
        X = self.X
        selector = ColumnSelector()
        self.assertTrue(selector.fit() is selector)
        assert_frame_equal(selector.predict(X)['predict'], X)

    def test_ColumnSelector__pick_one_column_first(self):
        X = self.X
        selector = ColumnSelector(mapping={'X': slice(0, 1)})
        self.assertTrue(selector.fit() is selector)
        assert_frame_equal(selector.predict(X)['X'], X.loc[:, ["V1"]])

    def test_ColumnSelector__pick_one_column_last(self):
        X = self.X
        selector = ColumnSelector(mapping={'y': slice(2, 3)})
        self.assertTrue(selector.fit() is selector)
        assert_frame_equal(selector.predict(X)['y'], X.loc[:, ["V3"]])

    def test_ColumnSelector__pick_two_columns(self):
        X = self.X
        selector = ColumnSelector(mapping={'X': slice(0, 2)})
        self.assertTrue(selector.fit() is selector)
        assert_frame_equal(selector.predict(X)['X'], X.loc[:, ["V1", "V2"]])

    def test_ColumnSelector__pick_three_columns(self):
        X = self.X
        selector = ColumnSelector(mapping={'X': slice(0, 3)})
        self.assertTrue(selector.fit() is selector)
        assert_frame_equal(selector.predict(X)['X'], X)


if __name__ == '__main__':
    unittest.main()
