import unittest
import pandas as pd
from pandas.util.testing import assert_frame_equal
from numpy.testing import assert_array_equal

from pipeGraph import *
from paella import *

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class TestPipeGraphCase(unittest.TestCase):
    def setUp(self):
        # URL = "https://raw.githubusercontent.com/mcasl/PAELLA/master/data/sin_60_percent_noise.csv"
        URL = "sin_60_percent_noise.csv"
        data = pd.read_csv(URL, usecols=['V1', 'V2'])
        x, y = data[['V1']], data[['V2']]
        self.X = x
        self.y = y

        self.graph_description = {
            'First':
                {'step': InputStep,
                 'connections': {'X': x,
                                 'y': y},
                 'use_for': ['fit', 'run'],
                 },

            'Concatenate_Xy':
                {'step': CustomConcatenationStep,
                 'connections': {'df1': ('First', 'X'),
                                 'df2': ('First', 'y')},
                 'use_for': ['fit'],
                 },

            'Gaussian_Mixture':
                {'step': GaussianMixture,
                 'kargs': {'n_components': 3},
                 'connections': {'X': ('Concatenate_Xy', 'Xy')},
                 'use_for': ['fit'],
                 },

            'Dbscan':
                {'step': DBSCAN,
                 'kargs': {'eps': 0.05},
                 'connections': {'X': ('Concatenate_Xy', 'Xy')},
                 'use_for': ['fit'],
                 },

            'Combine_Clustering':
                {'step': CustomCombinationStep,
                 'connections': {'dominant': ('Dbscan', 'prediction'),
                                 'other': ('Gaussian_Mixture', 'prediction')},
                 'use_for': ['fit'],
                 },

            'Paella':
                {'step': CustomPaellaStep,
                 'kargs': {'noise_label': -1,
                           'max_it': 20,
                           'regular_size': 400,
                           'minimum_size': 100,
                           'width_r': 0.99,
                           'n_neighbors': 5,
                           'power': 30,
                           'random_state': None},

                 'connections': {'X': ('First', 'X'),
                                 'y': ('First', 'y'),
                                 'classification': ('Combine_Clustering', 'classification')},
                 'use_for': ['fit'],
                 },

            'Regressor':
                {'step': LinearRegression,
                 'kargs': {'copy_X': True, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False},
                 'connections': {'X': ('First', 'X'),
                                 'y': ('First', 'y'),
                                 'sample_weight': ('Paella', 'prediction')},
                 'use_for': ['fit', 'run'],
                 },

            'Last':
                {'step': OutputStep,
                 'connections': {'prediction': ('Regressor', 'prediction'),
                                 },
                 'use_for': ['fit', 'run'],
                 },
        }

        self.graph = PipeGraph(self.graph_description)

    def test_description_attribute(self):
        self.assertEqual(self.graph_description, self.graph.description)

    def test_data_attribute(self):
        self.assertEqual(dict(), self.graph.data)

    def test_node_names(self):
        node_list = list(self.graph.graph.nodes)
        self.assertEqual(node_list, ['First',
                                     'Concatenate_Xy',
                                     'Gaussian_Mixture',
                                     'Dbscan',
                                     'Combine_Clustering',
                                     'Paella',
                                     'Regressor',
                                     'Last'])

    def test_graph_fit(self):
        self.graph.fit()
        assert_frame_equal(self.graph.data['First', 'X'], self.X)
        assert_frame_equal(self.graph.data['First', 'y'], self.y)

        self.assertEqual(self.graph.data['Dbscan', 'prediction'].shape[0], self.y.shape[0])
        self.assertEqual(self.graph.data['Dbscan', 'prediction'].min(), -1)

        self.assertEqual(self.graph.data['Gaussian_Mixture', 'prediction'].shape[0], self.y.shape[0])
        self.assertEqual(self.graph.data['Gaussian_Mixture', 'prediction'].min(), 0)

        self.assertEqual(self.graph.data['Combine_Clustering', 'classification'].shape[0], self.y.shape[0])
        self.assertEqual(self.graph.data['Combine_Clustering', 'classification'].min(), -1)
        self.assertEqual(self.graph.data['Combine_Clustering', 'classification'].max(),
                         self.graph.data['Gaussian_Mixture', 'prediction'].max())

        self.assertEqual(self.graph.data['Paella', 'prediction'].shape[0], self.y.shape[0])
        self.assertEqual(self.graph.data['Paella', 'prediction'].min(), 0)
        self.assertEqual(self.graph.data['Paella', 'prediction'].max(), 1)

        self.assertEqual(self.graph.data['Regressor', 'prediction'].shape[0], self.y.shape[0])

        assert_array_equal(self.graph.data['Last', 'prediction'],
                           self.graph.data['Regressor', 'prediction'])

    def test_graph_predict(self):
        self.graph.fit()
        self.graph.predict()

        assert_frame_equal(self.graph.data['First', 'X'], self.X)
        assert_frame_equal(self.graph.data['First', 'y'], self.y)

        self.assertEqual(self.graph.data['Regressor', 'prediction'].shape[0], self.y.shape[0])

        assert_array_equal(self.graph.data['Last', 'prediction'],
                           self.graph.data['Regressor', 'prediction'])

    def test_step_get_params(self):
        graph = self.graph.graph
        self.assertEqual(graph.node['First']['step'].get_params(), {})
        self.assertEqual(graph.node['Concatenate_Xy']['step'].get_params(), {})
        self.assertEqual(graph.node['Gaussian_Mixture']['step'].get_params()['n_components'], 3)
        self.assertEqual(graph.node['Dbscan']['step'].get_params()['eps'], 0.05)
        self.assertEqual(graph.node['Combine_Clustering']['step'].get_params(), {})
        self.assertEqual(graph.node['Paella']['step'].get_params()['noise_label'], -1)
        self.assertEqual(graph.node['Paella']['step'].get_params()['max_it'], 20)
        self.assertEqual(graph.node['Paella']['step'].get_params()['regular_size'], 400)
        self.assertEqual(graph.node['Paella']['step'].get_params()['minimum_size'], 100)
        self.assertEqual(graph.node['Paella']['step'].get_params()['width_r'], 0.99)
        self.assertEqual(graph.node['Paella']['step'].get_params()['n_neighbors'], 5)
        self.assertEqual(graph.node['Paella']['step'].get_params()['power'], 30)
        self.assertEqual(graph.node['Regressor']['step'].get_params(), {'copy_X': True, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False})
        self.assertEqual(graph.node['Last']['step'].get_params(), {})

    def test_step_set_params(self):
        graph = self.graph.graph
        self.assertRaises(NameError, graph.node['First']['step'].set_params, ham=902)
        self.assertRaises(NameError, graph.node['Concatenate_Xy']['step'].set_params, ham=2, spam=9)
        self.assertRaises(NameError, graph.node['Last']['step'].set_params, ham=92)
        self.assertEqual(graph.node['Gaussian_Mixture']['step'].set_params(n_components=5).get_params()['n_components'],
                         5)
        self.assertEqual(graph.node['Dbscan']['step'].set_params(eps=10.2).get_params()['eps'],
                         10.2)
        self.assertEqual(graph.node['Paella']['step'].set_params(noise_label=-3).get_params()['noise_label'],
                         -3)
        self.assertEqual(graph.node['Regressor']['step'].set_params(copy_X=False).get_params(),
                         {'copy_X': False, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False})
        self.assertEqual(graph.node['Regressor']['step'].set_params(n_jobs=13).get_params(),
                         {'copy_X': False, 'fit_intercept': True, 'n_jobs': 13, 'normalize': False})


if __name__ == '__main__':
    unittest.main()
