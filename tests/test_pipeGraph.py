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
                {'step': FirstStep,
                 'sklearn_class': None,
                 'kargs': None,
                 'input': {'X': x,
                           'y': y},
                 'use_for': ['fit', 'run'],
                 },

            'Gaussian_Mixture':
                {'step': ConcatenateInputStep,
                 'sklearn_class': GaussianMixture,
                 'kargs': {'n_components': 3},
                 'input': [('First', 'X'),
                           ('First', 'y')],
                 'use_for': ['fit'],
                 },

            'Dbscan':
                {'step': CustomDBSCANStep,
                 'sklearn_class': DBSCAN,
                 'kargs': {'eps': 0.05},
                 'input': [('First', 'X'),
                           ('First', 'y')],
                 'use_for': ['fit'],
                 },

            'Combine_Clustering':
                {'step': CustomCombinationStep,
                 'sklearn_class': None,
                 'kargs': None,
                 'input': [('Dbscan', 'output'),
                           ('Gaussian_Mixture', 'output')],
                 'use_for': ['fit'],
                 },

            'Paella':
                {'step': PaellaStep,
                 'sklearn_class': Paella,
                 'kargs': {'noise_label': -1,
                           'max_it': 20,
                           'regular_size': 400,
                           'minimum_size': 100,
                           'width_r': 0.99,
                           'n_neighbors': 5,
                           'power': 30,
                           'random_state': None},

                 'input': [('First', 'X'),
                           ('First', 'y'),
                           ('Combine_Clustering', 'output')],
                 'use_for': ['fit'],
                 },

            'Regressor':
                {'step': RegressorStep,
                 'sklearn_class': LinearRegression,
                 'kargs': {},
                 'input': [('First', 'X'),
                           ('First', 'y'),
                           ('Paella', 'sample_weight')],
                 'use_for': ['fit', 'run'],
                 },

            'Last':
                {'step': LastStep,
                 'sklearn_class': None,
                 'kargs': {},
                 'input': [('Regressor', 'prediction')],
                 'use_for': ['fit', 'run'],
                 },

        }

        self.graph = PipeGraph(self.graph_description)

    def test_description_attribute(self):
        self.assertEqual(self.graph_description, self.graph.description)

    def test_data_attribute(self):
        self.assertEqual(dict(), self.graph.data)

    def test_nodes(self):
        node_list = list(self.graph.nodes)
        self.assertEqual(node_list, ['First',
                                     'Gaussian_Mixture',
                                     'Dbscan',
                                     'Combine_Clustering',
                                     'Paella',
                                     'Regressor',
                                     'Last'])

    def test_iter(self):
        node_list = list(self.graph)
        self.assertEqual(node_list, ['First',
                                     'Gaussian_Mixture',
                                     'Dbscan',
                                     'Combine_Clustering',
                                     'Paella',
                                     'Regressor',
                                     'Last'])

    def test_get_fit_nodes(self):
        node_list = [step.node_name for step in self.graph.get_fit_steps()]
        self.assertEqual(node_list, ['First',
                                     'Dbscan',
                                     'Gaussian_Mixture',
                                     'Combine_Clustering',
                                     'Paella',
                                     'Regressor',
                                     'Last'])
    def test_get_run_nodes(self):
        node_list = [step.node_name for step in self.graph.get_run_steps()]
        self.assertEqual(node_list, ['First',
                                     'Regressor',
                                     'Last'])

    def test_graph_fit(self):
        self.graph.fit()
        assert_frame_equal(self.graph.data['First', 'X'], self.X)
        assert_frame_equal(self.graph.data['First', 'y'], self.y)

        self.assertEqual(self.graph.data['Dbscan', 'output'].shape[0], self.y.shape[0])
        self.assertEqual(self.graph.data['Dbscan', 'output'].min(), -1)

        self.assertEqual(self.graph.data['Gaussian_Mixture', 'output'].shape[0], self.y.shape[0])
        self.assertEqual(self.graph.data['Gaussian_Mixture', 'output'].min(), 0)


        self.assertEqual(self.graph.data['Combine_Clustering', 'output'].shape[0], self.y.shape[0])
        self.assertEqual(self.graph.data['Combine_Clustering', 'output'].min(), -1)
        self.assertEqual(self.graph.data['Combine_Clustering', 'output'].max(),
                         self.graph.data['Gaussian_Mixture', 'output'].max())

        self.assertEqual(self.graph.data['Paella', 'sample_weight'].shape[0], self.y.shape[0])
        self.assertEqual(self.graph.data['Paella', 'sample_weight'].min(), 0)
        self.assertEqual(self.graph.data['Paella', 'sample_weight'].max(), 1)

        self.assertEqual(self.graph.data['Regressor', 'prediction'].shape[0], self.y.shape[0])

        assert_array_equal(self.graph.data['Last'     , 'prediction'],
                         self.graph.data['Regressor', 'prediction'])

    def test_graph_run(self):
        self.graph.fit()
        self.graph.run()

        assert_frame_equal(self.graph.data['First', 'X'], self.X)
        assert_frame_equal(self.graph.data['First', 'y'], self.y)

        self.assertEqual(self.graph.data['Regressor', 'prediction'].shape[0], self.y.shape[0])

        assert_array_equal(self.graph.data['Last'     , 'prediction'],
                         self.graph.data['Regressor', 'prediction'])


if __name__ == '__main__':
    unittest.main()
