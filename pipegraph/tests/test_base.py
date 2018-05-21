import logging
import unittest
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from pipegraph.paella import Paella
from pipegraph.base import (PipeGraphRegressor,
                            PipeGraphClassifier,
                            PipeGraph,
                            Process,
                            wrap_adaptee_in_process,
                            build_graph,
                            make_connections_when_not_provided_to_init,
                            Concatenator,
                            ColumnSelector,
                            Reshape,
                            Demultiplexer,
                            NeutralRegressor,
                            NeutralClassifier)

from pipegraph.adapters import  (AdapterForFitTransformAdaptee,
                                 AdapterForFitPredictAdaptee,
                                 AdapterForAtomicFitPredictAdaptee,
                                 AdapterForCustomFitPredictWithDictionaryOutputAdaptee)

from pipegraph.demo_blocks import CustomCombination

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)



class TestRootFunctions(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = np.random.rand(self.size, 1)
        self.y = self.X + np.random.randn(self.size, 1)
        concatenator = Concatenator()
        gaussian_clustering = GaussianMixture(n_components=3)
        dbscan = DBSCAN(eps=0.5)
        mixer = CustomCombination()
        paellaModel = Paella(regressor=LinearRegression,
                             noise_label=None,
                             max_it=10,
                             regular_size=100,
                             minimum_size=30,
                             width_r=0.95,
                             power=10,
                             random_state=42)
        linear_model = LinearRegression()
        self.steps = steps = [('Concatenate_Xy', concatenator),
                              ('Gaussian_Mixture', gaussian_clustering),
                              ('Dbscan', dbscan),
                              ('Combine_Clustering', mixer),
                              ('Paella', paellaModel),
                              ('Regressor', linear_model),
                              ]

        self.connections = {
            'Concatenate_Xy': dict(df1='X',
                                   df2='y'),

            'Gaussian_Mixture': dict(X=('Concatenate_Xy', 'Xy')),

            'Dbscan': dict(X=('Concatenate_Xy', 'Xy')),

            'Combine_Clustering': dict(
                dominant=('Dbscan', 'predict'),
                other=('Gaussian_Mixture', 'predict')),

            'Paella': dict(X='X', y='y', classification=('Combine_Clustering', 'predict')),

            'Regressor': dict(X='X', y='y', sample_weight=('Paella', 'predict'))
        }

    def test_wrap_adaptee_in_process__right_classes(self):
        tests_table = [
            {'model': LinearRegression(),
             'expected_class': AdapterForFitPredictAdaptee},

            {'model': MinMaxScaler(),
             'expected_class': AdapterForFitTransformAdaptee},

            {'model': DBSCAN(),
             'expected_class': AdapterForAtomicFitPredictAdaptee},

            {'model': Demultiplexer(),
             'expected_class': AdapterForCustomFitPredictWithDictionaryOutputAdaptee}
        ]

        for test_dict in tests_table:
            model = test_dict['model']
            step = wrap_adaptee_in_process(model)
            self.assertEqual(isinstance(step._strategy, test_dict['expected_class']), True)

    def test_wrap_adaptee_in_process__wrap_process_does_nothing(self):
            lm = LinearRegression()
            wrapped_lm = wrap_adaptee_in_process(lm)
            double_wrap = wrap_adaptee_in_process(wrapped_lm)
            self.assertEqual(double_wrap._strategy._adaptee , lm)

    def test_wrap_adaptee_in_process__raise_exception(self):
        model = object()
        self.assertRaises(ValueError, wrap_adaptee_in_process, model)

    def test_build_graph__connections_None(self):
        graph = build_graph(None)
        self.assertTrue( graph is None)

    def test_build_graph__node_names(self):
        graph = build_graph(self.connections)

        node_list = list(graph.nodes())
        self.assertEqual(sorted(node_list), sorted(['Concatenate_Xy',
                                                    'Gaussian_Mixture',
                                                    'Dbscan',
                                                    'Combine_Clustering',
                                                    'Paella',
                                                    'Regressor',
                                                    ]))

    def test_build_graph__node_edges(self):
        graph = build_graph(self.connections)

        in_edges = {name: [origin for origin, destination in list(graph.in_edges(name))]
                    for name in graph}
        logger.debug("in_edges: %s", in_edges)
        self.assertEqual(in_edges['Gaussian_Mixture'], ['Concatenate_Xy'])
        self.assertEqual(in_edges['Dbscan'], ['Concatenate_Xy'])
        self.assertEqual(sorted(in_edges['Combine_Clustering']), sorted(['Gaussian_Mixture', 'Dbscan']))
        self.assertEqual(in_edges['Paella'], ['Combine_Clustering'])
        self.assertEqual(in_edges['Regressor'], ['Paella'])


    def test_make_connections_when_not_provided_to_init__Many_steps(self):
        steps = self.steps
        connections = make_connections_when_not_provided_to_init(steps=steps)
        expected = dict(Concatenate_Xy={'X': 'X'},
                        Gaussian_Mixture={'X': ('Concatenate_Xy', 'predict')},
                        Dbscan={'X': ('Gaussian_Mixture', 'predict')},
                        Combine_Clustering={'X': ('Dbscan', 'predict')},
                        Paella={'X': ('Combine_Clustering', 'predict')},
                        Regressor={'X': ('Paella', 'predict'),
                                   'y':'y'}
                        )
        self.assertEqual(connections, expected)

class TestPipegraph(unittest.TestCase):
    def setUp(self):
        self.size = 1000
        self.X = pd.DataFrame(dict(X=np.random.rand(self.size, )))
        self.y = pd.DataFrame(dict(y=(np.random.rand(self.size, ))))
        concatenator = Concatenator()
        gaussian_clustering = GaussianMixture(n_components=3)
        dbscan = DBSCAN(eps=0.5)
        mixer = CustomCombination()
        paellaModel = Paella(regressor=LinearRegression,
                             noise_label=None,
                             max_it=10,
                             regular_size=100,
                             minimum_size=30,
                             width_r=0.95,
                             power=10,
                             random_state=42)
        linear_model = LinearRegression()
        steps = [('Concatenate_Xy', concatenator),
                 ('Gaussian_Mixture', gaussian_clustering),
                 ('Dbscan', dbscan),
                 ('Combine_Clustering', mixer),
                 ('Paella', paellaModel),
                 ('Regressor', linear_model),
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

        self.steps_external = [('_External', concatenator),
                               ('Gaussian_Mixture', gaussian_clustering),
                               ('Dbscan', dbscan),
                               ('Combine_Clustering', mixer),
                               ('Paella', paellaModel),
                               ('Regressor', linear_model),
                               ]

        self.connections_external = {
            '_External': dict(df1='X',
                              df2='y'),

            'Gaussian_Mixture': dict(X=('Concatenate_Xy', 'predict')),

            'Dbscan': dict(X=('Concatenate_Xy', 'predict')),

            'Combine_Clustering': dict(
                dominant=('Dbscan', 'predict'),
                other=('Gaussian_Mixture', 'predict')),

            'Paella': dict(X='X', y='y', classification=('Combine_Clustering', 'predict')),

            'Regressor': dict(X='X', y='y', sample_weight=('Paella', 'predict'))
        }

        self.steps = steps
        self.connections = connections
        self.pgraph = PipeGraph(steps=steps, fit_connections=connections)

    def test_Pipegraph__External_step_name(self):
        pgraph = PipeGraph(steps=self.steps_external, fit_connections=self.connections_external)
        self.assertRaises(ValueError, pgraph.fit, self.X, self.y)

    def test_Pipegraph__example_1_no_connections(self):
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.linear_model import LinearRegression
        from pipegraph import PipeGraphRegressor

        X = np.random.rand(100, 1)
        y = 4 * X + 0.5 * np.random.randn(100, 1)

        scaler = MinMaxScaler()
        linear_model = LinearRegression()
        steps = [('scaler', scaler),
                 ('linear_model', linear_model)]

        pgraph = PipeGraphRegressor(steps=steps)
        self.assertTrue(pgraph._pipegraph.fit_connections is None)
        self.assertTrue(pgraph._pipegraph.predict_connections is None)
        pgraph.fit(X, y)
        y_pred = pgraph.predict(X)
        self.assertEqual(y_pred.shape[0], y.shape[0])
        self.assertEqual(pgraph._pipegraph.fit_connections, dict(scaler={'X': 'X'},
                                                                 linear_model={'X':('scaler', 'predict'),
                                                                              'y': 'y'}))
        self.assertEqual(pgraph._pipegraph.predict_connections, dict(scaler={'X': 'X'},
                                                                     linear_model={'X':('scaler', 'predict'),
                                                                              'y': 'y'}))


    def test_Pipegraph__ex_3_inject(self):
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import GridSearchCV
        from pipegraph.base import PipeGraphRegressor
        from pipegraph.demo_blocks import CustomPower

        X = pd.DataFrame(dict(X=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
                              sample_weight=np.array(
                                  [0.01, 0.95, 0.10, 0.95, 0.95, 0.10, 0.10, 0.95, 0.95, 0.95, 0.01])))
        y = np.array([10, 4, 20, 16, 25, -60, 85, 64, 81, 100, 150])

        scaler = MinMaxScaler()
        polynomial_features = PolynomialFeatures()
        linear_model = LinearRegression()
        custom_power = CustomPower()
        selector = ColumnSelector(mapping={'X': slice(0, 1),
                                           'sample_weight': slice(1, 2)})

        steps = [('selector', selector),
                 ('custom_power', custom_power),
                 ('scaler', scaler),
                 ('polynomial_features', polynomial_features),
                 ('linear_model', linear_model)]

        pgraph = PipeGraphRegressor(steps=steps)

        self.assertTrue(pgraph._pipegraph.fit_connections is None)
        self.assertTrue(pgraph._pipegraph.predict_connections is None)

        (pgraph.inject(sink='selector', sink_var='X', source='_External', source_var='X')
         .inject('custom_power', 'X', 'selector', 'sample_weight')
         .inject('scaler', 'X', 'selector', 'X')
         .inject('polynomial_features', 'X', 'scaler')
         .inject('linear_model', 'X', 'polynomial_features')
         .inject('linear_model', 'y', source_var='y')
         .inject('linear_model', 'sample_weight', 'custom_power'))

        self.assertTrue(pgraph._pipegraph.fit_connections is not None)
        self.assertTrue(pgraph._pipegraph.predict_connections is not None)
        pgraph.fit(X, y)
        self.assertEqual(pgraph._pipegraph.fit_connections,
                         {'selector': {'X': ('_External', 'X')},
                        'custom_power': {'X': ('selector', 'sample_weight')},
                        'scaler': {'X': ('selector', 'X')},
                        'polynomial_features': {'X': ('scaler', 'predict')},
                        'linear_model': {'X': ('polynomial_features', 'predict'),
                                       'y': ('_External', 'y'),
                                       'sample_weight': ('custom_power', 'predict')}})

        self.assertEqual(pgraph._pipegraph.predict_connections,
                         {'selector': {'X': ('_External', 'X')},
                          'custom_power': {'X': ('selector', 'sample_weight')},
                          'scaler': {'X': ('selector', 'X')},
                          'polynomial_features': {'X': ('scaler', 'predict')},
                          'linear_model': {'X': ('polynomial_features', 'predict'),
                                           'y': ('_External', 'y'),
                                           'sample_weight': ('custom_power', 'predict')}})



    def test_Pipegraph__fit_connections(self):
        pgraph = PipeGraph(self.steps, self.connections)
        pgraph.fit(self.X, self.y)
        fit_nodes_list = list(pgraph._filter_fit_nodes())
        self.assertEqual(sorted(fit_nodes_list), sorted(['Concatenate_Xy',
                                                         'Gaussian_Mixture',
                                                         'Dbscan',
                                                         'Combine_Clustering',
                                                         'Paella',
                                                         'Regressor',
                                                         ]))

    def test_Pipegraph__some_fit_connections(self):
        some_connections = {
            'Concatenate_Xy': dict(df1='X',
                                   df2='y'),
            'Gaussian_Mixture': dict(X=('Concatenate_Xy', 'predict')),
            'Dbscan': dict(X=('Concatenate_Xy', 'predict')),
        }

        pgraph = PipeGraph(steps=self.steps, fit_connections=some_connections, predict_connections=self.connections)
        pgraph.fit(self.X, self.y)

        fit_nodes_list = list(pgraph._filter_fit_nodes())
        self.assertEqual(sorted(fit_nodes_list), sorted(['Concatenate_Xy',
                                                         'Gaussian_Mixture',
                                                         'Dbscan',
                                                         ]))

    def test_Pipegraph__predict_connections(self):
        pgraph = PipeGraph(self.steps, self.connections)
        pgraph.fit(self.X, self.y)
        predict_nodes_list = list(pgraph._filter_predict_nodes())
        self.assertEqual(sorted(predict_nodes_list), sorted(['Concatenate_Xy',
                                                             'Gaussian_Mixture',
                                                             'Dbscan',
                                                             'Combine_Clustering',
                                                             'Paella',
                                                             'Regressor',
                                                             ]))

    def test_Pipegraph__some_predict_connections(self):
        some_connections = {
            'Concatenate_Xy': dict(df1='X',
                                   df2='y'),

            'Gaussian_Mixture': dict(X=('Concatenate_Xy', 'predict')),

            'Dbscan': dict(X=('Concatenate_Xy', 'predict')),
        }

        pgraph = PipeGraph(steps=self.steps, fit_connections=self.connections, predict_connections=some_connections)
        pgraph.fit(self.X, self.y)
        predict_nodes_list = list(pgraph._filter_predict_nodes())
        self.assertEqual(sorted(predict_nodes_list), sorted(['Concatenate_Xy',
                                                             'Gaussian_Mixture',
                                                             'Dbscan',
                                                             ]))

    def test_Pipegraph__read_step_inputs_from_fit_data(self):
        pgraph = self.pgraph
        pgraph._fit_data = {('_External', 'X'): self.X,
                            ('_External', 'y'): self.y,
                            ('Paella', 'predict'): self.y * 2,
                            ('Dbscan', 'predict'): self.y * 4}

        result = pgraph._read_fit_signature_variables_from_graph_data(graph_data=pgraph._fit_data,
                                                                      step_name='Regressor')
        assert_array_equal(result['X'], self.X)
        assert_array_equal(result['y'], self.y)
        assert_array_equal(result['sample_weight'], self.y * 2)
        self.assertEqual(len(result), 3)

    def test_Pipegraph__read_predict_signature_variables_from_graph_data(self):
        pgraph = self.pgraph
        pgraph._predict_data = {('_External', 'X'): self.X,
                                ('_External', 'y'): self.y,
                                ('Paella', 'predict'): self.y * 2,
                                ('Dbscan', 'predict'): self.y * 4}

        result = pgraph._read_predict_signature_variables_from_graph_data(graph_data=pgraph._predict_data,
                                                                          step_name='Regressor')
        assert_array_equal(result['X'], self.X)
        self.assertEqual(len(result), 1)

    def test_Pipegraph__step__predict_lm(self):
        X = self.X
        y = self.y
        lm = LinearRegression()
        lm_step = wrap_adaptee_in_process(lm)
        lm_step.fit(X=X, y=y)
        assert_array_equal(lm.predict(X), lm_step.predict(X=X)['predict'])

    def test_Pipegraph__under_fit__concatenate_Xy(self):
        pgraph = self.pgraph
        pgraph._fit_data = {('_External', 'X'): self.X,
                            ('_External', 'y'): self.y,
                            ('Paella', 'predict'): self.y * 2,
                            ('Dbscan', 'predict'): self.y * 4,
                            }
        expected = pd.concat([self.X, self.y], axis=1)
        pgraph._fit('Concatenate_Xy')
        self.assertEqual(expected.shape, pgraph._fit_data['Concatenate_Xy', 'predict'].shape)
        assert_frame_equal(self.X, pgraph._fit_data['Concatenate_Xy', 'predict'].loc[:,['X']])
        assert_frame_equal(self.y, pgraph._fit_data['Concatenate_Xy', 'predict'].loc[:,['y']])

    def test_Pipegraph__under_fit__Paella__fit(self):
        pgraph = self.pgraph
        pgraph._fit_data = {('_External', 'X'): self.X,
                            ('_External', 'y'): self.y,
                            }
        pgraph._fit('Concatenate_Xy')
        pgraph._fit('Gaussian_Mixture')
        pgraph._fit('Dbscan')
        pgraph._fit('Combine_Clustering')
        self.assertIsNone(pgraph._fit('Paella'))

    def test_Pipegraph__predict__concatenate_Xy(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        expected = pd.concat([X, y], axis=1)
        current_step = pgraph._processes['Concatenate_Xy']
        current_step.fit()
        result = current_step.predict(df1=X, df2=y)['predict']
        self.assertEqual(expected.shape, result.shape)
        assert_frame_equal(self.X, result.loc[:,['X']])
        assert_frame_equal(self.y, result.loc[:,['y']])
        assert_frame_equal(expected, result)

    def test_Pipegraph__predict__gaussian_mixture(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._processes['Gaussian_Mixture']
        current_step.fit(X=X)
        expected = current_step._strategy._adaptee.predict(X=X)
        result = current_step.predict(X=X)['predict']
        assert_array_equal(expected, result)

    def test_Pipegraph__predict__dbscan(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._processes['Dbscan']
        current_step.fit(X=X)
        expected = current_step._strategy._adaptee.fit_predict(X=X)
        result = current_step.predict(X=X)['predict']
        assert_array_equal(expected, result)

    def test_Pipegraph__combine_clustering_predict(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._processes['Gaussian_Mixture']
        current_step.fit(X=pd.concat([X, y], axis=1))
        result_gaussian = current_step.predict(X=pd.concat([X, y], axis=1))['predict']

        current_step = pgraph._processes['Dbscan']
        result_dbscan = current_step.predict(X=pd.concat([X, y], axis=1))['predict']
        self.assertEqual(result_dbscan.min(), 0)

        current_step = pgraph._processes['Combine_Clustering']
        current_step.fit(dominant=result_dbscan, other=result_gaussian)
        expected = current_step._strategy._adaptee.predict(dominant=result_dbscan, other=result_gaussian)
        result = current_step.predict(dominant=result_dbscan, other=result_gaussian)['predict']
        assert_array_equal(expected, result)
        self.assertEqual(result.min(), 0)

    def test_Pipegraph__strategy__dict_key(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._processes['Concatenate_Xy']
        current_step.fit()
        result = current_step.predict(df1=X, df2=y)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_Pipegraph__dbscan__dict_key(self):
        X = self.X
        pgraph = self.pgraph
        current_step = pgraph._processes['Dbscan']
        current_step.fit(X=X)
        result = current_step.predict(X=X)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_Pipegraph__combine_clustering__dict_key(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._processes['Combine_Clustering']
        current_step.fit(dominant=X, other=y)
        result = current_step.predict(dominant=X, other=y)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_Pipegraph__gaussian_mixture__dict_key(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._processes['Gaussian_Mixture']
        current_step.fit(X=X)
        result = current_step.predict(X=X)
        self.assertEqual(sorted(list(result.keys())), sorted(['predict', 'predict_proba']))

    def test_Pipegraph__regressor__dict_key(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._processes['Regressor']
        current_step.fit(X=X, y=y)
        result = current_step.predict(X=X)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_Pipegraph__data_attributes_empty_at_init(self):
        pgraph = self.pgraph
        self.assertEqual(pgraph._fit_data, {})
        self.assertEqual(pgraph._predict_data, {})

    def test_Pipegraph__fit_node_names(self):
        pgraph = self.pgraph.fit(self.X, self.y)
        node_list = list(pgraph._fit_graph.nodes())
        self.assertEqual(sorted(node_list), sorted(['Concatenate_Xy',
                                                    'Gaussian_Mixture',
                                                    'Dbscan',
                                                    'Combine_Clustering',
                                                    'Paella',
                                                    'Regressor',
                                                    ]))

    def test_Pipegraph__predict_node_names(self):
        pgraph = self.pgraph.fit(self.X, self.y)
        node_list = list(pgraph._predict_graph.nodes())
        self.assertEqual(sorted(node_list), sorted(['Concatenate_Xy',
                                                    'Gaussian_Mixture',
                                                    'Dbscan',
                                                    'Combine_Clustering',
                                                    'Paella',
                                                    'Regressor',
                                                    ]))

    def test_Pipegraph__filter_nodes_fit(self):
        pgraph = self.pgraph.fit(self.X, self.y)
        fit_nodes = list(pgraph._filter_fit_nodes())
        self.assertEqual(sorted(fit_nodes), sorted(['Concatenate_Xy',
                                                    'Dbscan',
                                                    'Gaussian_Mixture',
                                                    'Combine_Clustering',
                                                    'Paella',
                                                    'Regressor',
                                                    ]))

    def test_Pipegraph__filter_nodes_predict(self):
        alternative_connections = {
            'Regressor': dict(X='X', y='y')
        }

        pgraph = PipeGraph(steps=self.steps, fit_connections=self.connections,
                           predict_connections=alternative_connections)
        pgraph.fit(self.X, self.y)
        predict_nodes = list(pgraph._filter_predict_nodes())
        self.assertEqual(predict_nodes, ['Regressor'])

    def test_Pipegraph__graph_fit_using_keywords(self):
        pgraph = self.pgraph
        pgraph.fit(X=self.X, y=self.y)
        assert_frame_equal(pgraph._fit_data['_External', 'X'], self.X)
        assert_frame_equal(pgraph._fit_data['_External', 'y'], self.y)

        self.assertEqual(pgraph._fit_data['Dbscan', 'predict'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph._fit_data['Dbscan', 'predict'].min(), 0)

        self.assertEqual(pgraph._fit_data['Gaussian_Mixture', 'predict'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph._fit_data['Gaussian_Mixture', 'predict'].min(), 0)

        self.assertEqual(pgraph._fit_data['Combine_Clustering', 'predict'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph._fit_data['Combine_Clustering', 'predict'].min(), 0)
        self.assertEqual(pgraph._fit_data['Combine_Clustering', 'predict'].max(),
                         pgraph._fit_data['Gaussian_Mixture', 'predict'].max())

        self.assertEqual(pgraph._fit_data['Paella', 'predict'].shape[0], self.y.shape[0])
        self.assertTrue(pgraph._fit_data['Paella', 'predict'].min() >= 0)
        self.assertEqual(pgraph._fit_data['Paella', 'predict'].max(), 1)

        self.assertEqual(pgraph._fit_data['Regressor', 'predict'].shape[0], self.y.shape[0])

    def test_Pipegraph__graph_fit_three_positional(self):
        pgraph = self.pgraph
        self.assertRaises(ValueError, pgraph.fit, self.X, self.y, self.y)

    def test_Pipegraph__graph_fit_two_positional(self):
        pgraph = self.pgraph
        pgraph.fit(self.X, self.y)
        assert_frame_equal(pgraph._fit_data['_External', 'X'], self.X)
        assert_frame_equal(pgraph._fit_data['_External', 'y'], self.y)

        self.assertEqual(pgraph._fit_data['Dbscan', 'predict'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph._fit_data['Dbscan', 'predict'].min(), 0)

        self.assertEqual(pgraph._fit_data['Gaussian_Mixture', 'predict'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph._fit_data['Gaussian_Mixture', 'predict'].min(), 0)

        self.assertEqual(pgraph._fit_data['Combine_Clustering', 'predict'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph._fit_data['Combine_Clustering', 'predict'].min(), 0)
        self.assertEqual(pgraph._fit_data['Combine_Clustering', 'predict'].max(),
                         pgraph._fit_data['Gaussian_Mixture', 'predict'].max())

        self.assertEqual(pgraph._fit_data['Paella', 'predict'].shape[0], self.y.shape[0])
        self.assertTrue(pgraph._fit_data['Paella', 'predict'].min() >= 0)
        self.assertEqual(pgraph._fit_data['Paella', 'predict'].max(), 1)

        self.assertEqual(pgraph._fit_data['Regressor', 'predict'].shape[0], self.y.shape[0])

    def test_Pipegraph__graph_fit_one_positional(self):
        pgraph = self.pgraph
        pgraph.fit(self.X, y=self.y)
        assert_frame_equal(pgraph._fit_data['_External', 'X'], self.X)
        assert_frame_equal(pgraph._fit_data['_External', 'y'], self.y)

        self.assertEqual(pgraph._fit_data['Dbscan', 'predict'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph._fit_data['Dbscan', 'predict'].min(), 0)

        self.assertEqual(pgraph._fit_data['Gaussian_Mixture', 'predict'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph._fit_data['Gaussian_Mixture', 'predict'].min(), 0)

        self.assertEqual(pgraph._fit_data['Combine_Clustering', 'predict'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph._fit_data['Combine_Clustering', 'predict'].min(), 0)
        self.assertEqual(pgraph._fit_data['Combine_Clustering', 'predict'].max(),
                         pgraph._fit_data['Gaussian_Mixture', 'predict'].max())

        self.assertEqual(pgraph._fit_data['Paella', 'predict'].shape[0], self.y.shape[0])
        self.assertTrue(pgraph._fit_data['Paella', 'predict'].min() >= 0)
        self.assertEqual(pgraph._fit_data['Paella', 'predict'].max(), 1)

        self.assertEqual(pgraph._fit_data['Regressor', 'predict'].shape[0], self.y.shape[0])



    def test_Pipegraph__graph_predict_using_keywords(self):
        pgraph = self.pgraph
        pgraph.fit(X=self.X, y=self.y)
        pgraph.predict(X=self.X, y=self.y)
        assert_frame_equal(pgraph._predict_data['_External', 'X'], self.X)
        assert_frame_equal(pgraph._predict_data['_External', 'y'], self.y)
        self.assertEqual(pgraph._predict_data['Regressor', 'predict'].shape[0], self.y.shape[0])

    def test_Pipegraph__graph_predict_using_three_positional(self):
        pgraph = self.pgraph
        pgraph.fit(X=self.X, y=self.y)
        self.assertRaises(ValueError, pgraph.predict, self.X, self.y, self.y)

    def test_Pipegraph__graph_predict_using_two_positional(self):
        pgraph = self.pgraph
        pgraph.fit(X=self.X, y=self.y)
        pgraph.predict(self.X, self.y)
        assert_frame_equal(pgraph._predict_data['_External', 'X'], self.X)
        assert_frame_equal(pgraph._predict_data['_External', 'y'], self.y)
        self.assertEqual(pgraph._predict_data['Regressor', 'predict'].shape[0], self.y.shape[0])

    def test_Pipegraph__graph_predict_using_one_positional(self):
        pgraph = self.pgraph
        pgraph.fit(X=self.X, y=self.y)
        pgraph.predict(self.X, y=self.y)
        assert_frame_equal(pgraph._predict_data['_External', 'X'], self.X)
        assert_frame_equal(pgraph._predict_data['_External', 'y'], self.y)
        self.assertEqual(pgraph._predict_data['Regressor', 'predict'].shape[0], self.y.shape[0])


    def test_Pipegraph__step_get_params_multiple(self):
        pgraph = self.pgraph
        self.assertEqual(pgraph._processes['Concatenate_Xy'].get_params(), {})
        self.assertEqual(pgraph._processes['Gaussian_Mixture'].get_params()['n_components'], 3)
        self.assertEqual(pgraph._processes['Dbscan'].get_params()['eps'], 0.5)
        self.assertEqual(pgraph._processes['Combine_Clustering'].get_params(), {})
        self.assertEqual(pgraph._processes['Paella'].get_params()['noise_label'], None)
        self.assertEqual(pgraph._processes['Paella'].get_params()['max_it'], 10)
        self.assertEqual(pgraph._processes['Paella'].get_params()['regular_size'], 100)
        self.assertEqual(pgraph._processes['Paella'].get_params()['minimum_size'], 30)
        self.assertEqual(pgraph._processes['Paella'].get_params()['width_r'], 0.95)
        self.assertEqual(pgraph._processes['Paella'].get_params()['n_neighbors'], 1)
        self.assertEqual(pgraph._processes['Paella'].get_params()['power'], 10)
        self.assertEqual(pgraph._processes['Regressor'].get_params(),
                         {'copy_X': True, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False})

    def test_Pipegraph__step_set_params_multiple(self):
        pgraph = self.pgraph
        self.assertRaises(ValueError, pgraph._processes['Concatenate_Xy'].set_params, ham=2, spam=9)
        self.assertEqual(pgraph._processes['Gaussian_Mixture'].set_params(n_components=5).get_params()['n_components'],
                         5)
        self.assertEqual(pgraph._processes['Dbscan'].set_params(eps=10.2).get_params()['eps'],
                         10.2)
        self.assertEqual(pgraph._processes['Paella'].set_params(noise_label=-3).get_params()['noise_label'],
                         -3)
        self.assertEqual(pgraph._processes['Regressor'].set_params(copy_X=False).get_params(),
                         {'copy_X': False, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False})
        self.assertEqual(pgraph._processes['Regressor'].set_params(n_jobs=13).get_params(),
                         {'copy_X': False, 'fit_intercept': True, 'n_jobs': 13, 'normalize': False})

    def test_Pipegraph__named_steps(self):
        pgraph = self.pgraph
        self.assertEqual(pgraph.named_steps, dict(self.steps))
        self.assertEqual(len(pgraph.named_steps), 6)

class TestPipeGraphFitConnections(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = np.random.rand(self.size, 1)
        self.y = 2 * self.X

        sc = MinMaxScaler()
        gm = GaussianMixture(n_components=3)
        km = KMeans(n_clusters=4)

        steps = [('scaler', sc), ('gaussian', gm), ('kmeans', km)]
        connections_1 = {'scaler': dict(X='X'), 'gaussian': 'scaler'}
        connections_2 = {'scaler': dict(X='X'), 'kmeans': 'scaler'}

        self.sc = sc
        self.gm = gm
        self.km = km

        self.steps = steps
        self.connections_1 = connections_1
        self.connections_2 = connections_2
        self.pgraph = PipeGraph(steps=steps, fit_connections=connections_1)
        self.param_grid = dict(fit_connections=[connections_1, connections_2])

    def test_Pipegraph__fit_connections_get_params(self):
        pgraph = self.pgraph
        self.assertEqual(pgraph.get_params()['fit_connections'], self.connections_1)
        pgraph.set_params(fit_connections=self.connections_2)
        self.assertEqual(pgraph.get_params()['fit_connections'], self.connections_2)

class TestPipeGraphSingleNodeLinearModel(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = np.random.rand(self.size, 1)
        self.y = 2*self.X

        lm = LinearRegression()
        steps = [('linear_model', lm)]
        connections = {'linear_model': dict(X='X', y='y')}
        self.lm = lm
        self.steps = steps
        self.connections = connections
        self.pgraph = PipeGraph(steps=steps, fit_connections=connections)
        self.param_grid = dict(linear_model__fit_intercept=[False, True],
                               linear_model__normalize=[True, False])

    def test_single_node_fit(self):
        pgraph = self.pgraph
        self.assertFalse(hasattr(pgraph._processes['linear_model'], 'coef_'))
        pgraph.fit(X=self.X, y=self.y)
        self.assertTrue(hasattr(pgraph._processes['linear_model'], 'coef_'))
        self.assertAlmostEqual(pgraph._processes['linear_model'].coef_[0][0], 2)

        self.assertEqual(pgraph._fit_data['linear_model', 'predict'].shape[0], self.size)
        result = pgraph.predict(X=self.X)['predict']
        self.assertEqual(pgraph._fit_data['linear_model', 'predict'].shape[0], self.size)
        self.assertEqual(result.shape[0], self.size)

    def test_get_params(self):
        pgraph = self.pgraph
        result = pgraph.get_params()
        expected = {'linear_model': self.lm,
                    'linear_model__copy_X': True,
                    'linear_model__fit_intercept': True,
                    'linear_model__n_jobs': 1,
                    'linear_model__normalize': False,
                    'steps': self.steps}
        for item in expected:
            self.assertTrue(item in result)

    def test_set_params(self):
        pgraph = self.pgraph
        result_pre = pgraph.get_params()
        expected_pre = {'linear_model': self.lm,
                        'linear_model__copy_X': True,
                        'linear_model__fit_intercept': True,
                        'linear_model__n_jobs': 1,
                        'linear_model__normalize': False,
                        'steps': self.steps}
        for item in expected_pre:
            self.assertTrue(item in result_pre)

        result_post = pgraph.set_params(linear_model__copy_X=False).get_params()
        expected_post = {'linear_model': self.lm,
                         'linear_model__copy_X': False,
                         'linear_model__fit_intercept': True,
                         'linear_model__n_jobs': 1,
                         'linear_model__normalize': False,
                         'steps': self.steps}
        for item in expected_post:
            self.assertTrue(item in result_post)

    # def test_GridSearchCV_fit(self):
    #     pgraph = self.pgraph
    #     gs = GridSearchCV(pgraph, param_grid=self.param_grid, scoring=pgraph.r2_score)
    #     gs.fit(X=self.X, y=self.y)
    #     self.assertEqual(pgraph.data['linear_model', 'predict'].shape[0], self.size)
    #     result = pgraph.predict(X=self.X)
    #     self.assertEqual(pgraph.data['linear_model', 'predict'].shape[0], self.size)
    #     self.assertEqual(result.shape[0], self.size)


class TestTwoNodes(unittest.TestCase):
    def setUp(self):
        self.size = 1000
        self.X = np.random.rand(self.size, 1)
        self.y = self.X * 2
        sc = MinMaxScaler(feature_range=(0, 1))
        lm = LinearRegression()
        steps = [('scaler', sc),
                 ('linear_model', lm)]
        connections = {'scaler': dict(X='X'),
                       'linear_model': dict(X=('scaler', 'predict'),
                                            y='y')}
        self.lm = lm
        self.sc = sc
        self.steps = steps
        self.connections = connections
        self.pgraph = PipeGraph(steps=steps, fit_connections=connections)
        self.param_grid = dict(linear_model__fit_intercept=[False, True],
                               linear_model__normalize=[True, False],
                               )

    def test_TwoNodes_fit(self):
        pgraph = self.pgraph
        self.assertFalse(hasattr(pgraph._processes['linear_model'], 'coef_'))
        pgraph.fit(X=self.X, y=self.y)
        self.assertTrue(hasattr(pgraph._processes['linear_model'], 'coef_'))
        self.assertTrue(1.9 < pgraph._processes['linear_model'].coef_[0][0] < 2.1)

        self.assertEqual(pgraph._fit_data['linear_model', 'predict'].shape[0], self.size)
        result = pgraph.predict(X=self.X)['predict']
        self.assertEqual(pgraph._fit_data['linear_model', 'predict'].shape[0], self.size)
        self.assertEqual(result.shape[0], self.size)



class TestPipeGraphRegressor(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = pd.DataFrame(dict(X=np.random.rand(self.size, )))
        self.y = pd.DataFrame(dict(y=(np.random.rand(self.size, ))))
        sc = MinMaxScaler()
        lm = LinearRegression()
        neutral_regressor = NeutralRegressor()

        steps = [('scaler', sc),
                 ('model',  lm),
                 ]
        connections = {'scaler': {'X': 'X'},
                       'model': {'X': ('scaler', 'predict'), 'y': 'y'}, }
        model = PipeGraphRegressor(steps, connections)

        steps = [('scaler', sc),
                 ('model', lm),
                 ('neutral', neutral_regressor)
                 ]
        connections = {'scaler': {'X': 'X'},
                       'model': {'X': ('scaler', 'predict'), 'y': 'y'},
                       'neutral': {'X': 'model'}}

        model_custom = PipeGraphRegressor(steps, connections)

        self.sc =sc
        self.lm = lm
        self.model = model
        self.model_custom = model_custom

    def test_PipeGraphRegressor__GridSearchCV(self):
        model = self.model
        X = self.X
        y = self.y
        param_grid = [{'model__fit_intercept': [True, False]}, ]
        gs = GridSearchCV(estimator=model, param_grid=param_grid)
        gs.fit(X, y)
        result = gs.best_estimator_.predict(X)
        self.assertEqual(result.shape[0], X.shape[0])

    def test_PipeGraphRegressor__score_sklearn_last_step(self):
        model = self.model
        X = self.X
        y = self.y
        param_grid = [{'model__fit_intercept': [True, False]}, ]
        gs = GridSearchCV(estimator=model, param_grid=param_grid)
        gs.fit(X, y)
        result = gs.score(X, y)
        self.assertTrue(result > -42)

    def test_PipeGraphRegressor__score_custom_last_step(self):
        model = self.model_custom
        X = self.X
        y = self.y
        param_grid = [{'model__fit_intercept': [True, False]}, ]
        gs = GridSearchCV(estimator=model, param_grid=param_grid)
        gs.fit(X, y)
        result = gs.score(X, y)
        self.assertTrue(result > -42)


class TestPipeGraphClassifier(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = pd.DataFrame(dict(X=np.random.rand(self.size, )))
        self.y = pd.DataFrame(dict(y=(np.random.rand(self.size, ) > 0.5).astype(int)))
        sc = MinMaxScaler()
        nb = GaussianNB()
        steps = [('scaler', sc),
                 ('model', nb),
                 ]
        connections = {'scaler': {'X': 'X'},
                       'model': {'X': ('scaler', 'predict'), 'y': 'y'}, }
        model = PipeGraphClassifier(steps, connections)

        self.sc =sc
        self.nb = nb
        self.model = model

    def test_PipeGraphClassifier__GridSearchCV(self):
        model = self.model
        X = self.X
        y = self.y.values.reshape(-1,)
        param_grid = [{'scaler__feature_range': [(-0.5, 0.5), (0, 1), (-0.9, 0.9)]}, ]
        gs = GridSearchCV(estimator=model, param_grid=param_grid)
        gs.fit(X, y)
        result = gs.best_estimator_.predict(X)
        self.assertEqual(result.shape[0], X.shape[0])


class TestProcess(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = np.random.rand(self.size, 1)
        self.y = self.X + np.random.randn(self.size, 1)

    def test_process__init(self):
        lm = LinearRegression()
        stepstrategy = AdapterForFitPredictAdaptee(lm)
        step = Process(stepstrategy)
        self.assertEqual(step._strategy, stepstrategy)

    def test_process__fit_lm(self):
        X = self.X
        y = self.y
        lm = LinearRegression()
        stepstrategy = AdapterForFitPredictAdaptee(lm)
        step = Process(stepstrategy)
        self.assertEqual(hasattr(step, 'coef_'), False)
        result = step.fit(X=X, y=y)
        self.assertEqual(hasattr(step, 'coef_'), True)
        self.assertEqual(step, result)

    def test_process__fit_dbscan(self):
        X = self.X
        y = self.y
        db = DBSCAN()
        stepstrategy = AdapterForAtomicFitPredictAdaptee(db)
        step = Process(stepstrategy)
        self.assertEqual(hasattr(step, 'core_sample_indices_'), False)
        result_fit = step.fit(X=X)
        self.assertEqual(hasattr(step, 'core_sample_indices_'), False)
        self.assertEqual(step, result_fit)
        result_predict = step.predict(X=X)['predict']
        self.assertEqual(hasattr(step, 'core_sample_indices_'), True)
        self.assertEqual(result_predict.shape, (self.size,))

    def test_process__predict(self):
        X = self.X
        y = self.y
        lm = LinearRegression()
        gm = GaussianMixture()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        gm_strategy = AdapterForFitPredictAdaptee(gm)
        step_lm = Process(lm_strategy)
        step_gm = Process(gm_strategy)

        step_lm.fit(X=X, y=y)
        step_gm.fit(X=X)

        result_lm = step_lm.predict(X=X)
        result_gm = step_gm.predict(X=X)

        self.assertEqual(list(result_lm.keys()), ['predict'])
        self.assertEqual(sorted(list(result_gm.keys())), sorted(['predict', 'predict_proba']))

    def test_process__get_params(self):
        lm = LinearRegression()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        step = Process(lm_strategy)
        result_lm = step.get_params()
        self.assertEqual(result_lm, {'copy_X': True,
                                     'fit_intercept': True,
                                     'n_jobs': 1,
                                     'normalize': False})

    def test_process__set_params(self):
        lm = LinearRegression()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        step = Process(lm_strategy)
        result_lm_pre = step.get_params()
        self.assertEqual(result_lm_pre, {'copy_X': True,
                                         'fit_intercept': True,
                                         'n_jobs': 1,
                                         'normalize': False})
        step.set_params(fit_intercept=False)
        result_lm_post = step.get_params()
        self.assertEqual(result_lm_post, {'copy_X': True,
                                          'fit_intercept': False,
                                          'n_jobs': 1,
                                          'normalize': False})

    def test_process__get_fit_signature(self):
        lm = LinearRegression()
        gm = GaussianMixture()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        gm_strategy = AdapterForFitPredictAdaptee(gm)
        step_lm = Process(lm_strategy)
        step_gm = Process(gm_strategy)
        result_lm = step_lm._get_fit_signature()
        result_gm = step_gm._get_fit_signature()

        self.assertEqual(result_lm, ['X', 'y', 'sample_weight'])
        self.assertEqual(result_gm, ['X', 'y'])

    def test_process__get_predict_signature_lm(self):
        lm = LinearRegression()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        step_lm = Process(lm_strategy)
        result_lm = step_lm._get_predict_signature()
        self.assertEqual(result_lm, ['X'])

    def test_process__getattr__(self):
        lm = LinearRegression()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        step = Process(lm_strategy)
        self.assertEqual(step.copy_X, True)

    def test_process__setattr__(self):
        lm = LinearRegression()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        step = Process(lm_strategy)
        self.assertEqual(step.copy_X, True)
        step.copy_X = False
        self.assertEqual(step.copy_X, False)
        self.assertEqual('copy_X' in dir(step), False)

    def test_process__delattr__(self):
        lm = LinearRegression()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        step = Process(lm_strategy)
        self.assertEqual(step.copy_X, True)
        self.assertEqual('copy_X' in dir(step), False)
        self.assertEqual('copy_X' in dir(lm), True)
        del step.copy_X
        self.assertEqual('copy_X' in dir(lm), False)

    def test_process__repr__(self):
        lm = LinearRegression()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        step = Process(lm_strategy)
        result = step.__repr__()
        self.assertEqual(result, lm.__repr__())

class TestPipeGraphCompositable(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = np.random.rand(self.size, 1)
        self.y = 2*self.X
        lm = LinearRegression()
        steps = [('linear_model', lm)]
        self.lm = lm
        self.steps = steps
        self.pgraph = PipeGraph(steps=steps)

    def test_compositable__isinstance(self):
        X = self.X
        y = self.y
        new_graph = PipeGraph(steps=[('pgraph', self.pgraph)])
        self.assertEqual(new_graph.named_steps,  {'pgraph': self.pgraph})

        new_graph.fit(X, y)
        result = new_graph.predict(X)['predict']
        expected = self.pgraph.predict(X)['predict']
        self.assertEqual(result.shape[0], expected.shape[0])


if __name__ == '__main__':
    unittest.main()
