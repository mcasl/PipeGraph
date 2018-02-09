import logging
import unittest

import pandas as pd
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

from pipegraph.paella import Paella
from pipegraph.pipeGraph import (PipeGraph,
                                 Step,
                                 FitPredict,
                                 FitTransform,
                                 AtomicFitPredict,
                                 make_step,
                                 build_graph,
                                 CustomCombination,
                                 CustomConcatenation,
                                 TrainTestSplit,
                                 )

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestStepStrategy(unittest.TestCase):
    def setUp(self):
        URL = "https://raw.githubusercontent.com/mcasl/PAELLA/master/data/sin_60_percent_noise.csv"
        data = pd.read_csv(URL, usecols=['V1', 'V2'])
        self.X = data[['V1']]
        self.y = data[['V2']]

    def test_stepstrategy__init(self):
        lm = LinearRegression()
        stepstrategy = FitPredict(lm)
        self.assertEqual(stepstrategy._adaptee, lm)

    def test_stepstrategy__fit_FitPredict(self):
        X = self.X
        y = self.y
        lm = LinearRegression()
        stepstrategy = FitPredict(lm)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'coef_'), False)
        result = stepstrategy.fit(X=X, y=y)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'coef_'), True)
        self.assertEqual(stepstrategy, result)

    def test_stepstrategy__fit_FitTransform(self):
        X = self.X
        sc = MinMaxScaler()
        stepstrategy = FitTransform(sc)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'n_samples_seen_'), False)
        result = stepstrategy.fit(X=X)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'n_samples_seen_'), True)
        self.assertEqual(stepstrategy, result)

    def test_stepstrategy__fit_AtomicFitPredict(self):
        X = self.X
        db = DBSCAN()
        stepstrategy = AtomicFitPredict(db)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'core_sample_indices_'), False)
        result_fit = stepstrategy.fit(X=X)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'core_sample_indices_'), False)
        self.assertEqual(stepstrategy, result_fit)
        result_predict = stepstrategy.predict(X=X)['predict']
        self.assertEqual(hasattr(stepstrategy._adaptee, 'core_sample_indices_'), True)
        self.assertEqual(result_predict.shape, (10000,))

    def test_stepstrategy__predict_FitPredict(self):
        X = self.X
        y = self.y
        lm = LinearRegression()
        lm_strategy = FitPredict(lm)
        lm_strategy.fit(X=X, y=y)
        result_lm = lm_strategy.predict(X=X)
        self.assertEqual(list(result_lm.keys()), ['predict'])
        self.assertEqual(result_lm['predict'].shape, (10000, 1))

        gm = GaussianMixture()
        gm_strategy = FitPredict(gm)
        gm_strategy.fit(X=X)
        result_gm = gm_strategy.predict(X=X)
        self.assertEqual(sorted(list(result_gm.keys())),
                         sorted(['predict', 'predict_proba']))
        self.assertEqual(result_gm['predict'].shape, (10000,))

    def test_stepstrategy__predict_FitTransform(self):
        X = self.X
        sc = MinMaxScaler()
        sc_strategy = FitTransform(sc)
        sc_strategy.fit(X=X)
        result_sc = sc_strategy.predict(X=X)
        self.assertEqual(list(result_sc.keys()), ['predict'])
        self.assertEqual(result_sc['predict'].shape, (10000, 1))

    def test_stepstrategy__predict_AtomicFitPredict(self):
        X = self.X
        y = self.y
        db = DBSCAN()
        db_strategy = AtomicFitPredict(db)
        db_strategy.fit(X=X, y=y)
        result_db = db_strategy.predict(X=X)
        self.assertEqual(list(result_db.keys()), ['predict'])
        self.assertEqual(result_db['predict'].shape, (10000,))

    def test_stepstrategy__get_fit_signature(self):
        lm = LinearRegression()
        gm = GaussianMixture()
        lm_strategy = FitPredict(lm)
        gm_strategy = FitPredict(gm)

        result_lm = lm_strategy._get_fit_signature()
        result_gm = gm_strategy._get_fit_signature()

        self.assertEqual(sorted(result_lm), sorted(['X', 'y', 'sample_weight']))
        self.assertEqual(sorted(result_gm), sorted(['X', 'y']))

    def test_stepstrategy__get_predict_signature_FitPredict(self):
        lm = LinearRegression()
        lm_strategy = FitPredict(lm)
        result_lm = lm_strategy._get_predict_signature()
        self.assertEqual(result_lm, ['X'])

    def test_stepstrategy__get_predict_signature_FitTransform(self):
        sc = MinMaxScaler()
        sc_strategy = FitTransform(sc)
        result_sc = sc_strategy._get_predict_signature()
        self.assertEqual(result_sc, ['X'])

    def test_stepstrategy__get_predict_signature_AtomicFitPredict(self):
        db = DBSCAN()
        db_strategy = AtomicFitPredict(db)
        result_db = db_strategy._get_predict_signature()
        self.assertEqual(sorted(result_db), sorted(['X', 'y', 'sample_weight']))

    def test_stepstrategy__get_params(self):
        lm = LinearRegression()
        lm_strategy = FitPredict(lm)
        result_lm = lm_strategy.get_params()
        self.assertEqual(result_lm, {'copy_X': True,
                                     'fit_intercept': True,
                                     'n_jobs': 1,
                                     'normalize': False})

    def test_stepstrategy__set_params(self):
        lm = LinearRegression()
        lm_strategy = FitPredict(lm)
        result_lm_pre = lm_strategy.get_params()
        self.assertEqual(result_lm_pre, {'copy_X': True,
                                         'fit_intercept': True,
                                         'n_jobs': 1,
                                         'normalize': False})
        lm_strategy.set_params(fit_intercept=False)
        result_lm_post = lm_strategy.get_params()
        self.assertEqual(result_lm_post, {'copy_X': True,
                                          'fit_intercept': False,
                                          'n_jobs': 1,
                                          'normalize': False})

    def test_stepstrategy__getattr__(self):
        lm = LinearRegression()
        lm_strategy = FitPredict(lm)
        self.assertEqual(lm_strategy.copy_X, True)

    def test_stepstrategy__setattr__(self):
        lm = LinearRegression()
        lm_strategy = FitPredict(lm)
        self.assertEqual(lm_strategy.copy_X, True)
        lm_strategy.copy_X = False
        self.assertEqual(lm_strategy.copy_X, False)
        self.assertEqual('copy_X' in dir(lm_strategy), False)

    def test_stepstrategy__delattr__(self):
        lm = LinearRegression()
        lm_strategy = FitPredict(lm)
        self.assertEqual(lm_strategy.copy_X, True)
        self.assertEqual('copy_X' in dir(lm_strategy), False)
        self.assertEqual('copy_X' in dir(lm), True)
        del lm_strategy.copy_X
        self.assertEqual('copy_X' in dir(lm), False)

    def test_stepstrategy__repr__(self):
        lm = LinearRegression()
        lm_strategy = FitPredict(lm)
        result = lm_strategy.__repr__()
        self.assertEqual(result, lm.__repr__())


class TestStep(unittest.TestCase):
    def setUp(self):
        URL = "https://raw.githubusercontent.com/mcasl/PAELLA/master/data/sin_60_percent_noise.csv"
        data = pd.read_csv(URL, usecols=['V1', 'V2'])
        self.X = data[['V1']]
        self.y = data[['V2']]

    def test_step__init(self):
        lm = LinearRegression()
        stepstrategy = FitPredict(lm)
        step = Step(stepstrategy)
        self.assertEqual(step._strategy, stepstrategy)

    def test_step__fit_lm(self):
        X = self.X
        y = self.y
        lm = LinearRegression()
        stepstrategy = FitPredict(lm)
        step = Step(stepstrategy)
        self.assertEqual(hasattr(step, 'coef_'), False)
        result = step.fit(X=X, y=y)
        self.assertEqual(hasattr(step, 'coef_'), True)
        self.assertEqual(step, result)

    def test_step__fit_dbscan(self):
        X = self.X
        y = self.y
        db = DBSCAN()
        stepstrategy = AtomicFitPredict(db)
        step = Step(stepstrategy)
        self.assertEqual(hasattr(step, 'core_sample_indices_'), False)
        result_fit = step.fit(X=X)
        self.assertEqual(hasattr(step, 'core_sample_indices_'), False)
        self.assertEqual(step, result_fit)
        result_predict = step.predict(X=X)['predict']
        self.assertEqual(hasattr(step, 'core_sample_indices_'), True)
        self.assertEqual(result_predict.shape, (10000,))

    def test_step__predict(self):
        X = self.X
        y = self.y
        lm = LinearRegression()
        gm = GaussianMixture()
        lm_strategy = FitPredict(lm)
        gm_strategy = FitPredict(gm)
        step_lm = Step(lm_strategy)
        step_gm = Step(gm_strategy)

        step_lm.fit(X=X, y=y)
        step_gm.fit(X=X)

        result_lm = step_lm.predict(X=X)
        result_gm = step_gm.predict(X=X)

        self.assertEqual(list(result_lm.keys()), ['predict'])
        self.assertEqual(sorted(list(result_gm.keys())), sorted(['predict', 'predict_proba']))

    def test_step__get_params(self):
        lm = LinearRegression()
        lm_strategy = FitPredict(lm)
        step = Step(lm_strategy)
        result_lm = step.get_params()
        self.assertEqual(result_lm, {'copy_X': True,
                                     'fit_intercept': True,
                                     'n_jobs': 1,
                                     'normalize': False})

    def test_step__set_params(self):
        lm = LinearRegression()
        lm_strategy = FitPredict(lm)
        step = Step(lm_strategy)
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

    def test_step__get_fit_signature(self):
        lm = LinearRegression()
        gm = GaussianMixture()
        lm_strategy = FitPredict(lm)
        gm_strategy = FitPredict(gm)
        step_lm = Step(lm_strategy)
        step_gm = Step(gm_strategy)
        result_lm = step_lm._get_fit_signature()
        result_gm = step_gm._get_fit_signature()

        self.assertEqual(result_lm, ['X', 'y', 'sample_weight'])
        self.assertEqual(result_gm, ['X', 'y'])

    def test_step__get_predict_signature_lm(self):
        lm = LinearRegression()
        lm_strategy = FitPredict(lm)
        step_lm = Step(lm_strategy)
        result_lm = step_lm._get_predict_signature()
        self.assertEqual(result_lm, ['X'])

    def test_step__getattr__(self):
        lm = LinearRegression()
        lm_strategy = FitPredict(lm)
        step = Step(lm_strategy)
        self.assertEqual(step.copy_X, True)

    def test_step__setattr__(self):
        lm = LinearRegression()
        lm_strategy = FitPredict(lm)
        step = Step(lm_strategy)
        self.assertEqual(step.copy_X, True)
        step.copy_X = False
        self.assertEqual(step.copy_X, False)
        self.assertEqual('copy_X' in dir(step), False)

    def test_step__delattr__(self):
        lm = LinearRegression()
        lm_strategy = FitPredict(lm)
        step = Step(lm_strategy)
        self.assertEqual(step.copy_X, True)
        self.assertEqual('copy_X' in dir(step), False)
        self.assertEqual('copy_X' in dir(lm), True)
        del step.copy_X
        self.assertEqual('copy_X' in dir(lm), False)

    def test_step__repr__(self):
        lm = LinearRegression()
        lm_strategy = FitPredict(lm)
        step = Step(lm_strategy)
        result = step.__repr__()
        self.assertEqual(result, lm.__repr__())


class TestRootFunctions(unittest.TestCase):
    def setUp(self):
        URL = "https://raw.githubusercontent.com/mcasl/PAELLA/master/data/sin_60_percent_noise.csv"
        data = pd.read_csv(URL, usecols=['V1', 'V2'])
        self.X = X = data[['V1']]
        self.y = y = data[['V2']]
        concatenator = CustomConcatenation()
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

    def test_make_step(self):
        tests_table = [
            {'model': LinearRegression(),
             'expected_class': FitPredict},

            {'model': MinMaxScaler(),
             'expected_class': FitTransform},

            {'model': DBSCAN(),
             'expected_class': AtomicFitPredict},
        ]

        for test_dict in tests_table:
            model = test_dict['model']
            step = make_step(model)
            self.assertEqual(isinstance(step._strategy, test_dict['expected_class']), True)

    def test_make_step_raise_exception(self):
        model = object()
        self.assertRaises(ValueError, make_step, model)

    def test_build_graph_node_names(self):
        graph = build_graph(self.connections)

        node_list = list(graph.nodes)
        self.assertEqual(sorted(node_list), sorted(['Concatenate_Xy',
                                                    'Gaussian_Mixture',
                                                    'Dbscan',
                                                    'Combine_Clustering',
                                                    'Paella',
                                                    'Regressor',
                                                    ]))

    def test_build_graph_node_edges(self):
        graph = build_graph(self.connections)

        in_edges = {name: [origin for origin, destination in list(graph.in_edges(name))]
                    for name in graph}
        logger.debug("in_edges: %s", in_edges)
        self.assertEqual(in_edges['Gaussian_Mixture'], ['Concatenate_Xy'])
        self.assertEqual(in_edges['Dbscan'], ['Concatenate_Xy'])
        self.assertEqual(sorted(in_edges['Combine_Clustering']), sorted(['Gaussian_Mixture', 'Dbscan']))
        self.assertEqual(in_edges['Paella'], ['Combine_Clustering'])
        self.assertEqual(in_edges['Regressor'], ['Paella'])


class TestPipegraph(unittest.TestCase):
    def setUp(self):
        URL = "https://raw.githubusercontent.com/mcasl/PAELLA/master/data/sin_60_percent_noise.csv"
        data = pd.read_csv(URL, usecols=['V1', 'V2'])
        self.X = X = data[['V1']]
        self.y = y = data[['V2']]
        concatenator = CustomConcatenation()
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
        self.pgraph = PipeGraph(steps=steps, connections=connections)

    def test_Pipegraph__External_step_name(self):
        self.assertRaises(ValueError, PipeGraph, self.steps_external, self.connections_external)

    def test_Pipegraph__use_for_fit_all(self):
        pgraph = PipeGraph(self.steps, self.connections)

        fit_nodes_list = list(pgraph._filter_fit_nodes())
        self.assertEqual(sorted(fit_nodes_list), sorted(['Concatenate_Xy',
                                                         'Gaussian_Mixture',
                                                         'Dbscan',
                                                         'Combine_Clustering',
                                                         'Paella',
                                                         'Regressor',
                                                         ]))

    def test_Pipegraph__use_for_fit_some(self):
        some_connections = {
            'Concatenate_Xy': dict(df1='X',
                                   df2='y'),
            'Gaussian_Mixture': dict(X=('Concatenate_Xy', 'predict')),
            'Dbscan': dict(X=('Concatenate_Xy', 'predict')),
        }

        pgraph = PipeGraph(steps=self.steps,
                           connections=some_connections,
                           alternative_connections=self.connections)

        fit_nodes_list = list(pgraph._filter_fit_nodes())
        self.assertEqual(sorted(fit_nodes_list), sorted(['Concatenate_Xy',
                                                         'Gaussian_Mixture',
                                                         'Dbscan',
                                                         ]))

    def test_Pipegraph__use_for_predict_all(self):
        pgraph = PipeGraph(self.steps, self.connections)

        predict_nodes_list = list(pgraph._filter_predict_nodes())
        self.assertEqual(sorted(predict_nodes_list), sorted(['Concatenate_Xy',
                                                             'Gaussian_Mixture',
                                                             'Dbscan',
                                                             'Combine_Clustering',
                                                             'Paella',
                                                             'Regressor',
                                                             ]))

    def test_Pipegraph__use_for_predict_some(self):
        some_connections = {
            'Concatenate_Xy': dict(df1='X',
                                   df2='y'),

            'Gaussian_Mixture': dict(X=('Concatenate_Xy', 'predict')),

            'Dbscan': dict(X=('Concatenate_Xy', 'predict')),
        }

        pgraph = PipeGraph(steps=self.steps,
                           connections=self.connections,
                           alternative_connections=some_connections)

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
        lm_step = make_step(lm)
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
        assert_frame_equal(expected, pgraph._fit_data['Concatenate_Xy', 'predict'])

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
        current_step = pgraph._step['Concatenate_Xy']
        current_step.fit(df1=X, df2=y)
        result = current_step.predict(df1=X, df2=y)['predict']
        assert_frame_equal(expected, result)

    def test_Pipegraph__predict__gaussian_mixture(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._step['Gaussian_Mixture']
        current_step.fit(X=X)
        expected = current_step._strategy._adaptee.predict(X=X)
        result = current_step.predict(X=X)['predict']
        assert_array_equal(expected, result)

    def test_Pipegraph__predict__dbscan(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._step['Dbscan']
        current_step.fit(X=X)
        expected = current_step._strategy._adaptee.fit_predict(X=X)
        result = current_step.predict(X=X)['predict']
        assert_array_equal(expected, result)

    def test_Pipegraph__combine_clustering_predict(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._step['Gaussian_Mixture']
        current_step.fit(X=pd.concat([X, y], axis=1))
        result_gaussian = current_step.predict(X=pd.concat([X, y], axis=1))['predict']

        current_step = pgraph._step['Dbscan']
        result_dbscan = current_step.predict(X=pd.concat([X, y], axis=1))['predict']
        self.assertEqual(result_dbscan.min(), 0)

        current_step = pgraph._step['Combine_Clustering']
        current_step.fit(dominant=result_dbscan, other=result_gaussian)
        expected = current_step._strategy._adaptee.predict(dominant=result_dbscan, other=result_gaussian)
        result = current_step.predict(dominant=result_dbscan, other=result_gaussian)['predict']
        assert_array_equal(expected, result)
        self.assertEqual(result.min(), 0)

    def test_Pipegraph__strategy__dict_key(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._step['Concatenate_Xy']
        current_step.fit(df1=X, df2=y)
        result = current_step.predict(df1=X, df2=y)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_Pipegraph__dbscan__dict_key(self):
        X = self.X
        pgraph = self.pgraph
        current_step = pgraph._step['Dbscan']
        current_step.fit(X=X)
        result = current_step.predict(X=X)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_Pipegraph__combine_clustering__dict_key(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._step['Combine_Clustering']
        current_step.fit(dominant=X, other=y)
        result = current_step.predict(dominant=X, other=y)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_Pipegraph__gaussian_mixture__dict_key(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._step['Gaussian_Mixture']
        current_step.fit(X=X)
        result = current_step.predict(X=X)
        self.assertEqual(sorted(list(result.keys())), sorted(['predict', 'predict_proba']))

    def test_Pipegraph__regressor__dict_key(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._step['Regressor']
        current_step.fit(X=X, y=y)
        result = current_step.predict(X=X)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_Pipegraph__data_attributes_empty_at_init(self):
        pgraph = self.pgraph
        self.assertEqual(pgraph._fit_data, {})
        self.assertEqual(pgraph._predict_data, {})

    def test_Pipegraph__fit_node_names(self):
        pgraph = self.pgraph
        node_list = list(pgraph._fit_graph.nodes)
        self.assertEqual(sorted(node_list), sorted(['Concatenate_Xy',
                                                    'Gaussian_Mixture',
                                                    'Dbscan',
                                                    'Combine_Clustering',
                                                    'Paella',
                                                    'Regressor',
                                                    ]))

    def test_Pipegraph__predict_node_names(self):
        pgraph = self.pgraph
        node_list = list(pgraph._predict_graph.nodes)
        self.assertEqual(sorted(node_list), sorted(['Concatenate_Xy',
                                                    'Gaussian_Mixture',
                                                    'Dbscan',
                                                    'Combine_Clustering',
                                                    'Paella',
                                                    'Regressor',
                                                    ]))

    def test_Pipegraph__filter_nodes_fit(self):
        pgraph = self.pgraph
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

        pgraph = PipeGraph(steps=self.steps, connections=self.connections,
                           alternative_connections=alternative_connections)
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
        self.assertEqual(pgraph._step['Concatenate_Xy'].get_params(), {})
        self.assertEqual(pgraph._step['Gaussian_Mixture'].get_params()['n_components'], 3)
        self.assertEqual(pgraph._step['Dbscan'].get_params()['eps'], 0.5)
        self.assertEqual(pgraph._step['Combine_Clustering'].get_params(), {})
        self.assertEqual(pgraph._step['Paella'].get_params()['noise_label'], None)
        self.assertEqual(pgraph._step['Paella'].get_params()['max_it'], 10)
        self.assertEqual(pgraph._step['Paella'].get_params()['regular_size'], 100)
        self.assertEqual(pgraph._step['Paella'].get_params()['minimum_size'], 30)
        self.assertEqual(pgraph._step['Paella'].get_params()['width_r'], 0.95)
        self.assertEqual(pgraph._step['Paella'].get_params()['n_neighbors'], 1)
        self.assertEqual(pgraph._step['Paella'].get_params()['power'], 10)
        self.assertEqual(pgraph._step['Regressor'].get_params(),
                         {'copy_X': True, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False})

    def test_Pipegraph__step_set_params_multiple(self):
        pgraph = self.pgraph
        self.assertRaises(ValueError, pgraph._step['Concatenate_Xy'].set_params, ham=2, spam=9)
        self.assertEqual(pgraph._step['Gaussian_Mixture'].set_params(n_components=5).get_params()['n_components'],
                         5)
        self.assertEqual(pgraph._step['Dbscan'].set_params(eps=10.2).get_params()['eps'],
                         10.2)
        self.assertEqual(pgraph._step['Paella'].set_params(noise_label=-3).get_params()['noise_label'],
                         -3)
        self.assertEqual(pgraph._step['Regressor'].set_params(copy_X=False).get_params(),
                         {'copy_X': False, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False})
        self.assertEqual(pgraph._step['Regressor'].set_params(n_jobs=13).get_params(),
                         {'copy_X': False, 'fit_intercept': True, 'n_jobs': 13, 'normalize': False})

    def test_Pipegraph__named_steps(self):
        pgraph = self.pgraph
        self.assertEqual(pgraph.named_steps, dict(self.steps))
        self.assertEqual(len(pgraph.named_steps), 6)


class TestPaella(unittest.TestCase):
    def setUp(self):
        URL = "https://raw.githubusercontent.com/mcasl/PAELLA/master/data/sin_60_percent_noise.csv"
        data = pd.read_csv(URL, usecols=['V1', 'V2'])
        self.X = data[['V1']]
        self.y = data[['V2']]
        concatenator = CustomConcatenation()
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
        self.steps = steps
        self.connections = connections
        self.pgraph = PipeGraph(steps=steps, connections=connections)

    def test_Paella__init(self):
        logger.debug("Type of adaptee %s", type(self.pgraph._step['Paella']._strategy._adaptee))
        self.assertTrue(isinstance(self.pgraph._step['Paella']._strategy._adaptee, Paella))

    def test_Paella__under_fit__Paella__fit(self):
        pgraph = self.pgraph
        pgraph._fit_data = {('_External', 'X'): self.X,
                            ('_External', 'y'): self.y,
                            }
        pgraph._fit('Concatenate_Xy')
        pgraph._fit('Gaussian_Mixture')
        pgraph._fit('Dbscan')
        pgraph._fit('Combine_Clustering')

        paellador = pgraph._step['Paella']._strategy._adaptee

        X = pgraph._fit_data[('_External', 'X')]
        y = pgraph._fit_data[('_External', 'y')]
        classification = pgraph._fit_data[('Combine_Clustering', 'predict')]
        paellador.fit(X, y, classification)

    def test_Paella__under_predict__Paella__fit(self):
        pgraph = self.pgraph
        pgraph._fit_data = {('_External', 'X'): self.X,
                            ('_External', 'y'): self.y, }
        pgraph._fit('Concatenate_Xy')
        pgraph._fit('Gaussian_Mixture')
        pgraph._fit('Dbscan')
        pgraph._fit('Combine_Clustering')
        pgraph._fit('Paella')

        X = pgraph._fit_data[('_External', 'X')]
        y = pgraph._fit_data[('_External', 'y')]
        paellador = pgraph._step['Paella']._strategy._adaptee
        result = paellador.transform(X=X, y=y)
        self.assertEqual(result.shape[0], 10000)

    def test_Paella__get_params(self):
        pgraph = self.pgraph
        paellador = pgraph._step['Paella']
        result = paellador.get_params()['minimum_size']
        self.assertEqual(result, 30)

    def test_Paella__set_params(self):
        pgraph = self.pgraph
        paellador = pgraph._step['Paella']
        result_pre = paellador.get_params()['max_it']
        self.assertEqual(result_pre, 10)
        result_post = paellador.set_params(max_it=1000).get_params()['max_it']
        self.assertEqual(result_post, 1000)


class TestSingleNodeLinearModel(unittest.TestCase):
    def setUp(self):
        self.X = X = pd.DataFrame(dict(X=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        self.y = y = 2 * X
        lm = LinearRegression()
        steps = [('linear_model', lm)]
        connections = {'linear_model': dict(X='X', y='y')}
        self.lm = lm
        self.steps = steps
        self.connections = connections
        self.pgraph = PipeGraph(steps=steps, connections=connections)
        self.param_grid = dict(linear_model__fit_intercept=[False, True],
                               linear_model__normalize=[True, False])

    def test_single_node_fit(self):
        pgraph = self.pgraph
        self.assertFalse(hasattr(pgraph._step['linear_model'], 'coef_'))
        pgraph.fit(X=self.X, y=self.y)
        self.assertTrue(hasattr(pgraph._step['linear_model'], 'coef_'))
        self.assertAlmostEqual(pgraph._step['linear_model'].coef_[0][0], 2)

        self.assertEqual(pgraph._fit_data['linear_model', 'predict'].shape[0], 11)
        result = pgraph.predict(X=self.X)
        self.assertEqual(pgraph._fit_data['linear_model', 'predict'].shape[0], 11)
        self.assertEqual(result.shape[0], 11)

    def test_get_params(self):
        pgraph = self.pgraph
        result = pgraph.get_params()
        logger.debug("get_params result: %s", result)
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
        logger.debug("get_params result: %s", result_pre)
        expected_pre = {'linear_model': self.lm,
                        'linear_model__copy_X': True,
                        'linear_model__fit_intercept': True,
                        'linear_model__n_jobs': 1,
                        'linear_model__normalize': False,
                        'steps': self.steps}
        for item in expected_pre:
            self.assertTrue(item in result_pre)

        result_post = pgraph.set_params(linear_model__copy_X=False).get_params()
        logger.debug("set_params result: %s", result_post)
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
    #     self.assertEqual(pgraph.data['linear_model', 'predict'].shape[0], 10000)
    #     result = pgraph.predict(X=self.X)
    #     self.assertEqual(pgraph.data['linear_model', 'predict'].shape[0], 10000)
    #     self.assertEqual(result.shape[0], 10000)


class TestTwoNodes(unittest.TestCase):
    def setUp(self):
        self.X = X = pd.DataFrame(dict(X=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        self.y = y = 2 * X
        sc = MinMaxScaler()
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
        self.pgraph = PipeGraph(steps=steps,
                                connections=connections)
        self.param_grid = dict(linear_model__fit_intercept=[False, True],
                               linear_model__normalize=[True, False],
                               )

    def test_TwoNodes_fit(self):
        pgraph = self.pgraph
        self.assertFalse(hasattr(pgraph._step['linear_model'], 'coef_'))
        pgraph.fit(X=self.X, y=self.y)
        self.assertTrue(hasattr(pgraph._step['linear_model'], 'coef_'))
        self.assertAlmostEqual(pgraph._step['linear_model'].coef_[0][0], 20)

        self.assertEqual(pgraph._fit_data['linear_model', 'predict'].shape[0], 11)
        result = pgraph.predict(X=self.X)
        self.assertEqual(pgraph._fit_data['linear_model', 'predict'].shape[0], 11)
        self.assertEqual(result.shape[0], 11)


class TestTrainTestSplit(unittest.TestCase):
    def setUp(self):
        self.X = [1, 2, 3, 4, 5, 6, 7, 8]
        self.y = [101, 200, 300, 400, 500, 600, 700, 800]

    def test_train_test_predict(self):
        tts = TrainTestSplit()
        step = make_step(tts)
        result = step.predict(X=self.X, y=self.y)
        self.assertEqual(len(result), 4)
        self.assertEqual(sorted(list(result.keys())),
                         sorted(['X_train', 'X_test', 'y_train', 'y_test']))
        self.assertEqual(len(result['X_train']), 6)
        self.assertEqual(len(result['X_test']), 2)
        self.assertEqual(len(result['y_train']), 6)
        self.assertEqual(len(result['y_test']), 2)


if __name__ == '__main__':
    unittest.main()
