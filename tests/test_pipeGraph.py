import unittest

import numpy as np
import pandas as pd
from argparse import Namespace

from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal
from sklearn.cluster import DBSCAN
# from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import MinMaxScaler

from paella import *
from pipeGraph import *


class TestPipeGraphCase(unittest.TestCase):
    def setUp(self):
        URL = "https://raw.githubusercontent.com/mcasl/PAELLA/master/data/sin_60_percent_noise.csv"
        # URL = 'https://raw.githubusercontent.com/mcasl/PAELLA/master/data/sinusoidal_data.csv'

        data = pd.read_csv(URL, usecols=['V1', 'V2'])
        X = data[['V1']]
        y = data[['V2']]
        concatenator = CustomConcatenation()
        gaussian_clustering = GaussianMixture(n_components=3)
        dbscan = DBSCAN(eps=0.5)
        mixer = CustomCombination()
        paellaModel = CustomPaella(regressor=LinearRegression,
                                   noise_label=None,
                                   max_it=1,
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
            'First': {'X': X,
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

        self.paella_1 = Namespace(X=X, y=y, graph=graph)

    # ############ PAELLA 1 #######################

    # STRATEGY TESTS


    def test_stepstrategy__init(self):
        lm = LinearRegression()
        stepstrategy = FitPredictStrategy(lm)
        self.assertEqual(stepstrategy._adaptee, lm)

    def test_stepstrategy__fit_lm(self):
        X = self.paella_1.X
        y = self.paella_1.y
        lm = LinearRegression()
        stepstrategy = FitPredictStrategy(lm)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'coef_'), False)
        result = stepstrategy.fit(X=X, y=y)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'coef_'), True)
        self.assertEqual(stepstrategy, result)

    def test_stepstrategy__fit_dbscan(self):
        X = self.paella_1.X
        y = self.paella_1.y
        db = DBSCAN()
        stepstrategy = AtomicFitPredictStrategy(db)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'core_sample_indices_'), False)
        result_fit = stepstrategy.fit(X=X)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'core_sample_indices_'), False)
        self.assertEqual(stepstrategy, result_fit)
        result_predict = stepstrategy.predict(X=X)['predict']
        self.assertEqual(hasattr(stepstrategy._adaptee, 'core_sample_indices_'), True)
        self.assertEqual(result_predict.shape, (10000,))

    def test_stepstrategy__predict(self):
        X = self.paella_1.X
        y = self.paella_1.y
        lm = LinearRegression()
        gm = GaussianMixture()
        lm_strategy = FitPredictStrategy(lm)
        gm_strategy = FitPredictStrategy(gm)

        lm_strategy.fit(X=X, y=y)
        gm_strategy.fit(X=X)

        result_lm = lm_strategy.predict(X=X)
        result_gm = gm_strategy.predict(X=X)

        self.assertEqual(list(result_lm.keys()), ['predict'])
        self.assertEqual(list(result_gm.keys()).sort(), ['predict', 'predict_proba'].sort())

    def test_stepstrategy__get_fit_parameters_from_signature(self):
        lm = LinearRegression()
        gm = GaussianMixture()
        lm_strategy = FitPredictStrategy(lm)
        gm_strategy = FitPredictStrategy(gm)

        result_lm = lm_strategy.get_fit_parameters_from_signature()
        result_gm = gm_strategy.get_fit_parameters_from_signature()

        self.assertEqual(result_lm, ['X', 'y', 'sample_weight'])
        self.assertEqual(result_gm, ['X', 'y'])

    def test_stepstrategy__get_predict_parameters_from_signature_lm(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
        result_lm = lm_strategy.get_predict_parameters_from_signature()
        self.assertEqual(result_lm, ['X'])

    def test_stepstrategy__get_predict_parameters_from_signature_sc(self):
        sc = MinMaxScaler()
        sc_strategy = FitTransformStrategy(sc)
        result_sc = sc_strategy.get_predict_parameters_from_signature()
        self.assertEqual(result_sc, ['X'])

    def test_stepstrategy__get_predict_parameters_from_signature_db(self):
        db = DBSCAN()
        db_strategy = AtomicFitPredictStrategy(db)
        result_db = db_strategy.get_predict_parameters_from_signature()
        self.assertEqual(result_db, ['X', 'y', 'sample_weight'])

    def test_stepstrategy__get_params(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
        result_lm = lm_strategy.get_params()
        self.assertEqual(result_lm, {'copy_X': True,
                                     'fit_intercept': True,
                                     'n_jobs': 1,
                                     'normalize': False})

    def test_stepstrategy__set_params(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
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
        lm_strategy = FitPredictStrategy(lm)
        self.assertEqual(lm_strategy.copy_X, True)

    def test_stepstrategy__setattr__(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
        self.assertEqual(lm_strategy.copy_X, True)
        lm_strategy.copy_X = False
        self.assertEqual(lm_strategy.copy_X, False)
        self.assertEqual('copy_X' in dir(lm_strategy), False)

    def test_stepstrategy__delattr__(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
        self.assertEqual(lm_strategy.copy_X, True)
        self.assertEqual('copy_X' in dir(lm_strategy), False)
        self.assertEqual('copy_X' in dir(lm), True)
        del lm_strategy.copy_X
        self.assertEqual('copy_X' in dir(lm), False)


# ###########################
# STEP TESTS
    def test_step__init(self):
        lm = LinearRegression()
        stepstrategy = FitPredictStrategy(lm)
        step = Step(stepstrategy)
        self.assertEqual(step._strategy, stepstrategy)

    def test_step__fit_lm(self):
        X = self.paella_1.X
        y = self.paella_1.y
        lm = LinearRegression()
        stepstrategy = FitPredictStrategy(lm)
        step = Step(stepstrategy)
        self.assertEqual(hasattr(step, 'coef_'), False)
        result = step.fit(X=X, y=y)
        self.assertEqual(hasattr(step, 'coef_'), True)
        self.assertEqual(step, result)

    def test_step__fit_dbscan(self):
        X = self.paella_1.X
        y = self.paella_1.y
        db = DBSCAN()
        stepstrategy = AtomicFitPredictStrategy(db)
        step = Step(stepstrategy)
        self.assertEqual(hasattr(step, 'core_sample_indices_'), False)
        result_fit = step.fit(X=X)
        self.assertEqual(hasattr(step, 'core_sample_indices_'), False)
        self.assertEqual(step, result_fit)
        result_predict = step.predict(X=X)['predict']
        self.assertEqual(hasattr(step, 'core_sample_indices_'), True)
        self.assertEqual(result_predict.shape, (10000,))

    def test_step__get_params(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
        step = Step(lm_strategy)
        result_lm = step.get_params()
        self.assertEqual(result_lm, {'copy_X': True,
                                     'fit_intercept': True,
                                     'n_jobs': 1,
                                     'normalize': False})

    def test_step__set_params(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
        step = Step(lm_strategy)
        result_lm_pre = step.get_params()
        self.assertEqual(result_lm_pre, {'copy_X': True,
                                         'fit_intercept': True,
                                         'n_jobs': 1,
                                         'normalize': False})
        lm_strategy.set_params(fit_intercept=False)
        result_lm_post = step.get_params()
        self.assertEqual(result_lm_post, {'copy_X': True,
                                          'fit_intercept': False,
                                          'n_jobs': 1,
                                          'normalize': False})


    # # def fit(self, **kwargs):
    # #     self._strategy.fit(**kwargs)
    # #     return self
    # #
    # # def predict(self, **kwargs):
    # #     return self._strategy.predict(**kwargs)
    # #
    # # def get_fit_parameters_from_signature(self):
    # #     return self._strategy.get_fit_parameters_from_signature()
    # #
    # # def get_predict_parameters_from_signature(self):
    # #     return self._strategy.get_predict_parameters_from_signature()
    # #

    def test_step__getattr__(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
        step = Step(lm_strategy)
        self.assertEqual(step.copy_X, True)

    def test_step__setattr__(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
        step = Step(lm_strategy)
        self.assertEqual(step.copy_X, True)
        step.copy_X = False
        self.assertEqual(step.copy_X, False)
        self.assertEqual('copy_X' in dir(step), False)

    def test_step__delattr__(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
        step = Step(lm_strategy)
        self.assertEqual(step.copy_X, True)
        self.assertEqual('copy_X' in dir(step), False)
        self.assertEqual('copy_X' in dir(lm), True)
        del step.copy_X
        self.assertEqual('copy_X' in dir(lm), False)







# #########################





    def test_make_step(self):
        tests_table = [
            {'model': LinearRegression(),
             'expected_class': FitPredictStrategy},

            {'model': MinMaxScaler(),
             'expected_class': FitTransformStrategy},

            {'model': DBSCAN(),
             'expected_class': AtomicFitPredictStrategy},
        ]

        for test_dict in tests_table:
            model = test_dict['model']
            step = make_step(model)
            self.assertEqual(isinstance(step._strategy, test_dict['expected_class']), True)

    def test_step_predict_lm(self):
        X = self.paella_1.X
        y = self.paella_1.y
        lm = LinearRegression()
        lm_step = make_step(lm)
        lm_step.fit(X=X, y=y)
        assert_array_equal(lm.predict(X), lm_step.predict(X=X)['predict'])

    def test_concatenate_Xy_predict(self):
        X = self.paella_1.X
        y = self.paella_1.y
        pgraph = self.paella_1.graph
        expected = pd.concat([X, y], axis=1)
        current_step = pgraph._graph.node['Concatenate_Xy']['step']
        current_step.fit(df1=X, df2=y)
        result = current_step.predict(df1=X, df2=y)['predict']
        assert_frame_equal(expected, result)

    def test_gaussian_mixture_predict(self):
        X = self.paella_1.X
        pgraph = self.paella_1.graph
        current_step = pgraph._graph.node['Gaussian_Mixture']['step']
        current_step.fit(X=X)
        expected = current_step._strategy._adaptee.predict(X=X)
        result = current_step.predict(X=X)['predict']
        assert_array_equal(expected, result)

    def test_dbscan_predict(self):
        X = self.paella_1.X
        pgraph = self.paella_1.graph
        current_step = pgraph._graph.node['Dbscan']['step']
        current_step.fit(X=X)
        expected = current_step._strategy._adaptee.fit_predict(X=X)
        result = current_step.predict(X=X)['predict']
        assert_array_equal(expected, result)

    # def test_combine_clustering_predict(self):
    #     X = self.X
    #     y = self.y
    #     current_step = self.graph._graph.node['Gaussian_Mixture']['step']
    #     current_step.fit(X=pd.concat([X,y], axis=1))
    #     result_gaussian = current_step.predict(X=pd.concat([X,y], axis=1))['predict']
    #
    #     current_step = self.graph._graph.node['Dbscan']['step']
    #     result_dbscan = current_step.predict(X=pd.concat([X,y], axis=1))['predict']
    #     self.assertEqual(result_dbscan.min(), -1)
    #
    #     current_step = self.graph._graph.node['Combine_Clustering']['step']
    #     current_step.fit(dominant=result_dbscan, other=result_gaussian)
    #     expected = current_step.strategy.adaptee.predict(dominant=result_dbscan, other=result_gaussian)
    #     result = current_step.predict(dominant=result_dbscan, other=result_gaussian)['predict']
    #     assert_array_equal(expected, result)
    #     self.assertEqual(result.min(), -1)

    def test_strategy_dict_key(self):
        X = self.paella_1.X
        y = self.paella_1.y
        pgraph = self.paella_1.graph
        current_step = pgraph._graph.node['Concatenate_Xy']['step']
        current_step.fit(df1=X, df2=y)
        result = current_step.predict(df1=X, df2=y)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_dbscan_dict_key(self):
        X = self.paella_1.X
        y = self.paella_1.y
        pgraph = self.paella_1.graph
        current_step = pgraph._graph.node['Dbscan']['step']
        current_step.fit(X=X)
        result = current_step.predict(X=X)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_combine_clustering_dict_key(self):
        X = self.paella_1.X
        y = self.paella_1.y
        pgraph = self.paella_1.graph
        current_step = pgraph._graph.node['Combine_Clustering']['step']
        current_step.fit(dominant=X, other=y)
        result = current_step.predict(dominant=X, other=y)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_gaussian_mixture_dict_key(self):
        X = self.paella_1.X
        y = self.paella_1.y
        pgraph = self.paella_1.graph
        current_step = pgraph._graph.node['Gaussian_Mixture']['step']
        current_step.fit(X=X)
        result = current_step.predict(X=X)
        self.assertEqual(list(result.keys()), ['predict', 'predict_proba'])

    def test_regressor_dict_key(self):
        X = self.paella_1.X
        y = self.paella_1.y
        pgraph = self.paella_1.graph
        current_step = pgraph._graph.node['Regressor']['step']
        current_step.fit(X=X, y=y)
        result = current_step.predict(X=X)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_FirstNode_dict_key(self):
        X = self.paella_1.X
        y = self.paella_1.y
        pgraph = self.paella_1.graph
        current_step = pgraph._graph.node['First']['step']
        current_step.fit(X=X, y=y)
        result = current_step.predict(X=X)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_data_attribute(self):
        pgraph = self.paella_1.graph
        self.assertEqual(dict(), pgraph.data)

    def test_node_names(self):
        pgraph = self.paella_1.graph
        node_list = list(pgraph._graph.nodes)
        self.assertEqual(node_list, ['First',
                                     'Concatenate_Xy',
                                     'Gaussian_Mixture',
                                     'Dbscan',
                                     'Combine_Clustering',
                                     'Paella',
                                     'Regressor',
                                     ])

    def test_filter_nodes_fit(self):
        pgraph = self.paella_1.graph
        fit_nodes = [item['name'] for item in pgraph._filter_nodes(filter='fit')]
        self.assertEqual(fit_nodes, ['First',
                                     'Concatenate_Xy',
                                     'Dbscan',
                                     'Gaussian_Mixture',
                                     'Combine_Clustering',
                                     'Paella',
                                     'Regressor',
                                     ])

    def test_filter_nodes_predict(self):
        pgraph = self.paella_1.graph
        predict_nodes = [item['name'] for item in pgraph._filter_nodes(filter='predict')]
        self.assertEqual(predict_nodes, ['First',
                                         'Regressor',
                                         ])

    # def test_graph_fit(self):
    #    self.graph.fit()
    #     assert_frame_equal(self.graph.data['First', 'X'], self.X)
    #     assert_frame_equal(self.graph.data['First', 'y'], self.y)
    #
    #     self.assertEqual(self.graph.data['Dbscan', 'prediction'].shape[0], self.y.shape[0])
    #     self.assertEqual(self.graph.data['Dbscan', 'prediction'].min(), -1)
    #
    #     self.assertEqual(self.graph.data['Gaussian_Mixture', 'prediction'].shape[0], self.y.shape[0])
    #     self.assertEqual(self.graph.data['Gaussian_Mixture', 'prediction'].min(), 0)
    #
    #     self.assertEqual(self.graph.data['Combine_Clustering', 'classification'].shape[0], self.y.shape[0])
    #     self.assertEqual(self.graph.data['Combine_Clustering', 'classification'].min(), -1)
    #     self.assertEqual(self.graph.data['Combine_Clustering', 'classification'].max(),
    #                      self.graph.data['Gaussian_Mixture', 'prediction'].max())
    #
    #     self.assertEqual(self.graph.data['Paella', 'prediction'].shape[0], self.y.shape[0])
    #     self.assertEqual(self.graph.data['Paella', 'prediction'].min(), 0)
    #     self.assertEqual(self.graph.data['Paella', 'prediction'].max(), 1)
    #
    #     self.assertEqual(self.graph.data['Regressor', 'prediction'].shape[0], self.y.shape[0])
    #
    #     assert_array_equal(self.graph.data['Last', 'prediction'],
    #                        self.graph.data['Regressor', 'prediction'])
    #
    # def test_graph_predict(self):
    #     self.graph.fit()
    #     self.graph.predict()
    #
    #     assert_frame_equal(self.graph.data['First', 'X'], self.X)
    #     assert_frame_equal(self.graph.data['First', 'y'], self.y)
    #
    #     self.assertEqual(self.graph.data['Regressor', 'prediction'].shape[0], self.y.shape[0])
    #
    #     assert_array_equal(self.graph.data['Last', 'prediction'],
    #                        self.graph.data['Regressor', 'prediction'])
    #

    def test_step_get_params_multiple(self):
        pgraph = self.paella_1.graph
        graph = pgraph._graph
        self.assertEqual(graph.node['First']['step'].get_params(), {})
        self.assertEqual(graph.node['Concatenate_Xy']['step'].get_params(), {})
        self.assertEqual(graph.node['Gaussian_Mixture']['step'].get_params()['n_components'], 3)
        self.assertEqual(graph.node['Dbscan']['step'].get_params()['eps'], 0.5)
        self.assertEqual(graph.node['Combine_Clustering']['step'].get_params(), {})
        # self.assertEqual(graph.node['Paella']['step'].get_params()['noise_label'], -1)
        # self.assertEqual(graph.node['Paella']['step'].get_params()['max_it'], 20)
        # self.assertEqual(graph.node['Paella']['step'].get_params()['regular_size'], 400)
        # self.assertEqual(graph.node['Paella']['step'].get_params()['minimum_size'], 100)
        # self.assertEqual(graph.node['Paella']['step'].get_params()['width_r'], 0.99)
        # self.assertEqual(graph.node['Paella']['step'].get_params()['n_neighbors'], 5)
        # self.assertEqual(graph.node['Paella']['step'].get_params()['power'], 30)
        self.assertEqual(graph.node['Regressor']['step'].get_params(),
                         {'copy_X': True, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False})

    def test_step_set_params_multiple(self):
        pgraph = self.paella_1.graph
        graph = pgraph._graph
        self.assertRaises(ValueError, graph.node['First']['step'].set_params, ham=902)
        self.assertRaises(ValueError, graph.node['Concatenate_Xy']['step'].set_params, ham=2, spam=9)
        self.assertEqual(graph.node['Gaussian_Mixture']['step'].set_params(n_components=5).get_params()['n_components'],
                         5)
        self.assertEqual(graph.node['Dbscan']['step'].set_params(eps=10.2).get_params()['eps'],
                         10.2)
        # self.assertEqual(graph.node['Paella']['step'].set_params(noise_label=-3).get_params()['noise_label'],
        #                  -3)
        self.assertEqual(graph.node['Regressor']['step'].set_params(copy_X=False).get_params(),
                         {'copy_X': False, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False})
        self.assertEqual(graph.node['Regressor']['step'].set_params(n_jobs=13).get_params(),
                         {'copy_X': False, 'fit_intercept': True, 'n_jobs': 13, 'normalize': False})


if __name__ == '__main__':
    unittest.main()
