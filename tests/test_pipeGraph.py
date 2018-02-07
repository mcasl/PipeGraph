import unittest

from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

from paella import *
from pipeGraph import *


class TestStepStrategy(unittest.TestCase):
    def setUp(self):
        URL = "https://raw.githubusercontent.com/mcasl/PAELLA/master/data/sin_60_percent_noise.csv"
        data = pd.read_csv(URL, usecols=['V1', 'V2'])
        self.X = data[['V1']]
        self.y = data[['V2']]

    def test_stepstrategy__init(self):
        lm = LinearRegression()
        stepstrategy = FitPredictStrategy(lm)
        self.assertEqual(stepstrategy._adaptee, lm)

    def test_stepstrategy__fit_FitPredict(self):
        X = self.X
        y = self.y
        lm = LinearRegression()
        stepstrategy = FitPredictStrategy(lm)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'coef_'), False)
        result = stepstrategy.fit(X=X, y=y)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'coef_'), True)
        self.assertEqual(stepstrategy, result)

    def test_stepstrategy__fit_FitTransform(self):
        X = self.X
        sc = MinMaxScaler()
        stepstrategy = FitTransformStrategy(sc)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'n_samples_seen_'), False)
        result = stepstrategy.fit(X=X)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'n_samples_seen_'), True)
        self.assertEqual(stepstrategy, result)

    def test_stepstrategy__fit_AtomicFitPredict(self):
        X = self.X
        db = DBSCAN()
        stepstrategy = AtomicFitPredictStrategy(db)
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
        lm_strategy = FitPredictStrategy(lm)
        lm_strategy.fit(X=X, y=y)
        result_lm = lm_strategy.predict(X=X)
        self.assertEqual(list(result_lm.keys()), ['predict'])
        self.assertEqual(result_lm['predict'].shape, (10000, 1))

        gm = GaussianMixture()
        gm_strategy = FitPredictStrategy(gm)
        gm_strategy.fit(X=X)
        result_gm = gm_strategy.predict(X=X)
        self.assertEqual(list(result_gm.keys()).sort(), ['predict', 'predict_proba'].sort())
        self.assertEqual(result_gm['predict'].shape, (10000,))

    def test_stepstrategy__predict_FitTransform(self):
        X = self.X
        sc = MinMaxScaler()
        sc_strategy = FitTransformStrategy(sc)
        sc_strategy.fit(X=X)
        result_sc = sc_strategy.predict(X=X)
        self.assertEqual(list(result_sc.keys()), ['predict'])
        self.assertEqual(result_sc['predict'].shape, (10000, 1))

    def test_stepstrategy__predict_AtomicFitPredict(self):
        X = self.X
        y = self.y
        db = DBSCAN()
        db_strategy = AtomicFitPredictStrategy(db)
        db_strategy.fit(X=X, y=y)
        result_db = db_strategy.predict(X=X)
        self.assertEqual(list(result_db.keys()), ['predict'])
        self.assertEqual(result_db['predict'].shape, (10000,))

    def test_stepstrategy__get_fit_parameters_from_signature(self):
        lm = LinearRegression()
        gm = GaussianMixture()
        lm_strategy = FitPredictStrategy(lm)
        gm_strategy = FitPredictStrategy(gm)

        result_lm = lm_strategy._get_fit_parameters_from_signature()
        result_gm = gm_strategy._get_fit_parameters_from_signature()

        self.assertEqual(result_lm, ['X', 'y', 'sample_weight'])
        self.assertEqual(result_gm, ['X', 'y'])

    def test_stepstrategy__get_predict_parameters_from_signature_FitPredict(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
        result_lm = lm_strategy._get_predict_parameters_from_signature()
        self.assertEqual(result_lm, ['X'])

    def test_stepstrategy__get_predict_parameters_from_signature_FitTransform(self):
        sc = MinMaxScaler()
        sc_strategy = FitTransformStrategy(sc)
        result_sc = sc_strategy._get_predict_parameters_from_signature()
        self.assertEqual(result_sc, ['X'])

    def test_stepstrategy__get_predict_parameters_from_signature_AtomicFitPredict(self):
        db = DBSCAN()
        db_strategy = AtomicFitPredictStrategy(db)
        result_db = db_strategy._get_predict_parameters_from_signature()
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

    def test_stepstrategy__repr__(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
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
        stepstrategy = FitPredictStrategy(lm)
        step = Step(stepstrategy)
        self.assertEqual(step._strategy, stepstrategy)

    def test_step__fit_lm(self):
        X = self.X
        y = self.y
        lm = LinearRegression()
        stepstrategy = FitPredictStrategy(lm)
        step = Step(stepstrategy)
        self.assertEqual(hasattr(step, 'coef_'), False)
        result = step.fit(X=X, y=y)
        self.assertEqual(hasattr(step, 'coef_'), True)
        self.assertEqual(step, result)

    def test_step__fit_dbscan(self):
        X = self.X
        y = self.y
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

    def test_step__predict(self):
        X = self.X
        y = self.y
        lm = LinearRegression()
        gm = GaussianMixture()
        lm_strategy = FitPredictStrategy(lm)
        gm_strategy = FitPredictStrategy(gm)
        step_lm = Step(lm_strategy)
        step_gm = Step(gm_strategy)

        step_lm.fit(X=X, y=y)
        step_gm.fit(X=X)

        result_lm = step_lm.predict(X=X)
        result_gm = step_gm.predict(X=X)

        self.assertEqual(list(result_lm.keys()), ['predict'])
        self.assertEqual(list(result_gm.keys()).sort(), ['predict', 'predict_proba'].sort())

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
        step.set_params(fit_intercept=False)
        result_lm_post = step.get_params()
        self.assertEqual(result_lm_post, {'copy_X': True,
                                          'fit_intercept': False,
                                          'n_jobs': 1,
                                          'normalize': False})

    def test_step__get_fit_parameters_from_signature(self):
        lm = LinearRegression()
        gm = GaussianMixture()
        lm_strategy = FitPredictStrategy(lm)
        gm_strategy = FitPredictStrategy(gm)
        step_lm = Step(lm_strategy)
        step_gm = Step(gm_strategy)
        result_lm = step_lm._get_fit_parameters_from_signature()
        result_gm = step_gm._get_fit_parameters_from_signature()

        self.assertEqual(result_lm, ['X', 'y', 'sample_weight'])
        self.assertEqual(result_gm, ['X', 'y'])

    def test_step__get_predict_parameters_from_signature_lm(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
        step_lm = Step(lm_strategy)
        result_lm = step_lm._get_predict_parameters_from_signature()
        self.assertEqual(result_lm, ['X'])

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

    def test_step__repr__(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
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

            'Paella': dict(X='X', y='y', classification=('Combine_Clustering', 'classification')),

            'Regressor': dict(X='X', y='y', sample_weight=('Paella', 'predict'))
        }


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

    def test_make_step_raise_exception(self):
        model = object()
        self.assertRaises(ValueError, make_step, model)

    def test_build_graph_node_names(self):
        graph = build_graph(self.steps, self.connections)

        node_list = list(graph.nodes)
        self.assertEqual(node_list, ['Concatenate_Xy',
                                     'Gaussian_Mixture',
                                     'Dbscan',
                                     'Combine_Clustering',
                                     'Paella',
                                     'Regressor',
                                     ])


    def test_build_graph_node_edges(self):
        graph = build_graph(self.steps, self.connections)

        in_edges = {name: [origin for origin, destination in list(graph.in_edges(name))]
                    for name in graph}



        self.assertEqual(in_edges['Gaussian_Mixture'], ['Concatenate_Xy'])
        self.assertEqual(in_edges['Dbscan'], ['Concatenate_Xy'])
        self.assertEqual(in_edges['Combine_Clustering'].sort(), ['Gaussian_Mixture', 'Dbscan'].sort())
      #  self.assertEqual(in_edges['Paella'].sort(), ['Combine_Clustering'])
      #  self.assertEqual(in_edges['Regressor'].sort(), ['Paella'])


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
        self.steps = steps
        self.connections = connections
        self.pgraph = PipeGraph(steps=steps,
                                connections=connections,
                                use_for_fit='all',
                                use_for_predict=['Regressor'])

    def test_Pipegraph__use_for_fit_all(self):
        pgraph = PipeGraph(self.steps,
                           self.connections,
                           use_for_fit='all',
                           use_for_predict='all')

        fit_nodes_list = [name for name in pgraph._graph.nodes
                          if 'fit' in pgraph._use_for[name]]
        self.assertEqual(fit_nodes_list.sort(), ['Concatenate_Xy',
                                                 'Gaussian_Mixture',
                                                 'Dbscan',
                                                 'Combine_Clustering',
                                                 'Paella',
                                                 'Regressor',
                                                 ].sort())

    def test_Pipegraph__use_for_fit_some(self):
        pgraph = PipeGraph(self.steps,
                          self.connections,
                          use_for_fit=['Concatenate_Xy',
                                       'Gaussian_Mixture',
                                       'Dbscan',
                                       ],
                          use_for_predict='all')

        fit_nodes_list = [name for name in pgraph._graph.nodes
                          if 'fit' in pgraph._use_for[name]]
        self.assertEqual(fit_nodes_list.sort(), ['Concatenate_Xy',
                                                 'Gaussian_Mixture',
                                                 'Dbscan',
                                                 ].sort())

    def test_Pipegraph__use_for_fit_one(self):
        pgraph = PipeGraph(self.steps,
                          self.connections,
                          use_for_fit=['Concatenate_Xy'],
                          use_for_predict='all')

        fit_nodes_list = [name for name in pgraph._graph.nodes
                          if 'fit' in pgraph._use_for[name]]
        self.assertEqual(fit_nodes_list.sort(), ['Concatenate_Xy'].sort())

    def test_Pipegraph__use_for_fit_Empty(self):
        pgraph = PipeGraph(self.steps,
                           self.connections,
                           use_for_fit=[],
                           use_for_predict='all')
        graph = pgraph._graph
        fit_nodes_list = [name for name in pgraph._graph.nodes
                          if 'fit' in pgraph._use_for[name]]
        self.assertEqual(fit_nodes_list, [])

    def test_Pipegraph__use_for_fit_None(self):
        pgraph = PipeGraph(self.steps,
                          self.connections,
                          use_for_fit=None,
                          use_for_predict='all')

        fit_nodes_list = [name for name in pgraph._graph.nodes
                          if 'fit' in pgraph._use_for[name]]
        self.assertEqual(fit_nodes_list, [])

    def test_Pipegraph__use_for_fit_External(self):
        pgraph = PipeGraph(self.steps,
                          self.connections,
                          use_for_fit=['First'],
                          use_for_predict='all')

        fit_nodes_list = [name for name in pgraph._graph.nodes
                          if 'fit' in pgraph._use_for[name]]
        self.assertEqual(fit_nodes_list, [])

    def test_Pipegraph__use_for_predict_all(self):
        pgraph = PipeGraph(self.steps,
                          self.connections,
                          use_for_fit='all',
                          use_for_predict='all')

        predict_nodes_list = [name for name in pgraph._graph.nodes
                              if 'predict' in pgraph._use_for[name]]
        self.assertEqual(predict_nodes_list.sort(), ['Concatenate_Xy',
                                                     'Gaussian_Mixture',
                                                     'Dbscan',
                                                     'Combine_Clustering',
                                                     'Paella',
                                                     'Regressor',
                                                     ].sort())

    def test_Pipegraph__use_for_predict_some(self):
        pgraph = PipeGraph(self.steps,
                          self.connections,
                          use_for_fit='all',
                          use_for_predict=['Concatenate_Xy',
                                           'Gaussian_Mixture',
                                           'Dbscan',
                                           ])

        predict_nodes_list = [name for name in pgraph._graph.nodes
                              if 'predict' in pgraph._use_for[name]]
        self.assertEqual(predict_nodes_list.sort(), ['Concatenate_Xy',
                                                     'Gaussian_Mixture',
                                                     'Dbscan',
                                                     ].sort())

    def test_Pipegraph__use_for_predict_one(self):
        pgraph = PipeGraph(self.steps,
                          self.connections,
                          use_for_predict=['Concatenate_Xy'],
                          use_for_fit='all')

        predict_nodes_list = [name for name in pgraph._graph.nodes
                              if 'predict' in pgraph._use_for[name]]
        self.assertEqual(predict_nodes_list.sort(), ['Concatenate_Xy'].sort())

    def test_Pipegraph__use_for_predict_Empty(self):
        pgraph = PipeGraph(self.steps,
                          self.connections,
                          use_for_predict=[],
                          use_for_fit='all')

        predict_nodes_list = [name for name in pgraph._graph.nodes
                              if 'predict' in pgraph._use_for[name]]
        self.assertEqual(predict_nodes_list, [])

    def test_Pipegraph__use_for_predict_None(self):
        pgraph = PipeGraph(self.steps,
                           self.connections,
                           use_for_predict=None,
                           use_for_fit='all')
        predict_nodes_list = [name for name in pgraph._graph.nodes
                              if 'predict' in pgraph._use_for[name]]
        self.assertEqual(predict_nodes_list, [])

    def test_Pipegraph__read_data(self):
        pgraph = self.pgraph
        input_data = {'X': self.X,
                      'y': self.y,
                      'z': 57,
                      }

        fit_variables = pgraph._step['Regressor']._get_fit_parameters_from_signature()
        result = pgraph._filter_data(input_data, fit_variables)

        self.assertEqual(list(result.keys()).sort(), ['X', 'sample_weight', 'y'].sort())
        assert_array_equal(result['X'], self.X)
        assert_array_equal(result['y'], self.y)
        self.assertTrue(result['sample_weight'] is None)

    def test_Pipegraph__read_connections_regressor_node(self):
        pgraph = self.pgraph
        pgraph.data = {('_External', 'X'): self.X,
                       ('_External', 'y'): self.y,
                       ('Paella', 'predict'): self.y * 2,
                       ('Dbscan', 'predict'): self.y * 4,
                       }

        result = pgraph._read_from_connections('Regressor')
        assert_array_equal(result['X'], self.X)
        assert_array_equal(result['y'], self.y)
        assert_array_equal(result['sample_weight'], self.y * 2)
        self.assertEqual(len(result), 3)

    def test_Pipegraph__step__predict_lm(self):
        X = self.X
        y = self.y
        lm = LinearRegression()
        lm_step = make_step(lm)
        lm_step.fit(X=X, y=y)
        assert_array_equal(lm.predict(X), lm_step.predict(X=X)['predict'])

    def test_Pipegraph__under_fit__concatenate_Xy(self):
        pgraph = self.pgraph
        pgraph.data = {('_External', 'X'): self.X,
                       ('_External', 'y'): self.y,
                       ('Paella', 'predict'): self.y * 2,
                       ('Dbscan', 'predict'): self.y * 4,
                       }
        expected = pd.concat([self.X, self.y], axis=1)
        pgraph._fit('Concatenate_Xy')
        assert_frame_equal(expected, pgraph.data['Concatenate_Xy', 'predict'])

    def test_Pipegraph__under_fit__Paella__fit(self):
        pgraph = self.pgraph
        pgraph.data = {('_External', 'X'): self.X,
                       ('_External', 'y'): self.y,
                       }
        pgraph._fit('Concatenate_Xy')
        pgraph._fit('Gaussian_Mixture')
        pgraph._fit('Dbscan')
        pgraph._fit('Combine_Clustering')
        pgraph._fit('Paella')


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
        self.assertEqual(list(result.keys()), ['predict', 'predict_proba'])

    def test_Pipegraph__regressor__dict_key(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._step['Regressor']
        current_step.fit(X=X, y=y)
        result = current_step.predict(X=X)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_Pipegraph__data_attribute(self):
        pgraph = self.pgraph
        self.assertEqual(dict(), pgraph.data)

    def test_Pipegraph__node_names(self):
        pgraph = self.pgraph
        node_list = list(pgraph._graph.nodes)
        self.assertEqual(node_list, ['Concatenate_Xy',
                                     'Gaussian_Mixture',
                                     'Dbscan',
                                     'Combine_Clustering',
                                     'Paella',
                                     'Regressor',
                                     ])

    def test_Pipegraph__filter_nodes_fit(self):
        pgraph = self.pgraph
        fit_nodes = list(pgraph._filter_nodes(filter='fit'))
        self.assertEqual(fit_nodes, ['Concatenate_Xy',
                                     'Dbscan',
                                     'Gaussian_Mixture',
                                     'Combine_Clustering',
                                     'Paella',
                                     'Regressor',
                                     ])

    def test_Pipegraph__filter_nodes_predict(self):
        pgraph = self.pgraph
        predict_nodes = list(pgraph._filter_nodes(filter='predict'))
        self.assertEqual(predict_nodes, ['Regressor' ])

    def test_Pipegraph__graph_fit(self):
        pgraph = self.pgraph
        pgraph.fit(X=self.X, y=self.y)
        assert_frame_equal(pgraph.data['_External', 'X'], self.X)
        assert_frame_equal(pgraph.data['_External', 'y'], self.y)

        self.assertEqual(pgraph.data['Dbscan', 'predict'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph.data['Dbscan', 'predict'].min(), 0)

        self.assertEqual(pgraph.data['Gaussian_Mixture', 'predict'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph.data['Gaussian_Mixture', 'predict'].min(), 0)

        self.assertEqual(pgraph.data['Combine_Clustering', 'predict'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph.data['Combine_Clustering', 'predict'].min(), 0)
        self.assertEqual(pgraph.data['Combine_Clustering', 'predict'].max(),
                         pgraph.data['Gaussian_Mixture', 'predict'].max())

        self.assertEqual(pgraph.data['Paella', 'predict'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph.data['Paella', 'predict'].min(), 0)
        self.assertEqual(pgraph.data['Paella', 'predict'].max(), 1)

        self.assertEqual(pgraph.data['Regressor', 'predict'].shape[0], self.y.shape[0])


    def test_graph_predict(self):
        pgraph = self.pgraph
        pgraph.fit(X=self.X, y=self.y)
        pgraph.predict()

        assert_frame_equal(pgraph.data['_External', 'X'], self.X)
        assert_frame_equal(pgraph.data['_External', 'y'], self.y)

        self.assertEqual(pgraph.data['Regressor', 'predict'].shape[0], self.y.shape[0])



    def test_Pipegraph__step_get_params_multiple(self):
        pgraph = self.pgraph
        self.assertEqual(pgraph._step['Concatenate_Xy'].get_params(), {})
        self.assertEqual(pgraph._step['Gaussian_Mixture'].get_params()['n_components'], 3)
        self.assertEqual(pgraph._step['Dbscan'].get_params()['eps'], 0.5)
        self.assertEqual(pgraph._step['Combine_Clustering'].get_params(), {})
        # self.assertEqual(pgraph._step['Paella'].get_params()['noise_label'], -1)
        # self.assertEqual(pgraph._step['Paella'].get_params()['max_it'], 10)
        # self.assertEqual(pgraph._step['Paella'].get_params()['regular_size'], 400)
        # self.assertEqual(pgraph._step['Paella'].get_params()['minimum_size'], 100)
        # self.assertEqual(pgraph._step['Paella'].get_params()['width_r'], 0.99)
        # self.assertEqual(pgraph._step['Paella'].get_params()['n_neighbors'], 5)
        # self.assertEqual(pgraph._step['Paella'].get_params()['power'], 30)
        self.assertEqual(pgraph._step['Regressor'].get_params(),
                         {'copy_X': True, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False})

    def test_Pipegraph__step_set_params_multiple(self):
        pgraph = self.pgraph
        self.assertRaises(ValueError, pgraph._step['Concatenate_Xy'].set_params, ham=2, spam=9)
        self.assertEqual(pgraph._step['Gaussian_Mixture'].set_params(n_components=5).get_params()['n_components'],
                         5)
        self.assertEqual(pgraph._step['Dbscan'].set_params(eps=10.2).get_params()['eps'],
                         10.2)
        # self.assertEqual(pgraph._step['Paella'].set_params(noise_label=-3).get_params()['noise_label'],
        #                  -3)
        self.assertEqual(pgraph._step['Regressor'].set_params(copy_X=False).get_params(),
                         {'copy_X': False, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False})
        self.assertEqual(pgraph._step['Regressor'].set_params(n_jobs=13).get_params(),
                         {'copy_X': False, 'fit_intercept': True, 'n_jobs': 13, 'normalize': False})


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
        self.pgraph = PipeGraph(steps=steps,
                                connections=connections,
                                use_for_fit='all',
                                use_for_predict=['Regressor'])

    def test_Paella__init(self):
        logger.debug("Type of adaptee %s", type(self.pgraph._step['Paella']._strategy._adaptee))
        self.assertTrue(isinstance(self.pgraph._step['Paella']._strategy._adaptee, Paella))

    def test_Paella__under_fit__Paella__fit(self):
        pgraph = self.pgraph
        pgraph.data = {('_External', 'X'): self.X,
                       ('_External', 'y'): self.y,
                       }
        pgraph._fit('Concatenate_Xy')
        pgraph._fit('Gaussian_Mixture')
        pgraph._fit('Dbscan')
        pgraph._fit('Combine_Clustering')
        #pgraph._fit('Paella')

        paellador = pgraph._step['Paella']._strategy._adaptee

        X = pgraph.data[('_External', 'X')]
        y = pgraph.data[('_External', 'y')]
        classification = pgraph.data[('Combine_Clustering', 'predict')]
        paellador.fit(X,y,classification)

    # def test_Paella__predict(self):
    #     X = self.X
    #     y = self.y
    #     paellador = self.paellador
    #     paellador.fit(X=X, y=y)
    #     result = paellador.predict(X=X, y=y)['predict']
    #     self.assertEqual(result.shape, (10000, 1) )
    #
    # def test_Paella__get_params(self):
    #     paellador = self.paellador
    #     result = paellador.get_params()
    #     self.assertEqual(result, 12 )
    #
    # def test_Paella__set_params(self):
    #     paellador = self.paellador
    #     result = paellador.get_params()
    #     self.assertEqual(result, 12 )
    #


if __name__ == '__main__':
    unittest.main()
