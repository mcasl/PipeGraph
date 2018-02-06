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

        result_lm = lm_strategy.get_fit_parameters_from_signature()
        result_gm = gm_strategy.get_fit_parameters_from_signature()

        self.assertEqual(result_lm, ['X', 'y', 'sample_weight'])
        self.assertEqual(result_gm, ['X', 'y'])

    def test_stepstrategy__get_predict_parameters_from_signature_FitPredict(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
        result_lm = lm_strategy.get_predict_parameters_from_signature()
        self.assertEqual(result_lm, ['X'])

    def test_stepstrategy__get_predict_parameters_from_signature_FitTransform(self):
        sc = MinMaxScaler()
        sc_strategy = FitTransformStrategy(sc)
        result_sc = sc_strategy.get_predict_parameters_from_signature()
        self.assertEqual(result_sc, ['X'])

    def test_stepstrategy__get_predict_parameters_from_signature_AtomicFitPredict(self):
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
        result_lm = step_lm.get_fit_parameters_from_signature()
        result_gm = step_gm.get_fit_parameters_from_signature()

        self.assertEqual(result_lm, ['X', 'y', 'sample_weight'])
        self.assertEqual(result_gm, ['X', 'y'])

    def test_step__get_predict_parameters_from_signature_lm(self):
        lm = LinearRegression()
        lm_strategy = FitPredictStrategy(lm)
        step_lm = Step(lm_strategy)
        result_lm = step_lm.get_predict_parameters_from_signature()
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
        paellaModel = CustomPaella(regressor=LinearRegression,
                                   noise_label=None,
                                   max_it=1,
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

    def test_build_graph_use_for_fit_all(self):
        graph = build_graph(self.steps,
                            self.connections,
                            use_for_fit='all',
                            use_for_predict='all')

        fit_nodes_list = [name for name in graph.nodes
                          if 'fit' in graph.node[name]['use_for']]
        self.assertEqual(fit_nodes_list.sort(), ['Concatenate_Xy',
                                                 'Gaussian_Mixture',
                                                 'Dbscan',
                                                 'Combine_Clustering',
                                                 'Paella',
                                                 'Regressor',
                                                 ].sort())

    def test_build_graph_use_for_fit_some(self):
        graph = build_graph(self.steps,
                            self.connections,
                            use_for_fit=['Concatenate_Xy',
                                         'Gaussian_Mixture',
                                         'Dbscan',
                                         ],
                            use_for_predict='all')

        fit_nodes_list = [name for name in graph.nodes
                          if 'fit' in graph.node[name]['use_for']]
        self.assertEqual(fit_nodes_list.sort(), ['Concatenate_Xy',
                                                 'Gaussian_Mixture',
                                                 'Dbscan',
                                                 ].sort())

    def test_build_graph_use_for_fit_one(self):
        graph = build_graph(self.steps,
                            self.connections,
                            use_for_fit=['Concatenate_Xy'],
                            use_for_predict='all')

        fit_nodes_list = [name for name in graph.nodes
                          if 'fit' in graph.node[name]['use_for']]
        self.assertEqual(fit_nodes_list.sort(), ['Concatenate_Xy'].sort())

    def test_build_graph_use_for_fit_Empty(self):
        graph = build_graph(self.steps,
                            self.connections,
                            use_for_fit=[],
                            use_for_predict='all')

        fit_nodes_list = [name for name in graph.nodes
                          if 'fit' in graph.node[name]['use_for']]
        self.assertEqual(fit_nodes_list, [])

    def test_build_graph_use_for_fit_None(self):
        graph = build_graph(self.steps,
                            self.connections,
                            use_for_fit=None,
                            use_for_predict='all')

        fit_nodes_list = [name for name in graph.nodes
                          if 'fit' in graph.node[name]['use_for']]
        self.assertEqual(fit_nodes_list, [])

    def test_build_graph_use_for_fit_First(self):
        graph = build_graph(self.steps,
                            self.connections,
                            use_for_fit=['First'],
                            use_for_predict='all')

        fit_nodes_list = [name for name in graph.nodes
                          if 'fit' in graph.node[name]['use_for']]
        self.assertEqual(fit_nodes_list, [])

    def test_build_graph_use_for_predict_all(self):
        graph = build_graph(self.steps,
                            self.connections,
                            use_for_fit='all',
                            use_for_predict='all')

        predict_nodes_list = [name for name in graph.nodes
                              if 'predict' in graph.node[name]['use_for']]
        self.assertEqual(predict_nodes_list.sort(), ['Concatenate_Xy',
                                                     'Gaussian_Mixture',
                                                     'Dbscan',
                                                     'Combine_Clustering',
                                                     'Paella',
                                                     'Regressor',
                                                     ].sort())

    def test_build_graph_use_for_predict_some(self):
        graph = build_graph(self.steps,
                            self.connections,
                            use_for_fit='all',
                            use_for_predict=['Concatenate_Xy',
                                             'Gaussian_Mixture',
                                             'Dbscan',
                                             ])

        predict_nodes_list = [name for name in graph.nodes
                              if 'predict' in graph.node[name]['use_for']]
        self.assertEqual(predict_nodes_list.sort(), ['Concatenate_Xy',
                                                     'Gaussian_Mixture',
                                                     'Dbscan',
                                                     ].sort())

    def test_build_graph_use_for_predict_one(self):
        graph = build_graph(self.steps,
                            self.connections,
                            use_for_predict=['Concatenate_Xy'],
                            use_for_fit='all')

        predict_nodes_list = [name for name in graph.nodes
                              if 'predict' in graph.node[name]['use_for']]
        self.assertEqual(predict_nodes_list.sort(), ['Concatenate_Xy'].sort())

    def test_build_graph_use_for_predict_Empty(self):
        graph = build_graph(self.steps,
                            self.connections,
                            use_for_predict=[],
                            use_for_fit='all')

        predict_nodes_list = [name for name in graph.nodes
                              if 'predict' in graph.node[name]['use_for']]
        self.assertEqual(predict_nodes_list, [])

    def test_build_graph_use_for_predict_None(self):
        graph = build_graph(self.steps,
                            self.connections,
                            use_for_predict=None,
                            use_for_fit='all')

        predict_nodes_list = [name for name in graph.nodes
                              if 'predict' in graph.node[name]['use_for']]
        self.assertEqual(predict_nodes_list, [])

    def test_build_graph_node_names(self):
        graph = build_graph(self.steps,
                            self.connections,
                            use_for_fit='all',
                            use_for_predict='all')

        node_list = list(graph.nodes)
        self.assertEqual(node_list, ['Concatenate_Xy',
                                     'Gaussian_Mixture',
                                     'Dbscan',
                                     'Combine_Clustering',
                                     'Paella',
                                     'Regressor',
                                     ])

    def test_build_graph_node_steps(self):
        graph = build_graph(self.steps,
                            self.connections,
                            use_for_fit='all',
                            use_for_predict='all')

        for name in graph.nodes:
            self.assertTrue(isinstance(graph.nodes[name]['step'], Step))

        self.assertTrue(isinstance(graph.node['Concatenate_Xy']['step']._strategy, FitPredictStrategy))
        self.assertTrue(isinstance(graph.node['Gaussian_Mixture']['step']._strategy, FitPredictStrategy))
        self.assertTrue(isinstance(graph.node['Dbscan']['step']._strategy, AtomicFitPredictStrategy))
        self.assertTrue(isinstance(graph.node['Combine_Clustering']['step']._strategy, FitPredictStrategy))
        self.assertTrue(isinstance(graph.node['Paella']['step']._strategy, FitPredictStrategy))
        self.assertTrue(isinstance(graph.node['Regressor']['step']._strategy, FitPredictStrategy))

        self.assertTrue(isinstance(graph.node['Concatenate_Xy']['step']._strategy._adaptee, CustomConcatenation))
        self.assertTrue(isinstance(graph.node['Gaussian_Mixture']['step']._strategy._adaptee, GaussianMixture))
        self.assertTrue(isinstance(graph.node['Dbscan']['step']._strategy._adaptee, DBSCAN))
        self.assertTrue(isinstance(graph.node['Combine_Clustering']['step']._strategy._adaptee, CustomCombination))
        self.assertTrue(isinstance(graph.node['Paella']['step']._strategy._adaptee, CustomPaella))
        self.assertTrue(isinstance(graph.node['Regressor']['step']._strategy._adaptee, LinearRegression))

    def test_build_graph_node_edges(self):
        graph = build_graph(self.steps,
                            self.connections,
                            use_for_fit='all',
                            use_for_predict='all')

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

        self.pgraph = PipeGraph(steps=steps,
                                connections=connections,
                                use_for_fit='all',
                                use_for_predict=['Regressor'])

    def test_read_data(self):
        pgraph = self.pgraph
        input_data = {'X': self.X,
                      'y': self.y,
                      'z': 57,
                      }

        fit_variables = pgraph._graph.node['Regressor']['step'].get_fit_parameters_from_signature()
        result = pgraph._filter_data(input_data, fit_variables)

        self.assertEqual(list(result.keys()).sort(), ['X', 'sample_weight', 'y'].sort())
        assert_array_equal(result['X'], self.X)
        assert_array_equal(result['y'], self.y)
        self.assertTrue(result['sample_weight'] is None)

    def test_read_connections_regressor_node(self):
        pgraph = self.pgraph
        pgraph.data = {('_First', 'X'): self.X,
                       ('_First', 'y'): self.y,
                       ('Paella', 'predict'): self.y * 2,
                       ('Dbscan', 'predict'): self.y * 4,
                       }

        result = pgraph._read_from_connections('Regressor')
        assert_array_equal(result['X'], self.X)
        assert_array_equal(result['y'], self.y)
        assert_array_equal(result['sample_weight'], self.y * 2)
        self.assertEqual(len(result), 3)

    def test_step__predict_lm(self):
        X = self.X
        y = self.y
        lm = LinearRegression()
        lm_step = make_step(lm)
        lm_step.fit(X=X, y=y)
        assert_array_equal(lm.predict(X), lm_step.predict(X=X)['predict'])

    def test_concatenate_Xy__predict(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        expected = pd.concat([X, y], axis=1)
        current_step = pgraph._graph.node['Concatenate_Xy']['step']
        current_step.fit(df1=X, df2=y)
        result = current_step.predict(df1=X, df2=y)['predict']
        assert_frame_equal(expected, result)

    def test_gaussian_mixture__predict(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._graph.node['Gaussian_Mixture']['step']
        current_step.fit(X=X)
        expected = current_step._strategy._adaptee.predict(X=X)
        result = current_step.predict(X=X)['predict']
        assert_array_equal(expected, result)

    def test_dbscan__predict(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
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

    def test_strategy__dict_key(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._graph.node['Concatenate_Xy']['step']
        current_step.fit(df1=X, df2=y)
        result = current_step.predict(df1=X, df2=y)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_dbscan__dict_key(self):
        X = self.X
        pgraph = self.pgraph
        current_step = pgraph._graph.node['Dbscan']['step']
        current_step.fit(X=X)
        result = current_step.predict(X=X)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_combine_clustering__dict_key(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._graph.node['Combine_Clustering']['step']
        current_step.fit(dominant=X, other=y)
        result = current_step.predict(dominant=X, other=y)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_gaussian_mixture__dict_key(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._graph.node['Gaussian_Mixture']['step']
        current_step.fit(X=X)
        result = current_step.predict(X=X)
        self.assertEqual(list(result.keys()), ['predict', 'predict_proba'])

    def test_regressor__dict_key(self):
        X = self.X
        y = self.y
        pgraph = self.pgraph
        current_step = pgraph._graph.node['Regressor']['step']
        current_step.fit(X=X, y=y)
        result = current_step.predict(X=X)
        self.assertEqual(list(result.keys()), ['predict'])

    def test_data_attribute(self):
        pgraph = self.pgraph
        self.assertEqual(dict(), pgraph.data)

    def test_node_names(self):
        pgraph = self.pgraph
        node_list = list(pgraph._graph.nodes)
        self.assertEqual(node_list, ['Concatenate_Xy',
                                     'Gaussian_Mixture',
                                     'Dbscan',
                                     'Combine_Clustering',
                                     'Paella',
                                     'Regressor',
                                     ])

    def test_filter_nodes_fit(self):
        pgraph = self.pgraph
        fit_nodes = [item['name'] for item in pgraph._filter_nodes(filter='fit')]
        self.assertEqual(fit_nodes, ['Concatenate_Xy',
                                     'Dbscan',
                                     'Gaussian_Mixture',
                                     'Combine_Clustering',
                                     'Paella',
                                     'Regressor',
                                     ])

    def test_filter_nodes_predict(self):
        pgraph = self.pgraph
        predict_nodes = [item['name'] for item in pgraph._filter_nodes(filter='predict')]
        self.assertEqual(predict_nodes, ['Regressor' ])

    def test_graph_fit(self):
        pgraph = self.pgraph
        pgraph.fit()
        assert_frame_equal(pgraph.data['_First', 'X'], self.X)
        assert_frame_equal(pgraph.data['_First', 'y'], self.y)

        self.assertEqual(pgraph.data['Dbscan', 'prediction'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph.data['Dbscan', 'prediction'].min(), -1)

        self.assertEqual(pgraph.data['Gaussian_Mixture', 'prediction'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph.data['Gaussian_Mixture', 'prediction'].min(), 0)

        self.assertEqual(pgraph.data['Combine_Clustering', 'classification'].shape[0], self.y.shape[0])
        self.assertEqual(pgraph.data['Combine_Clustering', 'classification'].min(), -1)
        self.assertEqual(pgraph.data['Combine_Clustering', 'classification'].max(),
                         pgraph.data['Gaussian_Mixture', 'prediction'].max())

        # self.assertEqual(pgraph.data['Paella', 'prediction'].shape[0], self.y.shape[0])
        # self.assertEqual(pgraph.data['Paella', 'prediction'].min(), 0)
        # self.assertEqual(pgraph.data['Paella', 'prediction'].max(), 1)

        self.assertEqual(pgraph.data['Regressor', 'prediction'].shape[0], self.y.shape[0])

        assert_array_equal(pgraph.data['Last', 'prediction'],
                           pgraph.data['Regressor', 'prediction'])

    # def test_graph_predict(self):
    #     self.graph.fit()
    #     self.graph.predict()
    #
    #     assert_frame_equal(self.graph.data['_First', 'X'], self.X)
    #     assert_frame_equal(self.graph.data['_First', 'y'], self.y)
    #
    #     self.assertEqual(self.graph.data['Regressor', 'prediction'].shape[0], self.y.shape[0])
    #
    #     assert_array_equal(self.graph.data['Last', 'prediction'],
    #                        self.graph.data['Regressor', 'prediction'])
    #

    def test_step_get_params_multiple(self):
        pgraph = self.pgraph
        graph = pgraph._graph
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
        pgraph = self.pgraph
        graph = pgraph._graph
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


class TestCustomPaella(unittest.TestCase):
    def setUp(self):
        URL = "https://raw.githubusercontent.com/mcasl/PAELLA/master/data/sin_60_percent_noise.csv"
        data = pd.read_csv(URL, usecols=['V1', 'V2'])
        self.X = data[['V1']]
        self.y = data[['V2']]
        self.paellador = CustomPaella(regressor=LinearRegression,
                                      noise_label=None,
                                      max_it=1,
                                      regular_size=100,
                                      minimum_size=30,
                                      width_r=0.95,
                                      power=10,
                                      random_state=42)

    def test_Paella__init(self):
        self.assertTrue(isinstance(self.paellador._adaptee, Paella))

    # def test_Paella__fit(self):
    #     X = self.X
    #     y = self.y
    #     concatenated = pd.concat([X,y], axis=1)
    #     gm =GaussianMixture(n_components=3)
    #     gm.fit(concatenated)
    #     classification_gm = gm.predict(concatenated)
    #     paellador = self.paellador
    #     self.assertFalse(hasattr(paellador._adaptee, 'appointment_'))
    #     paellador.fit(X=X, y=y, classification=classification_gm )
    #     self.assertTrue(hasattr(paellador._adaptee, 'appointment_'))
    #
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
