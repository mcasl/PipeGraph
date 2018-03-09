import logging
import unittest
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from pipegraph.adapters import (AdapterForFitTransformAdaptee,
                                AdapterForFitPredictAdaptee,
                                AdapterForAtomicFitPredictAdaptee,
                                AdapterForCustomFitPredictWithDictionaryOutputAdaptee)

class TestAdapterForSkLearnLikeAdaptee(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = np.random.rand(self.size, 1)
        self.y = self.X + np.random.randn(self.size, 1)

    def test_baseadapter__init(self):
        lm = LinearRegression()
        stepstrategy = AdapterForFitPredictAdaptee(lm)
        self.assertEqual(stepstrategy._adaptee, lm)

    def test_baseadapter__fit_AdapterForFitPredictAdaptee(self):
        X = self.X
        y = self.y
        lm = LinearRegression()
        stepstrategy = AdapterForFitPredictAdaptee(lm)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'coef_'), False)
        result = stepstrategy.fit(X=X, y=y)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'coef_'), True)
        self.assertEqual(stepstrategy, result)

    def test_baseadapter__fit_AdapterForFitTransformAdaptee(self):
        X = self.X
        sc = MinMaxScaler()
        stepstrategy = AdapterForFitTransformAdaptee(sc)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'n_samples_seen_'), False)
        result = stepstrategy.fit(X=X)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'n_samples_seen_'), True)
        self.assertEqual(stepstrategy, result)

    def test_baseadapter__fit_AdapterForAtomicFitPredictAdaptee(self):
        X = self.X
        db = DBSCAN()
        stepstrategy = AdapterForAtomicFitPredictAdaptee(db)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'core_sample_indices_'), False)
        result_fit = stepstrategy.fit(X=X)
        self.assertEqual(hasattr(stepstrategy._adaptee, 'core_sample_indices_'), False)
        self.assertEqual(stepstrategy, result_fit)
        result_predict = stepstrategy.predict(X=X)['predict']
        self.assertEqual(hasattr(stepstrategy._adaptee, 'core_sample_indices_'), True)
        self.assertEqual(result_predict.shape, (self.size,))

    def test_baseadapter__get_fit_signature(self):
        lm = LinearRegression()
        gm = GaussianMixture()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        gm_strategy = AdapterForFitPredictAdaptee(gm)

        result_lm = lm_strategy._get_fit_signature()
        result_gm = gm_strategy._get_fit_signature()

        self.assertEqual(sorted(result_lm), sorted(['X', 'y', 'sample_weight']))
        self.assertEqual(sorted(result_gm), sorted(['X', 'y']))

    def test_baseadapter__get_params(self):
        lm = LinearRegression()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        result_lm = lm_strategy.get_params()
        self.assertEqual(result_lm, {'copy_X': True,
                                     'fit_intercept': True,
                                     'n_jobs': 1,
                                     'normalize': False})

    def test_baseadapter__set_params(self):
        lm = LinearRegression()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
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

    def test_baseadapter__getattr__(self):
        lm = LinearRegression()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        self.assertEqual(lm_strategy.copy_X, True)

    def test_baseadapter__setattr__(self):
        lm = LinearRegression()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        self.assertEqual(lm_strategy.copy_X, True)
        lm_strategy.copy_X = False
        self.assertEqual(lm_strategy.copy_X, False)
        self.assertEqual('copy_X' in dir(lm_strategy), False)

    def test_baseadapter__delattr__(self):
        lm = LinearRegression()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        self.assertEqual(lm_strategy.copy_X, True)
        self.assertEqual('copy_X' in dir(lm_strategy), False)
        self.assertEqual('copy_X' in dir(lm), True)
        del lm_strategy.copy_X
        self.assertEqual('copy_X' in dir(lm), False)

    def test_baseadapter__repr__(self):
        lm = LinearRegression()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        result = lm_strategy.__repr__()
        self.assertEqual(result, lm.__repr__())


class TestAdapterForFitTransformAdaptee(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = np.random.rand(self.size, 1)
        self.y = self.X + np.random.randn(self.size, 1)

    def test_FitTransform__predict(self):
        X = self.X
        sc = MinMaxScaler()
        sc_strategy = AdapterForFitTransformAdaptee(sc)
        sc_strategy.fit(X=X)
        result_sc = sc_strategy.predict(X=X)
        self.assertEqual(list(result_sc.keys()), ['predict'])
        self.assertEqual(result_sc['predict'].shape, (self.size, 1))

    def test_FitTransform__get_predict_signature(self):
        sc = MinMaxScaler()
        sc_strategy = AdapterForFitTransformAdaptee(sc)
        result_sc = sc_strategy._get_predict_signature()
        self.assertEqual(result_sc, ['X'])


class TestAdapterForFitPredictAdaptee(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = np.random.rand(self.size, 1)
        self.y = self.X + np.random.randn(self.size, 1)

    def test_FitPredict__predict(self):
        X = self.X
        y = self.y
        lm = LinearRegression()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        lm_strategy.fit(X=X, y=y)
        result_lm = lm_strategy.predict(X=X)
        self.assertEqual(list(result_lm.keys()), ['predict'])
        self.assertEqual(result_lm['predict'].shape, (self.size, 1))

        gm = GaussianMixture()
        gm_strategy = AdapterForFitPredictAdaptee(gm)
        gm_strategy.fit(X=X)
        result_gm = gm_strategy.predict(X=X)
        self.assertEqual(sorted(list(result_gm.keys())),
                         sorted(['predict', 'predict_proba']))
        self.assertEqual(result_gm['predict'].shape, (self.size,))

    def test_FitPredict__get_predict_signature(self):
        lm = LinearRegression()
        lm_strategy = AdapterForFitPredictAdaptee(lm)
        result_lm = lm_strategy._get_predict_signature()
        self.assertEqual(result_lm, ['X'])

class TestAdapterForAtomicFitPredictAdaptee(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = np.random.rand(self.size, 1)
        self.y = self.X + np.random.randn(self.size, 1)

    def test_AtomicFitPredict__predict(self):
        X = self.X
        y = self.y
        db = DBSCAN()
        db_strategy = AdapterForAtomicFitPredictAdaptee(db)
        db_strategy.fit(X=X, y=y)
        result_db = db_strategy.predict(X=X)
        self.assertEqual(list(result_db.keys()), ['predict'])
        self.assertEqual(result_db['predict'].shape, (self.size,))

    def test_AtomicFitPredict__get_predict_signature(self):
        db = DBSCAN()
        db_strategy = AdapterForAtomicFitPredictAdaptee(db)
        result_db = db_strategy._get_predict_signature()
        self.assertEqual(sorted(result_db), sorted(['X', 'y', 'sample_weight']))

class TestAdapterForCustomFitPredictWithDictionaryOutputAdaptee(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = np.random.rand(self.size, 1)
        self.y = self.X + np.random.randn(self.size, 1)

    def test_FitPredictWithDictionaryOutput__predict(self):
        X = self.X
        y = self.y.astype(int)
        gm = GaussianNB()
        wrapped_gm = AdapterForFitPredictAdaptee(gm)
        double_wrap= AdapterForCustomFitPredictWithDictionaryOutputAdaptee(wrapped_gm)

        double_wrap.fit(X=X, y=y)
        result = double_wrap.predict(X=X)
        self.assertEqual(sorted(list(result.keys())),
                         sorted(['predict', 'predict_proba', 'predict_log_proba']))
        self.assertEqual(result['predict'].shape[0], self.size)
        self.assertEqual(result['predict_proba'].shape[0], self.size)
        self.assertEqual(result['predict_log_proba'].shape[0], self.size)

    def test_FitPredictWithDictionaryOutput__get_fit_signature(self):
        lm = LinearRegression()
        wrapped_lm = AdapterForFitPredictAdaptee(lm)
        double_wrap= AdapterForCustomFitPredictWithDictionaryOutputAdaptee(wrapped_lm)
        result = double_wrap._get_fit_signature()
        self.assertEqual(sorted(result), sorted(['X', 'y', 'sample_weight']))

    def test_FitPredictWithDictionaryOutput__get_predict_signature(self):
        lm = LinearRegression()
        wrapped_lm = AdapterForFitPredictAdaptee(lm)
        double_wrap= AdapterForCustomFitPredictWithDictionaryOutputAdaptee(wrapped_lm)
        result = double_wrap._get_predict_signature()
        self.assertEqual(result, ['X'])


if __name__ == '__main__':
    unittest.main()
