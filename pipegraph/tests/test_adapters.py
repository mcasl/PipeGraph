import logging
import unittest
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from pipegraph.base import wrap_adaptee_in_process
from pipegraph.adapters import (FitTransformMixin,
                                FitPredictMixin,
                                AtomicFitPredictMixin,
                                CustomFitPredictWithDictionaryOutputMixin)

class TestAdapterForSkLearnLikeAdaptee(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = np.random.rand(self.size, 1)
        self.y = self.X + np.random.randn(self.size, 1)

    def test_baseadapter__get_fit_signature(self):
        lm = LinearRegression()
        gm = GaussianMixture()
        lm.__class__ = type('newClass', (type(lm), FitPredictMixin), {} )
        gm.__class__ = type('newClass', (type(gm), FitPredictMixin), {} )

        result_lm = lm._get_fit_signature()
        result_gm = gm._get_fit_signature()

        self.assertEqual(sorted(result_lm), sorted(['X', 'y', 'sample_weight']))
        self.assertEqual(sorted(result_gm), sorted(['X', 'y']))



class TestFitTransformMixin(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = np.random.rand(self.size, 1)
        self.y = self.X + np.random.randn(self.size, 1)

    def test_FitTransform__predict(self):
        X = self.X
        sc = MinMaxScaler()
        sc.__class__ = type('newClass', (type(sc), FitTransformMixin), {})
        sc.pg_fit(X=X)
        result_sc = sc.pg_predict(X=X)
        self.assertEqual(list(result_sc.keys()), ['predict'])
        self.assertEqual(result_sc['predict'].shape, (self.size, 1))

    def test_FitTransform__get_predict_signature(self):
        sc = MinMaxScaler()
        sc.__class__ = type('newClass', (type(sc), FitTransformMixin), {})
        result_sc = sc._get_predict_signature()
        self.assertEqual(result_sc, ['X'])


class TestFitPredictMixin(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = np.random.rand(self.size, 1)
        self.y = self.X + np.random.randn(self.size, 1)

    def test_FitPredict__predict(self):
        X = self.X
        y = self.y
        lm = LinearRegression()
        lm.__class__ = type('newClass', (type(lm), FitPredictMixin), {})
        lm.fit(X=X, y=y)
        result_lm = lm.pg_predict(X=X)
        self.assertEqual(list(result_lm.keys()), ['predict'])
        self.assertEqual(result_lm['predict'].shape, (self.size, 1))

        gm = GaussianMixture()
        gm.__class__ = type('newClass', (type(gm), FitPredictMixin), {})
        gm.pg_fit(X=X)
        result_gm = gm.pg_predict(X=X)
        self.assertEqual(sorted(list(result_gm.keys())),
                         sorted(['predict', 'predict_proba']))
        self.assertEqual(result_gm['predict'].shape, (self.size,))

    def test_FitPredict__get_predict_signature(self):
        lm = LinearRegression()
        lm.__class__ = type('newClass', (type(lm), FitPredictMixin), {})
        result_lm = lm._get_predict_signature()
        self.assertEqual(result_lm, ['X'])

class TestAtomicFitPredictMixin(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = np.random.rand(self.size, 1)
        self.y = self.X + np.random.randn(self.size, 1)

    def test_AtomicFitPredictMixin__predict(self):
        X = self.X
        y = self.y
        db = DBSCAN()
        db.__class__ = type('newClass', (type(db), AtomicFitPredictMixin), {})
        db.fit(X=X, y=y)
        result_db = db.pg_predict(X=X)
        self.assertEqual(list(result_db.keys()), ['predict'])
        self.assertEqual(result_db['predict'].shape, (self.size,))

    def test_AtomicFitPredictMixin__get_predict_signature(self):
        db = DBSCAN()
        db.__class__ = type('newClass', (type(db), AtomicFitPredictMixin), {})
        result_db = db._get_predict_signature()
        self.assertEqual(sorted(result_db), sorted(['X', 'y', 'sample_weight']))

class TestCustomFitPredictWithDictionaryOutputMixin(unittest.TestCase):
    def setUp(self):
        self.size = 100
        self.X = np.random.rand(self.size, 1)
        self.y = self.X + np.random.randn(self.size, 1)

    def test_FitPredictWithDictionaryOutput__predict(self):
        X = self.X
        y = self.y.astype(int)
        gm = GaussianNB()
        wrapped_gm = wrap_adaptee_in_process(gm)
        double_wrap= wrap_adaptee_in_process(wrapped_gm)

        double_wrap.fit(X=X, y=y)
        result = double_wrap.predict(X=X)
        self.assertEqual(sorted(list(result.keys())),
                         sorted(['predict', 'predict_proba', 'predict_log_proba']))
        self.assertEqual(result['predict'].shape[0], self.size)
        self.assertEqual(result['predict_proba'].shape[0], self.size)
        self.assertEqual(result['predict_log_proba'].shape[0], self.size)

    def test_FitPredictWithDictionaryOutput__get_fit_signature(self):
        lm = LinearRegression()
        wrapped_lm = wrap_adaptee_in_process(lm)
        double_wrap= wrap_adaptee_in_process(wrapped_lm)
        result = double_wrap._get_fit_signature()
        self.assertEqual(sorted(result), sorted(['X', 'y', 'sample_weight']))

    def test_FitPredictWithDictionaryOutput__get_predict_signature(self):
        lm = LinearRegression()
        wrapped_lm = wrap_adaptee_in_process(lm)
        double_wrap= wrap_adaptee_in_process(wrapped_lm)
        result = double_wrap._get_predict_signature()
        self.assertEqual(result, ['X'])


if __name__ == '__main__':
    unittest.main()
