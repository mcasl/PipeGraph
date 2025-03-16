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
from pandas.testing import assert_frame_equal
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import GridSearchCV

from pipegraph.base import (PipeGraph,
                            )


from pipegraph.demo_blocks import (CustomCombination,
                                   )

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


class TestExampleKmeansLDA(unittest.TestCase):
    def setUp(self):
        X, y = datasets.make_blobs(n_samples=10000, n_features=5, centers=10)
        self.X, self.y = X, y
        clustering = KMeans(n_clusters=10)
        classification = LinearDiscriminantAnalysis()

        steps = [('clustering', clustering),
                 ('classification', classification)
                 ]

        pgraph = PipeGraph(steps=steps)
        pgraph.inject(sink='clustering', sink_var='X', source='_External', source_var='X')
        pgraph.inject(sink='classification', sink_var='X', source='_External', source_var='X')
        pgraph.inject(sink='classification', sink_var='y', source='clustering', source_var='predict')
        self.pgraph=pgraph



    def test_kmeans_plus_lda(self):
        #gs = GridSearchCV(pgraph, param_grid=dict(clustering__n_clusters=[1, 30]))
        #gs.fit(X)
        pgraph, X, y = self.pgraph, self.X, self.y
        pgraph.fit(X)
        result = pgraph.score(X, y=None)
        expected = pgraph.named_steps['classification'].score(X, pgraph._predict_data[('clustering', 'predict')])
        self.assertEqual(result, expected)

    def test_gridsearch(self):
        pgraph, X, y = self.pgraph, self.X, self.y
        gs = GridSearchCV(pgraph, param_grid=dict(clustering__n_clusters=[2, 30]), cv=5, refit=True)
        gs.fit(X)
        result = gs.score(X, y=None)
        model = gs.best_estimator_
        expected = model.named_steps['classification'].score(X, model._predict_data[('clustering', 'predict')])
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
