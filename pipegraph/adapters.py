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

import inspect
from abc import abstractmethod

from sklearn.base import BaseEstimator

class AdapterMixins:
    def fit_using_varargs(self, *pargs, **kwargs):
        return self.fit(*pargs, **kwargs)

    def _get_fit_signature(self):
        return list(inspect.signature(self.fit).parameters)


class FitTransformMixin(AdapterMixins):
    def predict_dict(self, *pargs, **kwargs):
        result = {'predict': self.transform(*pargs, **kwargs)}
        return result

    def _get_predict_signature(self):
        return list(inspect.signature(self.transform).parameters)


class FitPredictMixin(AdapterMixins):
    def predict_dict(self, *pargs, **kwargs):
        result = {'predict': self.predict(*pargs, **kwargs)}
        if hasattr(self, 'predict_proba'):
            result['predict_proba'] = self.predict_proba(*pargs, **kwargs)
        if hasattr(self, 'predict_log_proba'):
            result['predict_log_proba'] = self.predict_log_proba(*pargs, **kwargs)
        return result

    def _get_predict_signature(self):
        return list(inspect.signature(self.predict).parameters)


class AtomicFitPredictMixin(AdapterMixins):
    def fit_using_varargs(self, *pargs, **kwargs):
        return self

    def predict_dict(self, *pargs, **kwargs):
        return {'predict': self.fit_predict(**kwargs)}

    def _get_predict_signature(self):
        return self._get_fit_signature()


class CustomFitPredictWithDictionaryOutputMixin(AdapterMixins):
    def predict_dict(self, *pargs, **kwargs):
        return self.predict(*pargs, **kwargs)

    def _get_predict_signature(self):
        return list(inspect.signature(self.predict).parameters)
