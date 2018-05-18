import inspect
from abc import abstractmethod

from sklearn.base import BaseEstimator

class AdapterMixins:
    def pg_fit(self, *pargs, **kwargs):
        return self.fit(*pargs, **kwargs)

    def _get_fit_signature(self):
        return list(inspect.signature(self.fit).parameters)


class FitTransformMixin(AdapterMixins):
    def pg_predict(self, *pargs, **kwargs):
        result = {'predict': self.transform(*pargs, **kwargs)}
        return result

    def _get_predict_signature(self):
        return list(inspect.signature(self.transform).parameters)


class FitPredictMixin(AdapterMixins):
    def pg_predict(self, *pargs, **kwargs):
        result = {'predict': self.predict(*pargs, **kwargs)}
        if hasattr(self, 'predict_proba'):
            result['predict_proba'] = self.predict_proba(*pargs, **kwargs)
        if hasattr(self, 'predict_log_proba'):
            result['predict_log_proba'] = self.predict_log_proba(*pargs, **kwargs)
        return result

    def _get_predict_signature(self):
        return list(inspect.signature(self.predict).parameters)

class AtomicFitPredictMixin(AdapterMixins):
    def pg_fit(self, *pargs, **kwargs):
        return self

    def pg_predict(self, *pargs, **kwargs):
        return {'predict': self.fit_predict(**kwargs)}

    def _get_predict_signature(self):
        return self._get_fit_signature()


class CustomFitPredictWithDictionaryOutputMixin(AdapterMixins):
    def pg_predict(self, *pargs, **kwargs):
        return self.predict(*pargs, **kwargs)

    def _get_predict_signature(self):
        return list(inspect.signature(self.predict).parameters)
