import inspect
from abc import abstractmethod

from sklearn.base import BaseEstimator


class AdapterForSkLearnLikeAdaptee(BaseEstimator):
    """
    This class is an adapter for Scikit-Learn objects in order to provide a common interface based on fit and predict
    methods irrespectively of whether the adapted object provided a transform, fit_predict, or predict interface.
    It is also used by the Process class as strategy.
    """

    def __init__(self, adaptee):
        """

        Args:
            adaptee:
        """
        self._adaptee = adaptee

    def fit(self, *pargs, **kwargs):
        """

        Args:
            pargs:
            kwargs:

        Returns:

        """
        self._adaptee.fit(*pargs, **kwargs)
        return self

    @abstractmethod
    def predict(self, *pargs, **kwargs):
        """ To be implemented by subclasses """

    def _get_fit_signature(self):
        """

        Returns:

        """
        if hasattr(self._adaptee, '_get_fit_signature'):
            return self._adaptee._get_fit_signature()
        else:
            return list(inspect.signature(self._adaptee.fit).parameters)

    @abstractmethod
    def _get_predict_signature(self):
        """ For easier predict params passing"""

    # These two methods work by introspection, do not remove because the __getattr__ trick does not work with them
    def get_params(self, deep=True):
        """

        Args:
            deep:

        Returns:

        """
        return self._adaptee.get_params(deep=deep)

    def set_params(self, **params):
        """

        Args:
            params:

        Returns:

        """
        self._adaptee.set_params(**params)
        return self

    def __getattr__(self, name):
        """

        Args:
            name:

        Returns:

        """
        return getattr(self.__dict__['_adaptee'], name)

    def __setattr__(self, name, value):
        """

        Args:
            name:
            value:
        """
        if name in ('_adaptee',):
            self.__dict__[name] = value
        else:
            setattr(self.__dict__['_adaptee'], name, value)

    def __delattr__(self, name):
        """

        Args:
            name:
        """
        delattr(self.__dict__['_adaptee'], name)

    def __repr__(self):
        """

        Returns:

        """
        return self._adaptee.__repr__()


class AdapterForFitTransformAdaptee(AdapterForSkLearnLikeAdaptee):
    """
    """

    def predict(self, *pargs, **kwargs):
        """

        Args:
            pargs:
            kwargs:

        Returns:

        """
        result = {'predict': self._adaptee.transform(*pargs, **kwargs)}
        return result

    def _get_predict_signature(self):
        """

        Returns:

        """
        if hasattr(self._adaptee, '_get_predict_signature'):
            return self._adaptee._get_predict_signature()
        else:
            return list(inspect.signature(self._adaptee.transform).parameters)


class AdapterForFitPredictAdaptee(AdapterForSkLearnLikeAdaptee):
    """

    """

    def predict(self, *pargs, **kwargs):
        """

        Args:
            pargs:
            kwargs:

        Returns:

        """
        result = {'predict': self._adaptee.predict(*pargs, **kwargs)}
        if hasattr(self._adaptee, 'predict_proba'):
            result['predict_proba'] = self._adaptee.predict_proba(*pargs, **kwargs)
        if hasattr(self._adaptee, 'predict_log_proba'):
            result['predict_log_proba'] = self._adaptee.predict_log_proba(*pargs, **kwargs)
        return result

    def _get_predict_signature(self):
        """

        Returns:

        """
        if hasattr(self._adaptee, '_get_predict_signature'):
            return self._adaptee._get_predict_signature()
        else:
            return list(inspect.signature(self._adaptee.predict).parameters)


class AdapterForAtomicFitPredictAdaptee(AdapterForSkLearnLikeAdaptee):
    """

    Handler of estimator that implements only fit & predict

    """

    def fit(self, *pargs, **kwargs):
        """

        Args:
            pargs:
            kwargs:

        Returns:

        """
        return self

    # Relies on the pipegraph iteration loop to run predict after fit in order to propagate the signals

    def predict(self, *pargs, **kwargs):
        """

        Args:
            pargs:
            kwargs:

        Returns:

        """
        return {'predict': self._adaptee.fit_predict(**kwargs)}

    def _get_predict_signature(self):
        """

        Returns:

        """
        if hasattr(self._adaptee, '_get_predict_signature'):
            return self._adaptee._get_predict_signature()
        else:
            return self._get_fit_signature()


class AdapterForCustomFitPredictWithDictionaryOutputAdaptee(AdapterForSkLearnLikeAdaptee):
    def predict(self, *pargs, **kwargs):
        """

        Args:
            pargs:
            kwargs:

        Returns:

        """
        return self._adaptee.predict(*pargs, **kwargs)

    def _get_predict_signature(self):
        """

        Returns:

        """
        if hasattr(self._adaptee, '_get_predict_signature'):
            return self._adaptee._get_predict_signature()
        else:
            return list(inspect.signature(self._adaptee.predict).parameters)
