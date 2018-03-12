import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split


class CustomCombination(BaseEstimator):
    """
    Only For PAELLA purposes
    """

    def fit(self, dominant, other):
        """

        Args:
            dominant:
            other:

        Returns:

        """
        return self

    def predict(self, dominant, other):
        """

        Args:
            dominant:
            other:

        Returns:

        """
        return np.where(dominant < 0, dominant, other)


class TrainTestSplit(BaseEstimator):
    """Split arrays or matrices into random train and test subsets
        Quick utility that wraps input validation and
        ``next(ShuffleSplit().split(X, y))`` and application to input data
        into a single call for splitting (and optionally subsampling) data in a
        oneliner.
        Read more in the :ref:`User Guide <cross_validation>`.
        Parameters
        ----------
        test_size : float, int, None, optional
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. By default, the value is set to 0.25.
            The default will change in version 0.21. It will remain 0.25 only
            if ``train_size`` is unspecified, otherwise it will complement
            the specified ``train_size``.
        train_size : float, int, or None, default None
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.
        random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
    """

    def __init__(self, test_size=0.25, train_size=None, random_state=None):
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def fit(self, *pargs, **kwargs):
        """Fit the model included in the step
        Returns
        -------
            self : returns an instance of _TrainTestSplit.
        """
        return self

    def predict(self, *pargs, **kwargs):
        """Fit the model included in the step
        Parameters
        ----------
        **kwargs: sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse
            matrices or pandas dataframes.
        Returns
        -------
        splitting : list, length=2 * len(arrays)
            List containing train-test split of inputs.
        """
        if len(pargs) > 0:
            raise ValueError("The developers assume you will use keyword parameters on the TrainTestSplit class.")
        array_names = list(kwargs.keys())
        train_test_array_names = sum([[item + "_train", item + "_test"] for item in array_names], [])

        result = dict(zip(train_test_array_names,
                          train_test_split(*kwargs.values())))
        return result


class CustomPower(BaseEstimator):
    """ Raises X data to power defined such as range as parameter
    Parameters
    ----------
    power : range of integers for the powering operation
    """

    def __init__(self, power=1):
        self.power = power

    def fit(self):
        """"
        Returns
        -------
            self : returns an instance of _CustomPower.
        """
        return self

    def predict(self, X):
        """"
        Parameters
        ----------
        X: iterable
            Data to power.
        Returns
        -------
        result of raising power operation
        """
        return X.values.reshape(-1, ) ** self.power


from pipegraph.adapters import (AdapterForFitPredictAdaptee,
                                AdapterForCustomFitPredictWithDictionaryOutputAdaptee,
                                )


strategies_for_demo_blocks_adaptees = {
    CustomCombination: AdapterForFitPredictAdaptee,
    TrainTestSplit: AdapterForCustomFitPredictWithDictionaryOutputAdaptee,
    CustomPower: AdapterForFitPredictAdaptee,
}


