import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from pipegraph.adapters import (AdapterForCustomFitPredictWithDictionaryOutputAdaptee,
                                AdapterForFitPredictAdaptee,
                                )


class Concatenator(BaseEstimator):
    """
    Concatenate a set of data
    """
    def fit(self):
        """Fit method that does, in this case, nothing but returning self.
        Returns
        -------
            self : returns an instance of _Concatenator.
        """
        return self

    def predict(self, **kwargs):
        """Check the input data type for correct concatenating.

        Parameters
        ----------
        **kwargs :  sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse
            matrices or pandas dataframes.
            Data to concatenate.
        Returns
        -------
        Pandas series or Pandas DataFrame with the data input concatenated
        """
        df_list = []
        for name in sorted(kwargs.keys()):
            item = kwargs[name]
            if isinstance(item, pd.Series) or isinstance(item, pd.DataFrame):
                df_list.append(item)
            else:
                df_list.append(pd.DataFrame(data=item))
        return pd.concat(df_list, axis=1)


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


class ColumnSelector(BaseEstimator):
    """ Slice data-input data in columns
    Parameters
    ----------
    mapping : list. Each element contains data-column and the new name assigned
    """
    def __init__(self, mapping=None):
        self.mapping = mapping

    def fit(self):
        """"

        Returns
        -------
        self : returns an instance of _CustomPower.
        """
        return self

    def predict(self, X):
        """"
        X: iterable object
                Data to slice.
        Returns
        -------
        returns a list with data-column and the new name assigned for each column selected
        """

        if self.mapping is None:
            return {'predict': X}
        result = {name: X.iloc[:, column_slice]
            for name, column_slice in self.mapping.items()}
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


class Reshape(BaseEstimator):
    def __init__(self, dimension):
        self.dimension = dimension

    def fit(self):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        return X.reshape(self.dimension)


class Demultiplexer(BaseEstimator):
    """ Slice data-input data in columns
    Parameters
    ----------
    mapping : list. Each element contains data-column and the new name assigned
    """
    def fit(self):
        return self

    def predict(self, **kwargs):
        selection = kwargs.pop('selection')
        result = dict()
        for variable, value in kwargs.items():
            for class_number in set(selection):
                if value is None:
                    result[variable + "_" + str(class_number)] = None
                else:
                    result[variable + "_" + str(class_number)] = pd.DataFrame(value).loc[selection == class_number, :]
        return result


class Multiplexer(BaseEstimator):
    def fit(self):
        return self

    def predict(self, **kwargs):
        #list of arrays with the same dimension
        selection = kwargs['selection']
        array_list = [pd.DataFrame(data=kwargs[str(class_number)],
                                   index=np.flatnonzero(selection == class_number))
                      for class_number in set(selection)]
        result = pd.concat(array_list, axis=0).sort_index()
        return result


strategies_for_custom_adaptees = {
    Concatenator: AdapterForFitPredictAdaptee,
    CustomCombination: AdapterForFitPredictAdaptee,
    TrainTestSplit: AdapterForCustomFitPredictWithDictionaryOutputAdaptee,
    ColumnSelector: AdapterForCustomFitPredictWithDictionaryOutputAdaptee,
    CustomPower: AdapterForFitPredictAdaptee,
    Reshape: AdapterForFitPredictAdaptee,
    Demultiplexer: AdapterForCustomFitPredictWithDictionaryOutputAdaptee,
    Multiplexer: AdapterForFitPredictAdaptee,
}
