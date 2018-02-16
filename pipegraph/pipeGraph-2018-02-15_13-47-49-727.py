# -*- coding: utf-8 -*-
import inspect
import logging
from abc import abstractmethod

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from sklearn.utils.metaestimators import _BaseComposition

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class PipeGraphRegressor(BaseEstimator, RegressorMixin):
    """Pipeline of transforms with a final estimator.
    Implementes a complex graph...................
    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.

    connections: list
        List of (step, data-input) for each step is necessary define witha  list of I

    """
    def __init__(self, steps, connections, alternative_connections=None):
        self.pipegraph = PipeGraph(steps, connections, alternative_connections)

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self.pipegraph.get_params(deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        self.pipegraph.set_params(**kwargs)
        return self

    @property
    def named_steps(self):
        return self.pipegraph.named_steps

    def fit(self, X, y=None, **fit_params):
        """Fit the model

        Fit the PipeGraph steps one after the other and following the topological order of the graph.


        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : PipeGraphRegressor
            This estimator
        """
        self.pipegraph.fit(X, y=y, **fit_params)
        return self

    def predict(self, X):
        """Predict the PipeGraph steps one after the other and following the topological
        order of the graph.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_pred : array-like
        """
        return self.pipegraph.predict(X)['predict']

    def fit_predict(self, X, y=None, **fit_params):
        """Applies fit_predict of last step in PipeGraph after it predicts the PipeGraph steps one after
        the other and following the topological order of the graph.

        Applies predict of a PipeGraph to the data following the topological order of the graph, followed by the
        fit_predict method of the final step in the PipeGraph. Valid only if the final step implements fit_predict.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : array-like
        """
        self.pipegraph.fit(X, y=y, **fit_params)
        return self.pipegraph.predict(X)

    def predict_proba(self, X):
        """Apply transforms, and predict_proba of the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of the pipeline.
        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        return self.pipegraph.predict(X)['predict_proba']

    def decision_function(self, X):
        """Apply transforms, and decision_function of the final estimator
         Parameters
         ----------
         X : iterable
            Data to predict on. Must fulfill input requirements of first step of the pipeline.
        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        self.pipegraph.predict(X)
        return self.pipegraph.steps[-1][-1].decision_function(X)

    def predict_log_proba(self, X):

        """Apply transforms, and predict_log_proba of the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of the pipeline.
        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        return self.pipegraph.predict(X)['predict_log_proba']

    def score(self, X, y=None, sample_weight=None):
        """Apply transforms, and score with the final estimator
         Parameters
         ----------

         X : iterable
             Data to predict on. Must fulfill input requirements of first step of the pipeline.
         y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all steps of the pipeline.
         sample_weight : array-like, default=None
            If not None, this argument is passed as sample_weight keyword argument to the score method of the final estimator.
         Returns
         -------
         y_proba : array-like, shape = [n_samples, n_classes]
         """
        self.pipegraph.predict(X)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        final_step_name = self.pipegraph.steps[-1][0]
        Xt = self.pipegraph._predict_data[self.pipegraph.connections[final_step_name]['X']]
        return self.pipegraph.steps[-1][-1].score(Xt, y, **score_params)


class PipeGraphClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, steps, connections, alternative_connections=None):
        self.pipegraph = PipeGraph(steps, connections, alternative_connections)

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self.pipegraph.get_params(deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        self.pipegraph.set_params(**kwargs)
        return self

    @property
    def named_steps(self):
        return pipegraph.named_steps

    def fit(self, X, y=None, **fit_params):
        """Fit the model

        Fit the PipeGraph steps one after the other and following the topological order of the graph.


        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : PipeGraphRegressor
            This estimator
        """
        self.pipegraph.fit(X, y=y, **fit_params)
        return self

    def predict(self, X):
        """Predict the PipeGraph steps one after the other and following the topological
        order of the graph.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_pred : array-like
        """
        return self.pipegraph.predict(X)['predict']

    def fit_predict(self, X, y=None, **fit_params):
        """Applies fit_predict of last step in PipeGraph after it predicts the PipeGraph steps one after
        the other and following the topological order of the graph.

        Applies predict of a PipeGraph to the data following the topological order of the graph, followed by the
        fit_predict method of the final step in the PipeGraph. Valid only if the final step implements fit_predict.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : array-like
        """
        self.pipegraph.fit(X, y=y, **fit_params)
        return self.pipegraph.predict(X)

    def predict_proba(self, X):

        """Apply transforms, and predict_proba of the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of the pipeline.
        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        return self.pipegraph.predict(X)['predict_proba']

    def decision_function(self, X):
        """Apply transforms, and decision_function of the final estimator
         Parameters
         ----------
         X : iterable
            Data to predict on. Must fulfill input requirements of first step of the pipeline.
        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        self.pipegraph.predict(X)
        return self.pipegraph.steps[-1][-1].decision_function(X)

    def predict_log_proba(self, X):
        """Apply transforms, and predict_log_proba of the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of the pipeline.
        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        return self.pipegraph.predict(X)['predict_log_proba']

    def score(self, X, y=None, sample_weight=None):
        """Apply transforms, and score with the final estimator
         Parameters
         ----------

         X : iterable
             Data to predict on. Must fulfill input requirements of first step of the pipeline.
         y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all steps of the pipeline.
         sample_weight : array-like, default=None
            If not None, this argument is passed as sample_weight keyword argument to the score method of the final estimator.
         Returns
         -------
         y_proba : array-like, shape = [n_samples, n_classes]
         """
        self.pipegraph.predict(X)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        final_step_name = self.pipegraph.steps[-1][0]
        Xt = self.pipegraph._predict_data[self.pipegraph.connections[final_step_name]['X']]
        return self.pipegraph.steps[-1][-1].score(Xt, y, **score_params)


class PipeGraph(_BaseComposition):
    """
    """

    def __init__(self, steps, connections, alternative_connections=None):
        """

        Args:
            steps:
            connections:
            alternative_connections:
        """
        self.steps = steps
        self.connections = connections
        self.alternative_connections = alternative_connections if alternative_connections is not None else connections

        node_names = [name for name, model in steps]
        if '_External' in node_names:
            raise ValueError("Please use another name for the _External node. _External is used internally.")

        self._fit_graph = build_graph(self.connections)
        self._predict_graph = build_graph(self.alternative_connections)
        self._step = {name: make_step(adaptee=step_model) for name, step_model in steps}
        self._fit_data = {}
        self._predict_data = {}

    # copied from sklearn
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params('steps', deep=deep)

    def fit(self, *pargs, **kwargs):
        """

        Args:
            pargs:
            kwargs:
        """
        if len(pargs) == 0:
            external_data = {}
        elif len(pargs) == 1:
            external_data = {('_External', 'X'): pargs[0]}
        elif len(pargs) == 2:
            external_data = {('_External', 'X'): pargs[0],
                             ('_External', 'y'): pargs[1]
                             }
        else:
            raise ValueError("The developer never thought that 3 positional parameters were possible."
                             "Try using keyword parameters from the third on.")

        external_data.update({('_External', key): item
                              for key, item in kwargs.items()})
        self._fit_data = external_data
        fit_nodes = self._filter_fit_nodes()
        for step_name in fit_nodes:
            logger.debug('Fitting: %s', step_name)
            self._fit(step_name)

    def _fit(self, step_name):
        """

        Args:
            step_name:
        """
        logger.debug("under_Fitting: %s", step_name)
        fit_inputs = self._read_fit_signature_variables_from_graph_data(graph_data=self._fit_data,
                                                                        step_name=step_name)
        self._step[step_name].fit(**fit_inputs)

        predict_inputs = self._read_predict_signature_variables_from_graph_data(graph_data=self._fit_data,
                                                                                step_name=step_name)
        results = self._step[step_name].predict(**predict_inputs)

        self._write_step_outputs(graph_data=self._fit_data,
                                 step_name=step_name,
                                 output_data=results)

    def predict(self, *pargs, **kwargs):
        """

        Args:
            pargs:
            kwargs:

        Returns:

        """
        if len(pargs) == 0:
            external_data = {}
        elif len(pargs) == 1:
            external_data = {('_External', 'X'): pargs[0]}
        elif len(pargs) == 2:
            external_data = {('_External', 'X'): pargs[0],
                             ('_External', 'y'): pargs[1]
                             }
        else:
            raise ValueError("The developer never thought that 3 positional parameters were possible. "
                             "Try using keyword parameters from the third on.")

        external_data.update({('_External', key): item for key, item in kwargs.items()})
        self._predict_data = external_data
        predict_nodes = self._filter_predict_nodes()

        for name in predict_nodes:
            logger.debug("Predicting: %s", name)
            self._predict(name)

        desired_output_step = self.steps[-1][0]
        result_dict = {label: self._predict_data.get((desired_output_step, label), None)
                       for label in {'predict', 'predict_proba', 'predict_log_proba'}}
        return result_dict

    def _predict(self, step_name):
        """

        Args:
            step_name:
        """

        logger.debug("under_Predicting: %s", step_name)
        predict_inputs = self._read_predict_signature_variables_from_graph_data(graph_data=self._predict_data,
                                                                                step_name=step_name)
        results = self._step[step_name].predict(**predict_inputs)
        self._write_step_outputs(graph_data=self._predict_data, step_name=step_name, output_data=results)

    @property
    def named_steps(self):
        """

        Returns:

        """
        return Bunch(**dict(self.steps))

    def _filter_fit_nodes(self):
        """

        Returns:

        """
        return (name for name in nx.topological_sort(self._fit_graph))

    def _filter_predict_nodes(self):
        """

        Returns:

        """
        return (name for name in nx.topological_sort(self._predict_graph))

    def _read_fit_signature_variables_from_graph_data(self, graph_data, step_name):
        """

        Args:
            step_name:

        Returns:

        """
        connections = self.connections if graph_data is self._fit_data else self.alternative_connections

        variable_list = self._step[step_name]._get_fit_signature()

        connection_tuples = {key: ('_External', value) if isinstance(value, str) else value
                             for key, value in connections[step_name].items()}

        if 'kwargs' in variable_list:
            input_data = {inner_variable: graph_data[node_and_outer_variable_tuple]
                          for inner_variable, node_and_outer_variable_tuple in connection_tuples.items()}
        else:
            input_data = {inner_variable: graph_data[node_and_outer_variable_tuple]
                          for inner_variable, node_and_outer_variable_tuple in connection_tuples.items()
                          if inner_variable in variable_list}

        return input_data

    def _read_predict_signature_variables_from_graph_data(self, graph_data, step_name):
        """

        Args:
            step_name:

        Returns:

        """
        connections = self.connections if graph_data is self._fit_data else self.alternative_connections

        variable_list = self._step[step_name]._get_predict_signature()

        connection_tuples = {key: ('_External', value) if isinstance(value, str) else value
                             for key, value in connections[step_name].items()}

        if variable_list == ['kwargs']:
            input_data = {inner_variable: graph_data[node_and_outer_variable_tuple]
                          for inner_variable, node_and_outer_variable_tuple in connection_tuples.items()}
        else:
            input_data = {inner_variable: graph_data[node_and_outer_variable_tuple]
                          for inner_variable, node_and_outer_variable_tuple in connection_tuples.items()
                          if inner_variable in variable_list}

        return input_data

    def _write_step_outputs(self, graph_data, step_name, output_data):
        """

        Args:
            node_name:
            data_dict:
        """
        graph_data.update(
            {(step_name, variable): value for variable, value in output_data.items()}
        )


class Step(BaseEstimator):
    """
    """

    def __init__(self, strategy):
        """

        Args:
            strategy:
        """
        self._strategy = strategy

    def get_params(self):
        """

        Returns:

        """
        return self._strategy.get_params()

    def set_params(self, **params):
        """

        Args:
            params:

        Returns:

        """
        self._strategy.set_params(**params)
        return self

    def fit(self, *pargs, **kwargs):
        """

        Args:
            pargs:
            kwargs:

        Returns:

        """
        self._strategy.fit(*pargs, **kwargs)
        return self

    def __getattr__(self, name):
        """

        Args:
            name:

        Returns:

        """
        return getattr(self.__dict__['_strategy'], name)

    def __setattr__(self, name, value):
        """

        Args:
            name:
            value:
        """
        if name in ('_strategy',):
            self.__dict__[name] = value
        else:
            setattr(self.__dict__['_strategy'], name, value)

    def __delattr__(self, name):
        """

        Args:
            name:
        """
        delattr(self.__dict__['_strategy'], name)

    def __repr__(self):
        """

        Returns:

        """
        return self._strategy.__repr__()


class StepStrategy(BaseEstimator):
    """
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
        """ document """

    def _get_fit_signature(self):
        """

        Returns:

        """
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


class FitTransform(StepStrategy):
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
        return list(inspect.signature(self._adaptee.transform).parameters)


class FitPredict(StepStrategy):
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
            result['predict_proba'] = self._adaptee.predict_proba(**kwargs)
        if hasattr(self._adaptee, 'predict_log_proba'):
            result['predict_log_proba'] = self._adaptee.predict_log_proba(**kwargs)
        return result

    def _get_predict_signature(self):
        """

        Returns:

        """
        return list(inspect.signature(self._adaptee.predict).parameters)


class AtomicFitPredict(StepStrategy):
    """
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
        return self._get_fit_signature()


class CustomStrategyWithDictionaryOutputAdaptee(StepStrategy):
    """
    """

    def fit(self, *pargs, **kwargs):
        """

        Args:
            pargs:
            kwargs:

        Returns:

        """
        self._adaptee.fit(*pargs, **kwargs)
        return self

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
        return list(inspect.signature(self._adaptee.predict).parameters)


class Concatenator(BaseEstimator):
    """
    """

    def fit(self):
        return self

    def predict(self, **kwargs):
        df_list = []
        for item in kwargs.values():
            if isinstance(item, pd.Series) or isinstance(item, pd.DataFrame):
                df_list.append(item)
            else:
                df_list.append(pd.DataFrame(data=item))
        return pd.concat(df_list, axis=1)


class CustomCombination(BaseEstimator):
    """
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
    def __init__(self, test_size=0.25, train_size=None, random_state=None):
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def fit(self, *pargs, **kwargs):
        return self

    def predict(self, *pargs, **kwargs):
        if len(pargs) > 0:
            raise ValueError("The developers assume you will use keyword parameters on the TrainTestSplit class.")
        array_names = list(kwargs.keys())
        train_test_array_names = sum([[item + "_train", item + "_test"] for item in array_names], [])

        result = dict(zip(train_test_array_names,
                          train_test_split(*kwargs.values())))
        return result


class ColumnSelector(BaseEstimator):
    def __init__(self, mapping=None):
        self.mapping = mapping

    def fit(self):
        return self

    def predict(self, X):
        if self.mapping is None:
            return {'predict': X}
        result = {name: X.iloc[:, column_slice]
                  for name, column_slice in self.mapping.items()}
        return result


class CustomPower(BaseEstimator):
    def __init__(self, power=1):
        self.power = power

    def fit(self):
        return self

    def predict(self, X):
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
        #list of arrays with the samen dimension
        selection = kwargs['selection']
        array_list = [pd.DataFrame(data=kwargs[str(class_number)],
                                   index=np.flatnonzero(selection == class_number))
                      for class_number in set(selection)]
        result = pd.concat(array_list, axis=0).sort_index()
        return result


def build_graph(connections):
    """

    Args:
        connections:

    Returns:

    """
    if '_External' in connections:
        raise ValueError("Please use another name for the _External node. _External name is used internally.")

    graph = nx.DiGraph()
    graph.add_nodes_from(connections)
    for name in connections:
        current_node = graph.node[name]
        current_node['name'] = name
        ascendants_set = set(value[0]
                             for inner_variable, value in connections[name].items()
                             if isinstance(value, tuple))
        for ascendant in ascendants_set:
            graph.add_edge(ascendant, name)
    #            logger.debug("Build graph %s", (ascendant, name))
    return graph


def make_step(adaptee, strategy_class=None):
    """

    Args:
        adaptee:

    Returns:

    """
    if strategy_class is not None:
        strategy = strategy_class(adaptee)
    elif adaptee.__class__ in strategies_for_custom_adaptees:
        strategy = strategies_for_custom_adaptees[adaptee.__class__](adaptee)
    elif hasattr(adaptee, 'transform'):
        strategy = FitTransform(adaptee)
    elif hasattr(adaptee, 'predict'):
        strategy = FitPredict(adaptee)
    elif hasattr(adaptee, 'fit_predict'):
        strategy = AtomicFitPredict(adaptee)
    else:
        raise ValueError('Error: adaptee of unknown behaviour')

    step = Step(strategy)
    return step


strategies_for_custom_adaptees = {
    TrainTestSplit: CustomStrategyWithDictionaryOutputAdaptee,
    ColumnSelector: CustomStrategyWithDictionaryOutputAdaptee,
    Multiplexer: FitPredict,
    Demultiplexer: CustomStrategyWithDictionaryOutputAdaptee,
}
