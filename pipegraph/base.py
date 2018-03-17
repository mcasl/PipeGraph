# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

from sklearn.utils import Bunch
from sklearn.utils.metaestimators import _BaseComposition

import logging
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)

class PipeGraphRegressor(BaseEstimator, RegressorMixin):
    """
    PipeGraph with a regressor default score.
    This class implements an interface to the PipeGraph base class which is compatible with GridSearchCV.
    This documentation is heavily based on Scikit-Learn's Pipeline on purpose as its aim is to provide an interface
    as similar to Pipeline as possible.

    Parameters
    ----------
    steps : list
        List of (name, action) tuples that are chained. The last one is considered to be the output step.

    connections: dictionary
        A dictionary whose keys of the top level entries of the dictionary must the same as those of the previously defined steps. The values assocciated to these keys define the variables from other steps that are going to be considered as inputs for the current step. They are dictionaries themselves, where:

            - The keys of the nested dictionary represents the input variable as named at the current step.
            - The values associated to these keys define the steps that hold the desired information, and the variables as named at that step. This information can be written as:

                - A tuple with the label of the step in position 0 followed by the name of the output variable in position 1.
                - A string representing a variable from an external source to the PipeGraphRegressor object, such as those provided by the user while invoking the fit, predict or fit_predict methods.

    alternative_connections: dictionary
        A dictionary as described for ``connections``. This parameters provides the possibility of specifying a PipeGraph
        that uses a different connection dictionary during fit than during predict. The default value, ``None``, implies that it is equivalent to the ``connections`` value, and thus PipeGraph uses the same graph for both ``fit`` and ``predict``.

    log_level: int
        Log level for traceability purposes. This is yet a unimplemented feature.
    """
    def __init__(self, steps, fit_connections=None, predict_connections=None, log_level=None):
        self._pipegraph = PipeGraph(steps, fit_connections, predict_connections, log_level)

    def inject(self, sink, sink_var, source='_External', source_var='predict', into='fit'):
        """
        Adds a connection to the graph.

        Parameters:
        -----------
        sink: Destination
        sink_var: Name of the variable at destination that is going to hold the information
        source: Origin
        source_var: Name of the variable at origin holding the information
        into: This can be either 'fit' or 'predict', indicating which connections are described: those belonging
              to 'fit_connections' or those belonging to 'predict_connections'. Default is 'fit'.

        Returns:
        --------
        self:  PipeGraphRegressor
            Returning self allows chaining operations
        """
        self._pipegraph.inject(sink=sink, sink_var=sink_var, source=source, source_var=source_var, into=into)
        return self

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and its nested estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._pipegraph.get_params(deep=deep)

    def set_params(self, **kwargs):
        """
        Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        self._pipegraph.set_params(**kwargs)
        return self

    @property
    def named_steps(self):
        return self._pipegraph.named_steps

    def fit(self, X, y=None, **fit_params):
        """
        Fit the PipeGraph steps one after the other and following the topological order of the graph defined by the connections attribute.

        Parameters
        ----------
        X: iterable object
            Training data. Must fulfill input requirements of first step of the PipeGraph.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of the PipeGraph.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : PipeGraphRegressor
            This estimator
        """
        self._pipegraph.fit(X, y=y, **fit_params)
        return self

    def predict(self, X):
        """
        Predict the PipeGraph steps one after the other and following the topological
        order defined by the alternative_connections attribute, in case it is not None, or the connections attribute otherwise.

        Parameters
        ----------
        X: iterable object
            Data to predict on. Must fulfill input requirements of first step of the PipeGraph.

        Returns
        -------
        y_pred : array-like
        """
        return self._pipegraph.predict(X, y=None)['predict']

    def fit_predict(self, X, y=None, **fit_params):
        """
        This is equivalent to applying the fit method and then predict method.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the PipeGraph
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of the PipeGraph.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : array-like
        """
        self._pipegraph.fit(X, y=y, **fit_params)
        return self._pipegraph.predict(X, y=None)

    def predict_proba(self, X):
        """
        Applies PipeGraphRegressor's predict method and returns the predict_proba output of the final estimator

        Parameters
        ----------
        X: iterable object
            Data to predict on. Must fulfill input requirements of first step of the PipeGraph.

        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        return self._pipegraph.predict(X, y=None)['predict_proba']

    def decision_function(self, X):
        """
        Applies PipeGraphRegressor's predict method and returns the decision_function output of the final estimator

         Parameters
         ----------
         X: iterable object
            Data to predict on. Must fulfill input requirements of first step of the PipeGraph.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        self._pipegraph.predict(X)
        return self._pipegraph.steps[-1][-1].decision_function(X)

    def predict_log_proba(self, X):
        """
        Applies PipeGraphRegressor's predict method and returns the predict_log_proba output of the final estimator

        Parameters
        ----------
        X: iterable object
            Data to predict on. Must fulfill input requirements of first step of the PipeGraph

        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        return self._pipegraph.predict(X, y=None)['predict_log_proba']

    def score(self, X, y=None, sample_weight=None):
        """
        Applies PipeGraphRegressor's predict method and returns the score output of the final estimator

        Parameters
        ----------
         X : iterable
             Data to predict on. Must fulfill input requirements of first step of the pipeGrpaph.
         y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all steps of the pipeGrpaph.
         sample_weight : array-like, default=None
            If not None, this argument is passed as sample_weight keyword argument to the score method of the final estimator.

         Returns
         -------
         y_proba : array-like, shape = [n_samples, n_classes]
         """
        self._pipegraph.predict(X)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        final_step_name, final_step = self._pipegraph.steps[-1]

        predict_inputs = self._pipegraph._read_predict_signature_variables_from_graph_data(
            graph_data=self._pipegraph._predict_data,
            step_name=final_step_name)

        Xt=predict_inputs['X']
        return final_step.score(Xt, y, **score_params)


class PipeGraphClassifier(BaseEstimator, ClassifierMixin):
    """PipeGraph with a classifier default score.
    This class implements an interface to the PipeGraph base class which is compatible with GridSearchCV.
    This documentation is heavily based on Scikit-Learn's Pipeline on purpose as it is the aim of this class
    to provide an interface as similar to Pipeline as possible.

    Parameters
    ----------
    steps : list
        List of (name, action) tuples that are chained. The last one is considered to be the output step.

    connections: dictionary
        A dictionary whose keys of the top level entries of the dictionary must the same as those of the previously defined steps. The values assocciated to these keys define the variables from other steps that are going to be considered as inputs for the current step. They are dictionaries themselves, where:
            - The keys of the nested dictionary represents the input variable as named at the current step.
            - The values associated to these keys define the steps that hold the desired information, and the variables as named at that step. This information can be written as:
                - A tuple with the label of the step in position 0 followed by the name of the output variable in position 1.
                - A string representing a variable from an external source to the PipeGraphClassifier object, such as those provided by the user while invoking the fit, predict or fit_predict methods.

    alternative_connections: dictionary
        A dictionary as described for ``connections``. This parameters provides the possibility of specifying a PipeGraph
        that uses different connection dictionaries during fit than during predict. The default value, ``None``, implies that it is equivalent to the ``connections`` value, and thus PipeGraph uses the same graph for both ``fit`` and ``predict``.


    log_level: int
        Log level for traceability purposes. This is yet a unimplemented feature.

        """
    def __init__(self, steps, fit_connections=None, predict_connections=None, log_level=None):
        self._pipegraph = PipeGraph(steps, fit_connections, predict_connections, log_level)

    def inject(self, sink, sink_var, source='_External', source_var='predict', into='fit'):
        """
                Adds a connection to the graph.

                Parameters:
                -----------
                sink: Destination
                sink_var: Name of the variable at destination that is going to hold the information
                source: Origin
                source_var: Name of the variable at origin holding the information
                into: This can be either 'fit' or 'predict', indicating which connections are described: those belonging
                      to 'fit_connections' or those belonging to 'predict_connections'. Default is 'fit'.

                Returns:
                --------
                self:  PipeGraphClassifier
                    Returning self allows chaining operations
                """
        self._pipegraph.inject(sink=sink, sink_var=sink_var, source=source, source_var=source_var, into=into)
        return self

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and its nested estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._pipegraph.get_params(deep=deep)

    def set_params(self, **kwargs):
        """
        Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        self._pipegraph.set_params(**kwargs)
        return self

    @property
    def named_steps(self):
        return pipegraph.named_steps

    def fit(self, X, y=None, **fit_params):
        """
        Fit the PipeGraph steps one after the other and following the topological order of the graph defined by the connections attribute.

        Parameters
        ----------
        X: iterable object
            Training data. Must fulfill input requirements of first step of the PipeGraph.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of the PipeGraph.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : PipeGraphClassifier
            This estimator
        """
        self._pipegraph.fit(X, y=y, **fit_params)
        return self

    def predict(self, X):
        """
        Predict the PipeGraph steps one after the other and following the topological
        order defined by the alternative_connections attribute, in case it is not None, or the connections attribute otherwise.

        Parameters
        ----------
        X: iterable object
            Data to predict on. Must fulfill input requirements of first step of the PipeGraph.

        Returns
        -------
        y_pred : array-like
        """
        return self._pipegraph.predict(X, y=None)['predict']

    def fit_predict(self, X, y=None, **fit_params):
        """
        Applies fit_predict of last step in PipeGraph after it predicts the PipeGraph steps one after
        the other and following the topological order of the graph.

        Applies predict of a PipeGraph to the data following the topological order of the graph, followed by the
        fit_predict method of the final step in the PipeGraph. Valid only if the final step implements fit_predict.

        Parameters
        ----------
        X: iterable object
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
        self._pipegraph.fit(X, y=y, **fit_params)
        return self._pipegraph.predict(X)

    def predict_proba(self, X):
        """
        Applies PipeGraphClassifier's predict method and returns the predict_proba output of the final estimator

        Parameters
        ----------
        X: iterable object
            Data to predict on. Must fulfill input requirements of first step of the PipeGraph.

        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        return self._pipegraph.predict(X, y=None)['predict_proba']

    def decision_function(self, X):
        """
        Applies PipeGraphClasifier's predict method and returns the decision_function output of the final estimator

         Parameters
         ----------
         X: iterable object
            Data to predict on. Must fulfill input requirements of first step of the PipeGraph.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """

        self._pipegraph.predict(X)
        return self._pipegraph.steps[-1][-1].decision_function(X)

    def predict_log_proba(self, X):
        """
        Applies PipeGraphRegressor's predict method and returns the predict_log_proba output of the final estimator

        Parameters
        ----------
        X: iterable object
            Data to predict on. Must fulfill input requirements of first step of the PipeGraph

        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        return self._pipegraph.predict(X, y=None)['predict_log_proba']

    def score(self, X, y=None, sample_weight=None):
        """
        Applies PipeGraphRegressor's predict method and returns the score output of the final estimator

         Parameters
         ----------
         X : iterable
             Data to predict on. Must fulfill input requirements of first step of the pipeGrpaph.
         y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all steps of the pipeGrpaph.
         sample_weight : array-like, default=None
            If not None, this argument is passed as sample_weight keyword argument to the score method of the final estimator.

         Returns
         -------
         y_proba : array-like, shape = [n_samples, n_classes]
         """
        self._pipegraph.predict(X)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        final_step_name = self._pipegraph.steps[-1][0]
        Xt = self._pipegraph._predict_data[self._pipegraph.fit_connections[final_step_name]['X']]
        return self._pipegraph.steps[-1][-1].score(Xt, y, **score_params)


class PipeGraph(_BaseComposition):
    """Class in charge of holding the steps, connections and graphs needed to perform graph like
    fits and predicts.

    Parameters:
    -----------
    steps : list
        List of (name, action) tuples that are chained. The last one is considered to be the output step.

    connections: dictionary
        A dictionary whose keys of the top level entries of the dictionary must the same as those of the previously defined steps. The values assocciated to these keys define the variables from other steps that are going to be considered as inputs for the current step. They are dictionaries themselves, where:

        - The keys of the nested dictionary represents the input variable as named at the current step.
        - The values assocciated to these keys define the steps that hold the desired information, and the variables as named at that step. This information can be written as:

            - A tuple with the label of the step in position 0 followed by the name of the output variable in position 1.
            - A string representing a variable from an external source to the PipeGraphRegressor object, such as those provided by the user while invoking the fit, predict or fit_predict methods.

    alternative_connections: dictionary
        A dictionary as described for ``connections``. This parameters provides the possibility of specifying a PipeGraph
        that uses a different connections dictionary during fit than during predict. The default value, ``None``, implies that it is equivalent to the ``connections`` value, and thus PipeGraph uses the same graph for both ``fit`` and ``predict``.

    log_level: int
        Log level for traceability purposes. This is yet a unimplemented feature.

    """

    def __init__(self, steps, fit_connections=None, predict_connections=None, log_level=None):
        self.steps = steps
        self.fit_connections = fit_connections
        self.predict_connections = predict_connections if predict_connections is not None else fit_connections
        self.log_level = log_level

        node_names = [name for name, model in steps]
        if '_External' in node_names:
            raise ValueError("Please use another name for the _External node. _External is used internally.")

        self._fit_graph = None
        self._predict_graph = None
        self._processes = {name: wrap_adaptee_in_process(adaptee=step_model) for name, step_model in steps}
        self._fit_data = {}
        self._predict_data = {}

    def inject(self, sink, sink_var, source, source_var='predict', into='fit'):
        """
                Adds a connection to the graph.

                Parameters:
                -----------
                sink: Destination
                sink_var: Name of the variable at destination that is going to hold the information
                source: Origin
                source_var: Name of the variable at origin holding the information
                into: This can be either 'fit' or 'predict', indicating which connections are described: those belonging
                      to 'fit_connections' or those belonging to 'predict_connections'. Default is 'fit'.

                Returns:
                --------
                self:  PipeGraph
                    Returning self allows chaining operations
                """
        if self.fit_connections is None:
            self.fit_connections = dict()

        if self.predict_connections is None:
            self.predict_connections = dict()

        if into == 'fit':
            connections = self.fit_connections
        elif into == 'predict':
            connections = self.predict_connections
        else:
            raise ValueError('Error: into parameter must be either fit or predict.')

        if sink in connections:
            connections[sink].update({sink_var: (source, source_var)})
        else:
            connections[sink] = {sink_var: (source, source_var)}
        return self

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

        if self.fit_connections is None:
            self.fit_connections = make_connections_when_not_provided_to_init(steps=self.steps)
            self.predict_connections = self.fit_connections
            self._fit_graph = build_graph(self.fit_connections)
            self._predict_graph = self._fit_graph

        if self.predict_connections == {}:
            self.predict_connections = self.fit_connections

        if self._fit_graph is None:
            self._fit_graph = build_graph(self.fit_connections)
            self._predict_graph = build_graph(self.predict_connections)

        fit_nodes = self._filter_fit_nodes()
        for step_name in fit_nodes:
            self._fit(step_name)
        return self

    def _fit(self, step_name):
        """

        Args:
            step_name:
        """
        fit_inputs = self._read_fit_signature_variables_from_graph_data(graph_data=self._fit_data,
                                                                        step_name=step_name)
        self._processes[step_name].fit(**fit_inputs)

        predict_inputs = self._read_predict_signature_variables_from_graph_data(graph_data=self._fit_data,
                                                                                step_name=step_name)
        results = self._processes[step_name].predict(**predict_inputs)

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

        predict_inputs = self._read_predict_signature_variables_from_graph_data(graph_data=self._predict_data,
                                                                                step_name=step_name)
        results = self._processes[step_name].predict(**predict_inputs)
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
        connections = self.fit_connections if graph_data is self._fit_data else self.predict_connections

        variable_list = self._processes[step_name]._get_fit_signature()

        connection_tuples = {}
        for key, value in connections[step_name].items():
            if not isinstance(value, str):
                connection_tuples.update({key: value})
            elif value in connections:
                connection_tuples.update({key: (value, 'predict')})
            else:
                connection_tuples.update({key: ('_External', value)})


        if 'kwargs' in variable_list:
            input_data = {inner_variable: graph_data.get(node_and_outer_variable_tuple, None)
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
        connections = self.fit_connections if graph_data is self._fit_data else self.predict_connections

        variable_list = self._processes[step_name]._get_predict_signature()

        connection_tuples = {}
        for key, value in connections[step_name].items():
            if not isinstance(value, str):
                connection_tuples.update({key: value})
            elif value in connections:
                connection_tuples.update({key: (value, 'predict')})
            else:
                connection_tuples.update({key: ('_External', value)})

        if 'kwargs' in variable_list:
            input_data = {inner_variable: graph_data.get(node_and_outer_variable_tuple, None)
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


class Process(BaseEstimator):
    """
    This class holds an strategy which is in charge of fitting and predicting the PipeGraph steps.

    Parameters
    ----------
    strategy : pipegraph.adapters.AdapterForSkLearnLikeAdaptee
    A wrapper for a sklearn like object

    """

    def __init__(self, strategy):
        self._strategy = strategy

    def get_params(self):
        """
        Get parameters for this estimator or pipeGraph CustomBlock.
        Parameters
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._strategy.get_params()

    def set_params(self, **params):
        """"
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        such as pipeGraph). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        self._strategy.set_params(**params)
        return self

    def fit(self, *pargs, **kwargs):
        """
        Fit the model included in the step
        ----------
        *pargs : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        **kwargs : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
            self : returns an instance of _strategy.
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


def build_graph(connections):
    """

    Args:
        connections: dict
        A dictionary as described in PipeGraphRegressor and PipeGraphClassifier

    Returns:
        A graph ready to be fitted
    """
    if connections is None:
        return None
    elif '_External' in connections:
        raise ValueError("Please use another name for the _External node. _External name is used internally.")

    graph = nx.DiGraph()
    graph.add_nodes_from(connections)
    for name in connections:
        current_node = graph.node[name]
        current_node['name'] = name
        ascendants_set = set()
        for inner_variable, value in connections[name].items():
            if isinstance(value, tuple):
                ascendants_set.add(value[0])
            elif value in connections:
                ascendants_set.add(value)

        ascendants_set.discard('_External')
        for ascendant in ascendants_set:
            graph.add_edge(ascendant, name)
    return graph



def wrap_adaptee_in_process(adaptee, adapter_class=None):
    """
    This function wraps the objects defined in Pipegraph's steps parameters in order to provide a common interface for them all.
    This interface declares two main methods: fit and predict. So, no matter whether the adaptee is capable of doing
    predict, transform or fit_predict, once wrapped the adapter uses predict as method for producing output.

    Parameters:
    -----------
        adaptee: a Scikit-Learn object, for instance; or a user made custom estimator may be.
            The object to be wrapped.
        adapter_class: A user made class; or one already defined in pipegraph.adapters
            The wrapper.

    Returns:
    -------
        An object wrapped into a first adapter layer that provides a common fit and predict interface and then wrapped
        again in a second external layer using the Process class. Besides of being used by PipeGraph itself,
        the user can find this function useful for inserting a user made block as one of the steps
        in PipeGraph step's parameter.
    """
    if adapter_class is not None:
        strategy = adapter_class(adaptee)
    elif isinstance(adaptee, Process):
        return adaptee
    elif isinstance(adaptee, PipeGraph):
        strategy = AdapterForCustomFitPredictWithDictionaryOutputAdaptee(adaptee)
    elif adaptee.__class__ in strategies_for_custom_adaptees:
        strategy = strategies_for_custom_adaptees[adaptee.__class__](adaptee)
    elif hasattr(adaptee, 'transform'):
        strategy = AdapterForFitTransformAdaptee(adaptee)
    elif hasattr(adaptee, 'predict'):
        strategy = AdapterForFitPredictAdaptee(adaptee)
    elif hasattr(adaptee, 'fit_predict'):
        strategy = AdapterForAtomicFitPredictAdaptee(adaptee)
    else:
        raise ValueError('Error: Unknown adaptee!')

    process = Process(strategy)
    return process


def make_connections_when_not_provided_to_init(steps):
    names = [name for name, _ in steps]
    connections = dict()
    if len(names) == 0:
        ValueError("Error: No steps to chain. Please provide a list of steps to PipeGraph")

    for index, name in enumerate(names):
        if index == 0:
            connections[name] = {'X': 'X'}
        else:
            connections[name] = {'X': (names[index-1], 'predict')}

    connections[names[-1]].update({'y': 'y'})
    return connections


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


class Reshape(BaseEstimator):
    def __init__(self, dimension):
        self.dimension = dimension

    def fit(self, *pargs, **kwargs):
        return self

    def predict(self, X, *pargs, **kwargs):
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
        # list of arrays with the same dimension
        selection = kwargs['selection']
        array_list = [pd.DataFrame(data=kwargs[str(class_number)],
                                   index=np.flatnonzero(selection == class_number))
                      for class_number in set(selection)]
        result = pd.concat(array_list, axis=0).sort_index()
        return result


class RegressorsWithParametrizedNumberOfReplicas(PipeGraph, RegressorMixin):
    def __init__(self, number_of_replicas=1, model_prototype=LinearRegression(), model_parameters={}):
        self.number_of_replicas = number_of_replicas
        self.model_class = model_prototype
        self.model_parameters = model_parameters

        steps = ([('demux', Demultiplexer())] +
                 [('model_' + str(i), model_prototype.__class__(**model_parameters)) for i in range(number_of_replicas)] +
                 [('mux', Multiplexer())]
                 )

        connections = dict(demux={'X': 'X',
                                  'y': 'y',
                                  'selection': 'selection'})

        for i in range(number_of_replicas):
            connections['model_' + str(i)] = {'X': ('demux', 'X_' + str(i)),
                                               'y': ('demux', 'y_' + str(i))}

        connections['mux'] = {str(i): ('model_' + str(i)) for i in range(number_of_replicas)}
        connections['mux']['selection'] = 'selection'
        super().__init__(steps=steps, fit_connections=connections)


class RegressorsWithDataDependentNumberOfReplicas(PipeGraph, RegressorMixin):
    def __init__(self, model_prototype=LinearRegression(), model_parameters={}):
        self.model_prototype = model_prototype
        self.model_parameters = model_parameters
        self._fit_data = {}
        self._predict_data = {}
        self.steps = []

    def fit(self, *pargs, **kwargs):
        number_of_replicas = len(set(kwargs['selection']))
        self.steps = [('models', RegressorsWithParametrizedNumberOfReplicas(number_of_replicas=number_of_replicas,
                                                                            model_parameters=self.model_parameters))]

        self._processes = {name: wrap_adaptee_in_process(adaptee=step_model) for name, step_model in self.steps}

        self.fit_connections = dict(models={'X': 'X',
                                            'y': 'y',
                                            'selection': 'selection'})
        self.predict_connections = self.fit_connections
        self._fit_graph = build_graph(self.fit_connections)
        self._predict_graph = build_graph(self.predict_connections)
        super().fit(*pargs, **kwargs)
        return self


class ClassifierAndRegressorsBundle(PipeGraph, RegressorMixin):
    def __init__(self,
                 number_of_replicas=1,
                 classifier_prototype=GaussianMixture(),
                 classifier_parameters={},
                 model_prototype=LinearRegression(),
                 model_parameters={}):

        self.number_of_replicas = number_of_replicas
        self.classifier_prototype = classifier_prototype
        self.classifier_parameters = classifier_parameters
        self.model_prototype = model_prototype
        self.model_parameters = model_parameters

        steps = [('classifier', self.classifier_prototype.__class__(n_components=number_of_replicas, **classifier_parameters)) ,
                 ('models', RegressorsWithParametrizedNumberOfReplicas(number_of_replicas=number_of_replicas,
                                                                        model_prototype=model_prototype,
                                                                        model_parameters=model_parameters))]
        connections = dict(classifier={'X': 'X'},
                           models= {'X': 'X',
                                    'y': 'y',
                                    'selection': 'classifier'})
        super().__init__(steps=steps, fit_connections=connections)




class NeutralRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X):
        return self

    def predict(self, X):
        return X


class NeutralClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X):
        return self

    def predict(self, X):
        return X



from pipegraph.adapters import (AdapterForFitTransformAdaptee,
                                AdapterForFitPredictAdaptee,
                                AdapterForAtomicFitPredictAdaptee,
                                AdapterForCustomFitPredictWithDictionaryOutputAdaptee,
                                )
strategies_for_custom_adaptees = {
    Concatenator: AdapterForFitPredictAdaptee,
    ColumnSelector: AdapterForCustomFitPredictWithDictionaryOutputAdaptee,
    Reshape: AdapterForFitPredictAdaptee,
    Demultiplexer: AdapterForCustomFitPredictWithDictionaryOutputAdaptee,
    Multiplexer: AdapterForFitPredictAdaptee,
    RegressorsWithParametrizedNumberOfReplicas: AdapterForCustomFitPredictWithDictionaryOutputAdaptee,
    RegressorsWithDataDependentNumberOfReplicas: AdapterForCustomFitPredictWithDictionaryOutputAdaptee,
}

from pipegraph.demo_blocks import strategies_for_demo_blocks_adaptees
strategies_for_custom_adaptees.update(strategies_for_demo_blocks_adaptees)

