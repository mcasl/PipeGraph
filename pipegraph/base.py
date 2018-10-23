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

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

from sklearn.utils import Bunch
from sklearn.utils.metaestimators import _BaseComposition
from pipegraph.adapters import AdapterMixins

import logging
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


class PipeGraph(_BaseComposition):
    """PipeGraph class holds the steps, connections and graphs needed to perform graph-like fits and predicts.

    Parameters:
    -----------
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
        that uses a different connections dictionary during fit than during predict. The default value, ``None``, implies that it is equivalent to the ``connections`` value, and thus PipeGraph uses the same graph for both ``fit`` and ``predict``.

    log_level: int
        Log level for traceability purposes. This is yet a unimplemented feature.

    """

    def __init__(self, steps, fit_connections=None, predict_connections=None, log_level=None):
        self.steps = steps
        self.fit_connections = fit_connections
        self.predict_connections = predict_connections if predict_connections is not None else fit_connections
        self.log_level = log_level
        self._fit_graph = None
        self._predict_graph = None
        self._fit_data = {}
        self._predict_data = {}

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

    # copied from sklearn
    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        self._set_params('steps', **kwargs)
        return self

    def fit(self, X, y=None, **kwargs):
        """
        Fit the PipeGraph steps one after the other and following the topological order of the graph defined by the connections attribute.

        Parameters
        ----------
        X: iterable object
            Input fit data.

        y : iterable, default=None
            Output fit data.

        **kwargs : dict of string -> object
            Other input data

        Returns
        -------
        self : PipeGraphClassifier
            This estimator
        """

        node_names = [name for name, model in self.steps]
        if '_External' in node_names:
            raise ValueError("Please use another name for the _External node. _External is used internally.")
        self._steps_dict = {name: add_mixins_to_step(step=step_model) for name, step_model in self.steps}

        external_data = {('_External', 'X'): X,
                         ('_External', 'y'): y
                        }

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

        self._fit_graph = build_graph(self.fit_connections)
        self._predict_graph = build_graph(self.predict_connections)

        fit_nodes = self._filter_fit_nodes()
        for step_name in fit_nodes:
            self._fit_single(step_name)
        return self

    def _fit_single(self, step_name):
        """

        Args:
            step_name:
        """
        fit_inputs = self._fetch_signature_values(graph_data=self._fit_data,
                                                  step_name=step_name,
                                                  method='fit')
        try:
            self._steps_dict[step_name].fit(**fit_inputs)
        except ValueError:
            print("ERROR: step.fit call ValueError!")

        predict_inputs = self._fetch_signature_values(graph_data=self._fit_data,
                                                      step_name=step_name,
                                                      method='predict')
        results = dict()
        try:
            results = self._steps_dict[step_name].predict_dict(**predict_inputs)
        except ValueError:
            print("ERROR: _fit.predict call ValueError!")


        self._write_step_outputs(graph_data=self._fit_data,
                                 step_name=step_name,
                                 output_data=results)

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
        return self.predict_dict(X, y=None)['predict']

    def predict_proba(self, X):
        """
        Applies PipeGraph's predict method and returns the predict_proba output of the final estimator

        Parameters
        ----------
        X: iterable object
            Data to predict on. Must fulfill input requirements of first step of the PipeGraph.

        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        return self.predict_dict(X, y=None)['predict_proba']

    def decision_function(self, X):
        """
        Applies PipeGraph's predict method and returns the decision_function output of the final estimator

         Parameters
         ----------
         X: iterable object
            Data to predict on. Must fulfill input requirements of first step of the PipeGraph.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """

        self.predict(X)
        return self.steps[-1][-1].decision_function(X)

    def predict_log_proba(self, X):
        """
        Applies PipeGraph's predict method and returns the predict_log_proba output of the final estimator

        Parameters
        ----------
        X: iterable object
            Data to predict on. Must fulfill input requirements of first step of the PipeGraph

        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        return self.predict_dict(X, y=None)['predict_log_proba']

    def predict_dict(self, X, **kwargs):
        """
        Predict the PipeGraph steps one after the other and following the topological
        order defined by the alternative_connections attribute, in case it is not None, or the connections attribute otherwise.

        Parameters
        ----------
        X :
            Input data
        **kwargs : dict of string -> object
             Arbitrary number of keyword arguments.

        Returns
        -------
        result_dict : dict
             Dictionary containing a single or multiple outputs
        """
        external_data = {('_External', 'X'): X,
                        }

        external_data.update({('_External', key): item for key, item in kwargs.items()})
        self._predict_data = external_data

        predict_nodes = self._filter_predict_nodes()

        for name in predict_nodes:
            self._predict_single_step(name)

        desired_output_step = self.steps[-1][0]
        result_dict = {label: self._predict_data.get((desired_output_step, label), None)
                       for label in {'predict', 'predict_proba', 'predict_log_proba'}}
        return result_dict

    def _predict_single_step(self, step_name):
        """

        Args:
            step_name:
        """

        predict_inputs = self._fetch_signature_values(graph_data=self._predict_data,
                                                      step_name=step_name,
                                                      method='predict')
        results =  {}
        try:
            results = self._steps_dict[step_name].predict_dict(**predict_inputs)
        except ValueError:
            print("ERROR: _predict call ValueError!")

        self._write_step_outputs(graph_data=self._predict_data, step_name=step_name, output_data=results)

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
        self.fit(X, y=y, **fit_params)
        return self.predict(X)

    def score(self, X, y=None, sample_weight=None):
        """
        Applies PipeGraph's predict method and returns the score output of the final estimator

        Parameters
        ----------
         X : iterable
             Data to predict on. Must fulfill input requirements of first step of the pipeGraph.
         y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all steps of the pipeGraph.
         sample_weight : array-like, default=None
            If not None, this argument is passed as sample_weight keyword argument to the score method of the final estimator.

         Returns
         -------
         y_proba : array-like, shape = [n_samples, n_classes]
         """
        self.predict(X)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        final_step_name, final_step = self.steps[-1]

        predict_inputs = self._fetch_signature_values(graph_data=self._predict_data,
                                                  step_name=final_step_name,
                                                  method='predict')
        Xt = predict_inputs['X']

        if y is None:
            node_and_outer_variable_tuple = self.predict_connections[final_step_name]['y']
            y = self._predict_data.get(node_and_outer_variable_tuple, None)

        return final_step.score(Xt, y, **score_params)

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

    def _fetch_signature_values(self, graph_data, step_name, method):
        """

        Args:
            step_name:

        Returns:
        :rtype: dict

        """
        connections = self.fit_connections if graph_data is self._fit_data else self.predict_connections
        connection_tuples = {}
        for key, value in connections[step_name].items():
            if not isinstance(value, str):
                connection_tuples.update({key: value})
            elif value in connections:
                connection_tuples.update({key: (value, 'predict')})
            else:
                connection_tuples.update({key: ('_External', value)})

        if isinstance(self._steps_dict[step_name], PipeGraph):
            input_data = {inner_variable: graph_data.get(node_and_outer_variable_tuple, None)
                          for inner_variable, node_and_outer_variable_tuple in connection_tuples.items()}
            return input_data

        if method == 'fit':
            variable_list = self._steps_dict[step_name]._get_fit_signature()
        elif method=='predict':
            variable_list = self._steps_dict[step_name]._get_predict_signature()

        if 'kwargs' in variable_list:
            input_data = {inner_variable: graph_data.get(node_and_outer_variable_tuple, None)
                          for inner_variable, node_and_outer_variable_tuple in connection_tuples.items()}
            return input_data

        input_data = {inner_variable: graph_data.get(node_and_outer_variable_tuple, None)
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



def add_mixins_to_step(step, mixin=None):
    """
    This function adds mixin classes to models in order to provide a common interface for them all.
    This interface declares two main methods: fit and predict. So, no matter whether the adaptee is capable of doing
    predict, transform or fit_predict, once the mixin class is added the model is capable
    of using pg_predict as method for producing output.

    Parameters:
    -----------
        step: a Scikit-Learn object, for instance; or a user made custom estimator may be.
        mixin: A user made class; or one already defined in pipegraph.adapters.

    Returns:
    -------
        The step object transformed by adding a mixing class.
    """

    if mixin is None:
        is_a_mixin_not_needed = isinstance(step, AdapterMixins) or  isinstance(step, PipeGraph)
        if is_a_mixin_not_needed:
            return step
        elif step.__class__ in strategies_for_custom_adaptees:
            mixin = strategies_for_custom_adaptees[step.__class__]
        elif hasattr(step, 'predict'):
            mixin = FitPredictMixin
        elif hasattr(step, 'transform'):
            mixin = FitTransformMixin
        elif hasattr(step, 'fit_predict'):
            mixin = AtomicFitPredictMixin
        else:
            raise ValueError('Error: Unknown step class!')
    elif isinstance(step, mixin):
        return step

    new_class_name = step.__class__.__name__
    step.__class__ = type(new_class_name, (type(step), mixin), {})
    return step


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
    def __init__(self, number_of_replicas=1, regressor=LinearRegression()):
        self.number_of_replicas = number_of_replicas
        self.regressor = regressor

        steps = ([('demux', Demultiplexer())] +
                 [('regressor_' + str(i), clone(regressor)) for i in range(number_of_replicas)] +
                 [('mux', Multiplexer())]
                 )

        connections = dict(demux={'X': 'X',
                                  'y': 'y',
                                  'selection': 'selection'})

        for i in range(number_of_replicas):
            connections['regressor_' + str(i)] = {'X': ('demux', 'X_' + str(i)),
                                               'y': ('demux', 'y_' + str(i))}

        connections['mux'] = {str(i): ('regressor_' + str(i)) for i in range(number_of_replicas)}
        connections['mux']['selection'] = 'selection'
        super().__init__(steps=steps, fit_connections=connections)


class RegressorsWithDataDependentNumberOfReplicas(PipeGraph, RegressorMixin):
    def __init__(self, steps=[('regressor', LinearRegression() )]):
        self.steps = steps

    def fit(self, *pargs, **kwargs):
        regressor = self.named_steps.regressor
        number_of_clusters = len(set(kwargs['selection']))
        multiple_regressors = RegressorsWithParametrizedNumberOfReplicas(number_of_replicas=number_of_clusters,
                                                                         regressor=regressor)
        steps = [('regressorsBundle', multiple_regressors)]
        connections = dict(regressorsBundle={'X': 'X', 'y': 'y', 'selection': 'selection'})
        self._pipegraph = PipeGraph(steps=steps, fit_connections=connections)
        self._pipegraph.fit(*pargs, **kwargs)
        return self

    def predict_dict(self, *pargs, **kwargs):
        return self._pipegraph.predict_dict(*pargs, **kwargs)


def query_number_of_clusters_from_classifier(classifier):

    number_of_clusters_dictionary = {
        'KMeans': 'n_clusters',
        'SpectralClustering': 'n_clusters',
        'AgglomerativeClustering': 'n_clusters',
        'GaussianMixture': 'n_components',
        'Birch': 'n_clusters',
    }

    classifier_class = type(classifier).__name__
    number_of_clusters_str = number_of_clusters_dictionary[classifier_class]
    number_of_cluster = classifier.get_params()[number_of_clusters_str]
    return number_of_cluster

class ClassifierAndRegressorsBundle(PipeGraph, RegressorMixin):
    def __init__(self, steps=[('classifier', GaussianMixture()), ('regressor', LinearRegression())]):
        self.steps = steps

    def fit(self, *pargs, **kwargs):
        classifier = self.named_steps.classifier
        regressor = self.named_steps.regressor
        number_of_clusters = query_number_of_clusters_from_classifier(classifier)
        multiple_regressors = RegressorsWithParametrizedNumberOfReplicas(number_of_replicas=number_of_clusters,
                                                                         regressor=regressor)

        steps = [('classifier', classifier), ('regressorsBundle', multiple_regressors)]
        connections = dict(classifier={'X': 'X'},
                           regressorsBundle={'X': 'X', 'y': 'y', 'selection': 'classifier'})
        self._adaptee = PipeGraph(steps=steps, fit_connections=connections)

        self._adaptee.fit(*pargs, **kwargs)
        return self

    def predict_dict(self, *pargs, **kwargs):
        return self._adaptee.predict_dict(*pargs, **kwargs)


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



from pipegraph.adapters import (FitTransformMixin,
                                FitPredictMixin,
                                AtomicFitPredictMixin,
                                CustomFitPredictWithDictionaryOutputMixin)

strategies_for_custom_adaptees = {
    NeutralRegressor:  FitPredictMixin,
    NeutralClassifier:  FitPredictMixin,
    Concatenator: FitPredictMixin,
    ColumnSelector: CustomFitPredictWithDictionaryOutputMixin,
    Reshape: FitPredictMixin,
    Demultiplexer: CustomFitPredictWithDictionaryOutputMixin,
    Multiplexer: FitPredictMixin,
}

from pipegraph.demo_blocks import strategies_for_demo_blocks_adaptees
strategies_for_custom_adaptees.update(strategies_for_demo_blocks_adaptees)

