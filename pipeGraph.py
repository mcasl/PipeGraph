# -*- coding: utf-8 -*-
import logging
import inspect

from abc import abstractmethod

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import _BaseComposition

from paella import Paella

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# TO-DO: remove Paella's study case related classes

class PipeGraph(_BaseComposition):
    def __init__(self, steps, connections, use_for_fit='all', use_for_predict='all'):
        self.connections = connections
        self.data = dict()
        self._graph = build_graph(steps,
                                  connections,
                                  use_for_fit=use_for_fit,
                                  use_for_predict=use_for_predict)

    def fit(self):
        fit_nodes = self._filter_nodes(filter='fit')
        for node in fit_nodes:
            current_step = node['step']
            current_name = node['name']
            logger.debug("Fitting: %s", current_name)
            input = self._read_connections(current_name)
            #           fit_dict = dict.fromkeys(current_step.get_fit_parameters_from_signature())
            #           predict_dict = dict.fromkeys(current_step.get_predict_parameters_from_signature())
            #           current_step.fit(**fit_dict)
            results = current_step.predict(**predict_dict)
            self._update_graph_data(current_name, {'output': results})

    def predict(self):
        """
        Iterates over the graph steps in topological order and executes their run method
        """
        predict_nodes = self._filter_nodes(filter='predict')
        for node in predict_nodes:
            logger.debug("Predicting: %s", node.name)
            node['step'].predict(**input_data)

    @property
    def steps(self):
        return [(name, self._graph.node[name]['step'])
                for name in nx.topological_sort(self._graph)]

    def _filter_nodes(self, filter):
        return (self._graph.node[name] for name in nx.topological_sort(self._graph)
                if filter in self._graph.node[name]['use_for'])

    def _read_connections(self, step_name):
        if step_name == 'First':
            return self.connections[step_name]
        else:
            input_data = {inner_variable: self.data[node_and_outer_variable_tuple]
                          for inner_variable, node_and_outer_variable_tuple in self.connections[step_name].items()}
            return input_data

    def _update_graph_data(self, node_name, data_dict):
        self.pipegraph.data.update(
            {(node_name, variable): value for variable, value in data_dict.items()}
        )


################################
#  STEP
################################

class Step(BaseEstimator):

    def __init__(self, strategy):
        self._strategy = strategy

    def get_params(self):
        return self._strategy.get_params()

    def set_params(self, **params):
        self._strategy.set_params(**params)
        return self

    def fit(self, **kwargs):
        self._strategy.fit(**kwargs)
        return self

    # def predict(self, **kwargs):
    #     return self._strategy.predict(**kwargs)
    #
    # def get_fit_parameters_from_signature(self):
    #     return self._strategy.get_fit_parameters_from_signature()
    #
    # def get_predict_parameters_from_signature(self):
    #     return self._strategy.get_predict_parameters_from_signature()
    #
    def __getattr__(self, name):
        return getattr(self.__dict__['_strategy'], name)

    def __setattr__(self, name, value):
        if name in ('_strategy',):
            self.__dict__[name] = value
        else:
            setattr(self.__dict__['_strategy'], name, value)

    def __delattr__(self, name):
        delattr(self.__dict__['_strategy'], name)


################################
#  Strategies
################################

class StepStrategy(BaseEstimator):
    def __init__(self, adaptee):
        self._adaptee = adaptee

    def fit(self, **kwargs):
        self._adaptee.fit(**kwargs)
        return self

    @abstractmethod
    def predict(self, **kwargs):
        """ document """

    def get_fit_parameters_from_signature(self):
        return list(inspect.signature(self._adaptee.fit).parameters)

    @abstractmethod
    def get_predict_parameters_from_signature(self):
        """ For easier predict params passing"""

    # These two methods work by introspection, do not remove because the __getattr__ trick does not work with them
    def get_params(self, deep=True):
        return self._adaptee.get_params(deep=deep)

    def set_params(self, **params):
        self._adaptee.set_params(**params)
        return self

    def __getattr__(self, name):
        return getattr(self.__dict__['_adaptee'], name)

    def __setattr__(self, name, value):
        if name in ('_adaptee',):
            self.__dict__[name] = value
        else:
            setattr(self.__dict__['_adaptee'], name, value)

    def __delattr__(self, name):
        delattr(self.__dict__['_adaptee'], name)


class FitTransformStrategy(StepStrategy):
    def predict(self, **kwargs):
        result = {'predict': self._adaptee.transform(**kwargs)}
        return result

    def get_predict_parameters_from_signature(self):
        return list(inspect.signature(self._adaptee.transform).parameters)


class FitPredictStrategy(StepStrategy):
    def predict(self, **kwargs):
        result = {'predict': self._adaptee.predict(**kwargs)}
        if hasattr(self._adaptee, 'predict_proba'):
            result['predict_proba'] = self._adaptee.predict_proba(**kwargs)
        return result

    def get_predict_parameters_from_signature(self):
        return list(inspect.signature(self._adaptee.predict).parameters)


class AtomicFitPredictStrategy(StepStrategy):
    def fit(self, **kwargs):
        return self

    # Relies on the pipegraph iteration loop to run predict after fit in order to propagate the signals

    def predict(self, **kwargs):
        return {'predict': self._adaptee.fit_predict(**kwargs)}

    def get_predict_parameters_from_signature(self):
        return self.get_fit_parameters_from_signature()


################################
#  Custom models
################################

class FirstNodeModel(BaseEstimator):
    def fit(self, **kwargs):
        return self

    def predict(self, **kwargs):
        return None


class CustomConcatenation(BaseEstimator):
    def fit(self, df1, df2): return self

    def predict(self, df1, df2): return pd.concat([df1, df2], axis=1)


class CustomCombination(BaseEstimator):
    def fit(self, dominant, other):
        return self

    def predict(self, dominant, other):
        return np.where(dominant < 0, dominant, other)


class CustomPaella(BaseEstimator):
    def __init__(self, **kwargs):
        self._adaptee = Paella(**kwargs)

    def fit(self, X, y):
        self._adaptee.fit(X, y)

    def predict(self, X, y):
        return self._adaptee.transform(X, y)


########################################

def build_graph(steps, connections, use_for_fit='all', use_for_predict='all'):
    steps.insert(0, ('First', FirstNodeModel()))
    node_names = [name for name, model in steps]

    if use_for_fit == 'all':
        use_for_fit = node_names
    else:
        use_for_fit.insert(0, 'First')

    if use_for_predict == 'all':
        use_for_predict = node_names
    else:
        use_for_predict.insert(0, 'First')

    use_for = {name: [] for name in node_names}
    for name in node_names:
        if name in use_for_fit:
            use_for[name].append('fit')
        if name in use_for_predict:
            use_for[name].append('predict')

    graph = nx.DiGraph()
    graph.add_nodes_from(node_names)
    for name, step_model in steps:
        current_node = graph.node[name]
        current_node['name'] = name
        current_node['step'] = make_step(adaptee=step_model)
        current_node['use_for'] = use_for[name]
        if name != 'First':
            for ascendant in set(node_and_outer_variable_tuple[0]
                                 for inner_variable, node_and_outer_variable_tuple in
                                 connections[name].items()):
                graph.add_edge(ascendant, name)
    return graph


def make_step(adaptee):
    if hasattr(adaptee, 'transform'):
        strategy = FitTransformStrategy(adaptee)
    elif hasattr(adaptee, 'predict'):
        strategy = FitPredictStrategy(adaptee)
    elif hasattr(adaptee, 'fit_predict'):
        strategy = AtomicFitPredictStrategy(adaptee)
    else:
        raise ValueError('Error: adaptee of unknown behaviour')

    step = Step(strategy)
    return step
