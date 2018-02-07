# -*- coding: utf-8 -*-
import logging
import inspect

from abc import abstractmethod

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch
from sklearn.utils.metaestimators import _BaseComposition

from paella import Paella

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


#TODO: remove Paella's study case related classes

class PipeGraph(_BaseComposition):
    def __init__(self, steps, connections, use_for_fit='all', use_for_predict='all'):
        self.steps = steps
        self.connections = connections
        self.data = dict()

        node_names = [name for name, model in steps]
        if '_External' in node_names:
            raise ValueError("Please use another name for the _External node. _External name is used internally.")

        if use_for_fit == 'all':
            use_for_fit = node_names
        elif use_for_fit is None:
            use_for_fit = []

        if use_for_predict == 'all':
            use_for_predict = node_names
        elif use_for_predict is None:
            use_for_predict = []

        self._use_for = {name: [] for name in node_names}
        for name in node_names:
            if name in use_for_fit:
                self._use_for[name].append('fit')
            if name in use_for_predict:
                self._use_for[name].append('predict')

        self._graph = build_graph(steps, connections)

        self._step = {name: make_step(adaptee=step_model) for name, step_model in steps}

    def fit(self, **kwargs):
        external_data = {('_External', key): item for key, item in kwargs.items()}
        self.data.update(external_data)
        fit_nodes = self._filter_nodes(filter='fit')
        for name in fit_nodes:
            logger.debug('Fitting: %s', name)
            self._fit(name)

    def _fit(self, step_name):
        current_step = self._step[step_name]
        logger.debug("under_Fitting: %s", step_name)
        input_data = self._read_from_connections(step_name)
        fit_dict = self._filter_data(input_data, current_step._get_fit_parameters_from_signature())
        predict_dict = self._filter_data(input_data, current_step._get_predict_parameters_from_signature())

        current_step.fit(**fit_dict)
        results = current_step.predict(**predict_dict)
        self._update_graph_data(step_name,  results)

    def predict(self, **kwargs):
        external_data = {('_External', key): item for key, item in kwargs.items()}
        self.data.update(external_data)
        predict_nodes = self._filter_nodes(filter='predict')
        for name in predict_nodes:
            logger.debug("Predicting: %s", name)
            self._predict(name)

    def _predict(self, step_name):
        current_step = self._step[step_name]
        logger.debug("under_Predicting: %s", step_name)
        input_data = self._read_from_connections(step_name)
        predict_dict = self._filter_data(input_data, current_step._get_predict_parameters_from_signature())
        results = current_step.predict(**predict_dict)
        self._update_graph_data(step_name,  results)

    @property
    def named_steps(self):
        return Bunch(**dict(self.steps))

    def _filter_nodes(self, filter):
        return (name for name in nx.topological_sort(self._graph)
                if filter in self._use_for[name])

    def _read_from_connections(self, step_name):
        connection_tuples = {key: ('_External', value) if isinstance(value, str) else value
                             for key, value in self.connections[step_name].items()}

        input_data = {inner_variable: self.data[node_and_outer_variable_tuple]
                      for inner_variable, node_and_outer_variable_tuple in connection_tuples.items()}

        return input_data

    def _update_graph_data(self, node_name, data_dict):
        self.data.update(
            {(node_name, variable): value for variable, value in data_dict.items()}
        )

    def _filter_data(self, data_dict, variable_list):
        result = { k: data_dict.get(k,None) for k in variable_list}
        return result

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

    def __repr__(self):
        return self._strategy.__repr__()

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

    def _get_fit_parameters_from_signature(self):
        return list(inspect.signature(self._adaptee.fit).parameters)

    @abstractmethod
    def _get_predict_parameters_from_signature(self):
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

    def __repr__(self):
        return self._adaptee.__repr__()

class FitTransformStrategy(StepStrategy):
    def predict(self, **kwargs):
        result = {'predict': self._adaptee.transform(**kwargs)}
        return result

    def _get_predict_parameters_from_signature(self):
        return list(inspect.signature(self._adaptee.transform).parameters)


class FitPredictStrategy(StepStrategy):
    def predict(self, **kwargs):
        result = {'predict': self._adaptee.predict(**kwargs)}
        if hasattr(self._adaptee, 'predict_proba'):
            result['predict_proba'] = self._adaptee.predict_proba(**kwargs)
        return result

    def _get_predict_parameters_from_signature(self):
        return list(inspect.signature(self._adaptee.predict).parameters)


class AtomicFitPredictStrategy(StepStrategy):
    def fit(self, **kwargs):
        return self

    # Relies on the pipegraph iteration loop to run predict after fit in order to propagate the signals

    def predict(self, **kwargs):
        return {'predict': self._adaptee.fit_predict(**kwargs)}

    def _get_predict_parameters_from_signature(self):
        return self._get_fit_parameters_from_signature()


################################
#  Custom models
################################

class CustomConcatenation(BaseEstimator):
    def fit(self, df1, df2):
        return self

    def predict(self, df1, df2):
        return pd.concat([df1, df2], axis=1)


class CustomCombination(BaseEstimator):
    def fit(self, dominant, other):
        return self

    def predict(self, dominant, other):
        return np.where(dominant < 0, dominant, other)


########################################

def build_graph(steps, connections):
    node_names = [name for name, model in steps]
    if '_External' in node_names:
        raise ValueError("Please use another name for the _External node. _External name is used internally.")

    graph = nx.DiGraph()
    graph.add_nodes_from(node_names)
    for name, step_model in steps:
        current_node = graph.node[name]
        current_node['name'] = name
        ascendants_set = set( value[0]
                              for inner_variable, value in connections[name].items()
                              if isinstance(value, tuple))
        for ascendant in ascendants_set:
            graph.add_edge(ascendant, name)
#            logger.debug("Build graph %s", (ascendant, name))
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
