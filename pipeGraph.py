# -*- coding: utf-8 -*-
import logging
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
        self.graph = _create_graph_from_description(steps,
                                                    connections,
                                                    use_for_fit=use_for_fit,
                                                    use_for_predict=use_for_predict)

    def fit(self):
        """
        Iterates over the graph steps in topological order and executes their fit method
        """
        fit_nodes = self._filter_nodes(filter='fit')
        for node in fit_nodes:
            logger.debug("Fitting: %s", node.name)
            input = self._read_connections(node.name)
            fit_dict = dict.fromkeys(node['step'].get_fit_parameters_from_signature())
            predict_dict = dict.fromkeys(node['step'].get_predict_parameters_from_signature())
            node['step'].fit(**fit_dict)
            results = node['step'].predict(**predict_dict)
            self._update_graph_data(self, data_dict)

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
        return [(name, self.graph.node[name]['step'])
                for name in nx.topological_sort(self.graph)]

    def _filter_nodes(self, filter):
        return (self.graph.node[name] for name in nx.topological_sort(self.graph)
                if filter in self.graph.node[name]['step'].use_for)

    def _read_connections(self, step_name):
        if step_name == 'First':
            return self.connections[step_name]
        else:
            input_data = {inner_variable: self.data[node_and_outer_variable_tuple]
                          for inner_variable, node_and_outer_variable_tuple in self.connections[step_name].items()}
            return input_data

    def _update_graph_data(self, data_dict):
        self.pipegraph.data.update(
            {(self.node_name, variable): value for variable, value in data_dict.items()}
        )


################################
#  Strategies
################################

class StepStrategy(BaseEstimator):
    """
    Adapter to provide a common interface to fit(X,y), fit(X), etc.
    """

    @abstractmethod
    def fit(self, **kwargs):
        """ Adapt fit methods"""

    @abstractmethod
    def predict(self, **kwargs):
        """ Adapt predict, transform, fit_predict"""

    @abstractmethod
    def get_fit_parameters_from_signature(self):
        """ For easier fit params passing"""

    @abstractmethod
    def get_predict_parameters_from_signature(self):
        """ For easier predict params passing"""

    def __init__(self, adaptee):
        self.adaptee = adaptee

    def get_params(self, deep=True):
        return self.adaptee.get_params(deep=deep)

    def set_params(self, **params):
        self.adaptee.set_params(**params)
        return self

    def __repr__(self):
        return self.adaptee.__repr__()


class FitTransformStrategy(StepStrategy):
    def fit(self, **kwargs):
        self.adaptee.fit(**kwargs)
        return self

    def predict(self, **kwargs):
        return self.adaptee.transform(**kwargs)

    def get_fit_parameters_from_signature(self):
        return inspect.signature(self.fit).parameters

    def get_predict_parameters_from_signature(self):
        return inspect.signature(self.transform).parameters


class FitPredictStrategy(StepStrategy):
    def fit(self, **kwargs):
        self.adaptee.fit(**kwargs)
        return self

    def predict(self, **kwargs):
        return self.adaptee.predict(**kwargs)

    def get_fit_parameters_from_signature(self):
        return inspect.signature(self.fit).parameters

    def get_predict_parameters_from_signature(self):
        return inspect.signature(self.predict).parameters


class AtomicFitPredictStrategy(StepStrategy):
    def fit(self, **kwargs):
        self.adaptee.fit_predict(**kwargs)
        return self

    def predict(self, **kwargs):
        return self.adaptee.fit_predict(**kwargs)

    def get_fit_parameters_from_signature(self):
        return inspect.signature(self.fit_predict).parameters

    def get_predict_parameters_from_signature(self):
        return self.get_fit_parameters_from_signature()


################################
#  STEP
################################
class Step(BaseEstimator):

    def __init__(self, strategy):
        self.strategy = strategy

    def get_params(self, deep=True): return self.strategy.get_params(deep=deep)

    def set_params(self, **params):
        self.strategy.set_params(**params)
        return self

    def fit(self, **kwargs): self.strategy.fit(**kwargs)

    def predict(self, **kwargs): self.strategy.predict(**kwargs)

    def get_fit_parameters_from_signature(self): return self.strategy.get_fit_parameters_from_signature()

    def get_predict_parameters_from_signature(self): return self.strategy.get_predict_parameters_from_signature()



################################
#  Custom models
################################

class FirstNodeModel(BaseEstimator):
    def fit(self, **kwargs):
        return self

    def predict(self, **kwargs):
        return self


class CustomConcatenation(BaseEstimator):
    def fit(self, df1, df2): return self
    def predict(self, df1, df2): return pd.concat([df1, df2], axis=1)


class CustomCombination(BaseEstimator):
    def fit(self, dominant, other): return self
    def predict(self, dominant, other): return np.where(dominant < 0, dominant, other)


class CustomPaella(BaseEstimator):
    def __init__(self, **kwargs):  super().__init__(adaptee=Paella(**kwargs))

    def fit(self, X, y): self.adaptee.fit(X, y)
    def predict(self, X, y): return self.adaptee.transform(X, y)


########################################

def _create_graph_from_description(steps, connections, use_for_fit='all', use_for_predict='all'):
    steps = [('First', FirstNodeModel())] + steps
    node_names = [name for name, model in steps]

    if use_for_fit == 'all':
        use_for_fit=node_names

    if use_for_predict == 'all':
        use_for_predict=node_names

    use_for = {name:[] for name in node_names}
    for name in node_names:
        if name in use_for_fit:
            use_for[name].append('fit')
        if name in use_for_predict:
            use_for[name].append('predict')


    graph = nx.DiGraph()
    graph.add_nodes_from(node_names)
    for name, step_model in steps:
        current_node = graph.node[name]
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
        strategy=FitTransformStrategy(adaptee)
    elif hasattr(adaptee, 'predict'):
        strategy=FitPredictStrategy(adaptee)
    elif hasattr(adaptee, 'fit_predict'):
        strategy = AtomicFitPredictStrategy(adaptee)

    step=Step(strategy)
    return step
