# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx

from abc import ABC, abstractmethod
from collections import Iterator
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.base import BaseEstimator

from paella import Paella
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# TO-DO: remove Paella's study case related classes

class PipeGraph(_BaseComposition):
    """
    """

    def __init__(self, description):
        """
        Args:
            description: A dictionary expressing the contents of the graph. See tests examples.
        """
        self.description = description
        self.data = dict()
        self.graph = _create_graph_from_description(pipegraph=self, graph_description=description)

    def fit(self):
        """
        Iterates over the graph steps in topological order and executes their fit method
        """
        fit_steps = (self.graph.node[name]['step'] for name in nx.topological_sort(self.graph)
                     if 'fit' in self.graph.node[name]['step'].use_for)
        for step in fit_steps:
            logger.debug("Fitting: %s", step.node_name)
            input_data = read_connection()
            step.fit(**input_data)

    def predict(self):
        """
        Iterates over the graph steps in topological order and executes their run method
        """
        predict_steps = (self.graph.node[name]['step'] for name in nx.topological_sort(self.graph)
                         if 'run' in self.graph.node[name]['step'].use_for)
        for step in predict_steps:
            logger.debug("Predicting: %s", step.node_name)
            step.run()

    @property
    def steps(self):
        return [(name, self.graph.node[name]['step'])
                for name in nx.topological_sort(self.graph)]


class Runner(BaseEstimator):
    """
    Adapter to provide a common interface to fit(X,y), fit(X), etc.
    """

    def __init__(self, adaptee):
        self.adaptee = adaptee

    def get_params(self, deep=True):
        self.adaptee.get_params(deep=deep)

    def set_params(self, **params):
        self.adaptee.set_params(**params)

    @abstractmethod
    def fit(self, data_dictionary, **fit_params):
        """ Adapt fit methods"""

    @abstractmethod
    def run(self, data_dictionary, **run_params):
        """ Adapt run, transform, fit_predict"""


class TransformerArity1Runner(Runner):
    def fit(self, data_dictionary, **fit_params):
        X=data_dictionary['X']
        self.adaptee.fit(X, **fit_params)
        return self

    def run(self, data_dictionary, **run_params):
        X = data_dictionary['X']
        return self.adaptee.transform(X)

    def fit_run(self, data_dictionary, **fit_params):
        X = data_dictionary['X']
        return self.adaptee.fit(X, **fit_params).transform(X)



class TransformerArity2Runner(Runner):
    def fit(self, data_dictionary, **fit_params):
        X=data_dictionary['X']
        y = data_dictionary['y']
        self.adaptee.fit(X, y, **fit_params)
        return self

    def run(self, data_dictionary, **run_params):
        X = data_dictionary['X']
        return self.adaptee.transform(X)


    def fit_run(self, data_dictionary, **fit_params):
        X = data_dictionary['X']
        y = data_dictionary['y']
        return self.adaptee.fit(X, y, **fit_params).transform(X)


class FitXPredictXRunner(Runner):
    def fit(self, data_dictionary, **fit_params):
        X=data_dictionary['X']
        self.adaptee.fit(X, **fit_params)
        return self

    def run(self, data_dictionary, **run_params):
        X = data_dictionary['X']
        return self.adaptee.fit_predict(X, **fit_params)


class FitXyPredictX(Runner):
    def fit(self, data_dictionary, **fit_params):
        X=data_dictionary['X']
        y = data_dictionary['y']
        self.adaptee.fit(X, y, **fit_params)
        return self

    def run(self, data_dictionary, **run_params):
        X = data_dictionary['X']
        return self.adaptee.transform(X)

    def fit_run(self, data_dictionary, **fit_params):
        X = data_dictionary['X']
        y = data_dictionary['y']
        return self.adaptee.fit(X, y, **fit_params).transform(X)



################################
#  STEPS
################################
class Step(BaseEstimator):
    """
    Decorator providing extra functionality like pre fit/run and post fit/run processing information
    """

    def __init__(self, pipegraph, node_name, connections, use_for=['fit', 'run'], runner, **kargs):
        """

        Args:
            pipegraph:
            node_name:
            connections:
            use_for:
            sklearn_class:
            kargs:
        """
        self.pipegraph = pipegraph
        self.data = self.pipegraph.data
        self.node_name = node_name
        self.connections = connections
        self.use_for = use_for
        self.output = dict()
        self.kargs = kargs
        self.runner = runner

    def get_params(self, deep=True):
        return self.runner.get_params(deep=deep)

    def set_params(self, **params):
        self.runner.set_params(**params)

    def fit(self):
        """
        Template method for fitting
        """
        self._prepare_fit()
        self._fit()
        self._conclude_fit()

    def run(self):
        """
        Template method for predicting
        """
        self._prepare_run()
        self._run()
        self._conclude_run()

    def _read_connections(self):
        if isinstance(self, InputStep):
            input_data = self.connections
        else:
            input_data = {inner_variable: self.data[node_and_outer_variable_tuple]
                          for inner_variable, node_and_outer_variable_tuple in self.connections.items()}
        return input_data

    def _prepare_fit(self):
        self.input = self._read_connections()

    def _fit(self):
        self.runner.fit(self.input, self.kargs)

    def _conclude_fit(self):
        self._update_graph_data(self.output)

    def _prepare_run(self):
        self.input = self._read_connections()

    def _run(self):
        self.runner.run(self.input, self.kargs)

    def _conclude_run(self):
        self._update_graph_data(self.output)

    def _update_graph_data(self, data_dict):
        self.pipegraph.data.update(
            {(self.node_name, variable): value for variable, value in data_dict.items()}
        )

""""
TODO: nice decorator for keystrikes saving

class StepFromFunction(Step):
    def __init__(self, custom_function, pipegraph, node_name, connections, use_for=['fit', 'run'],
                 sklearn_class=None, **kargs):
        super().__init__(pipegraph, node_name, connections, use_for, sklearn_class, **kargs)
        self.custom_function = custom_function

    def run(self):
        self.output['output'] = self.custom_function(self.input)
"""

class PassStep(Step):
    """
    This class implements a step that just let's the information flow unaltered
    """

    def _run(self):
        self.output = self.input


class InputStep(PassStep):
    pass


class OutputStep(PassStep):
    pass


class StrategyArity1(Step):
    def _fit(self):
        self.sklearn_object.fit(**self.input)
        self.run()

    def _run(self):
        predict_dict = dict(self.input)

        if 'y' in predict_dict:
            del predict_dict['y']

        if 'sample_weight' in predict_dict:
            del predict_dict['sample_weight']

        self.output['prediction'] = self.sklearn_object.run(**predict_dict)


class SkLearnFitPredictAlternativeStep(Step):
    def _run(self):
        self.output['prediction'] = self.sklearn_object.fit_predict(**self.input)


class CustomConcatenationStep(Step):
    def _run(self):
        self.output['Xy'] = pd.concat(self.input, axis=1)


class CustomCombinationStep(Step):
    def _run(self):
        self.output['classification'] = np.where(self.input['dominant'] < 0, self.input['dominant'],
                                                 self.input['other'])


class CustomPaellaStep(Step):
    def __init__(self, pipegraph, node_name, connections, use_for=['fit', 'run'], sklearn_class=None, **kargs):
        super().__init__(pipegraph, node_name, connections, use_for=['fit', 'run'], sklearn_class=None, **kargs)
        self.sklearn_object = Paella(**kargs)

    def _fit(self):
        self.sklearn_object.fit(**self.input)
        self.run()

    def _run(self):
        self.output['prediction'] = self.sklearn_object.transform(self.input['X'], self.input['y'])


########################################

sklearn_models_dict = {
    'GaussianMixture': SkLearnFitPredictRegularStep,
    'DBSCAN': SkLearnFitPredictAlternativeStep,
    'LinearRegression': SkLearnFitPredictRegularStep,
}


def _create_graph_from_description(pipegraph, graph_description):
    graph = nx.DiGraph()
    graph.add_nodes_from(graph_description)
    for step_name, step_description in graph_description.items():
        current_node = graph.node[step_name]

        is_step_a_sklearn_class = step_description['step'].__name__ in sklearn_models_dict
        sklearn_class = step_description['step'] if is_step_a_sklearn_class else None
        step_class = sklearn_models_dict[sklearn_class.__name__] if is_step_a_sklearn_class else step_description[
            'step']
        kargs = step_description.get('kargs', {})
        current_node['step'] = step_class(pipegraph=pipegraph,
                                          node_name=step_name,
                                          connections=step_description['connections'],
                                          use_for=step_description['use_for'],
                                          sklearn_class=sklearn_class,
                                          **kargs)

        if step_name != 'First':
            for ascendant in set(node_and_outer_variable_tuple[0]
                                 for inner_variable, node_and_outer_variable_tuple in
                                 step_description['connections'].items()):
                graph.add_edge(ascendant, step_name)

    return graph
