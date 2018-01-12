# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx

from abc import ABC, abstractmethod
from collections import Iterator
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.base import BaseEstimator

from paella import Paella


# TO-DO: remove Paella's study case related classes

class PipeGraph(_BaseComposition):
    """
    """

    def __init__(self, graph_description):
        """
        Args:
            graph_description: A dictionary expressing the contents of the graph. See tests examples.
        """
        self.description = graph_description
        self.data = dict()
        self.graph = _create_graph_from_description(pipegraph=self, graph_description=graph_description)

    def __iter__(self):
        """
        Part of the iterator protocol

        Returns: An iterator over the graph nodes. Topological order is not ensured.
        """
        return self.graph.__iter__()

    def fit(self):
        """
        Iterates over the graph steps in topological order and executes their fit method
        """
        for step in self.fit_steps():
            print("Fitting: ", step.node_name)
            step.fit()

    def predict(self):
        """
        Iterates over the graph steps in topological order and executes their predict method
        """
        for step in self.predict_steps():
            print("Predicting: ", step.node_name)
            step.predict()

    @property
    def nodes(self):
        """
        Returns the graph nodes
        """
        return self.graph.nodes

    @property
    def steps(self):
        return [(name, self.graph.node[name]['step'])
                for name in nx.topological_sort(self.pipeGraph.graph)]

    def fit_steps(self):
        """
        Returns: An iterator over those steps with active fit status
        """
        return FitIterator(self)

    def predict_steps(self):
        """
        Returns: An iterator over those steps with active predict status
        """
        return PredictIterator(self)


#####################################
#   Iterators
#####################################
class StepIterator(Iterator):
    """
    An iterator over those steps in PipeGraph object with fit in use_for attribute
    """

    def __init__(self, pipegraph):
        """
        Args:
            pipeGraph: Reference to the PipeGraph object over witch iteration is performed
        """
        self.pipeGraph = pipegraph


class FitIterator(StepIterator):
    """
    An iterator over those steps in PipeGraph object with fit in use_for attribute
    """
    def __init__(self, pipegraph):
        """
        Args:
            pipeGraph: Reference to the PipeGraph object over witch iteration is performed
        """
        super().__init__(pipegraph)
        self.ordered_nodes_generator = (name for name in nx.topological_sort(self.pipeGraph.graph)
                                        if 'fit' in self.pipeGraph.graph.node[name]['step'].use_for)

    def __next__(self):
        """
        Part of the Iterator protocol
        Returns: An iterator over the nodes with fit in use_for attribute
        """
        next_node = next(self.ordered_nodes_generator)
        return self.pipeGraph.graph.node[next_node]['step']


class PredictIterator(StepIterator):
    """
    An iterator over those steps in PipeGraph object with predict in use_for attribute
    """
    def __init__(self, pipegraph):
        """
        Args:
            pipeGraph: Reference to the PipeGraph object over witch iteration is performed
        """
        super().__init__(pipegraph)
        self.ordered_nodes_generator = (name for name in nx.topological_sort(self.pipeGraph.graph)
                                        if 'predict' in self.pipeGraph.graph.node[name]['step'].use_for)

    def __next__(self):
        """
        Part of the Iterator protocol
        Returns: An iterator over the nodes with predict in use_for attribute
        """
        next_node = next(self.ordered_nodes_generator)
        return self.pipeGraph.graph.node[next_node]['step']


################################
#  STEPS
################################

class Step(BaseEstimator):
    """
    """

    def __init__(self, pipegraph, node_name, connections, use_for=['fit', 'predict'], sklearn_class=None, **kargs):
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
        self.sklearn_object = sklearn_class(**kargs) if sklearn_class is not None else None
        # If sklearn_class is None and the user tries to use it a tremendous error is expected to be raised,
        # which is a nice free error signal for the user

    def fit(self):
        """
        Template method for fitting
        """
        self._prepare_fit()
        self._fit()
        self._post_fit()

    def predict(self):
        """
        Template method for predicting
        """
        self._prepare_predict()
        self._predict()
        self._post_predict()

    def get_params(self, deep=True):
        if self.sklearn_object is not None:
            return self.sklearn_object.get_params(deep=deep)
        else:
            return self.kargs

    def set_params(self, **params):
        #TO-DO: nested params in case step is a pipegraph itself
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        if self.sklearn_object is not None:
            self.sklearn_object.set_params(**params)
        elif all(item in self.kargs for item in params):
            self.kargs.update(params)
        else:
            raise NameError('Step.set_params Error: key not in self.kargs')
        return self


    def _read_connections(self):
        if isinstance(self, FirstStep):
            input_data = self.connections
        else:
            input_data = {inner_variable: self.data[node_and_outer_variable_tuple]
                          for inner_variable, node_and_outer_variable_tuple in self.connections.items()}
        return input_data

    def _prepare_fit(self):
        self.input = self._read_connections()

    def _fit(self):
        self.predict()

    def _post_fit(self):
        self._update_graph_data(self.output)

    def _prepare_predict(self):
        self.input = self._read_connections()

    def _predict(self):
        pass

    def _post_predict(self):
        self._update_graph_data(self.output)

    def _update_graph_data(self, data_dict):
        self.pipegraph.data.update(
            {(self.node_name, variable): value for variable, value in data_dict.items()}
        )


class StepFromFunction(Step):
    def __init__(self, custom_function, pipegraph, node_name, connections, use_for=['fit', 'predict'],
                 sklearn_class=None, **kargs):
        super().__init__(pipegraph, node_name, connections, use_for, sklearn_class, **kargs)
        self.custom_function = custom_function

    def predict(self):
        self.output['output'] = self.custom_function(self.input)


class FirstStep(Step):
    """
    This class implements the First step doing nothing but publishing its input data in the graph data attribute
    """

    def _predict(self):
        self.output = self.input


class LastStep(Step):
    """
    This class implements the Last step doing nothing but publishing its input data in the graph data attribute
    """

    def _predict(self):
        self.output = self.input


class SkLearnFitPredictRegularStep(Step):
    def _fit(self):
        self.sklearn_object.fit(**self.input)
        self.predict()

    def _predict(self):
        predict_dict = dict(self.input)

        if 'y' in predict_dict:
            del predict_dict['y']

        if 'sample_weight' in predict_dict:
            del predict_dict['sample_weight']

        self.output['prediction'] = self.sklearn_object.predict(**predict_dict)


class SkLearnFitPredictAlternativeStep(Step):
    def _predict(self):
        self.output['prediction'] = self.sklearn_object.fit_predict(**self.input)


class CustomConcatenationStep(Step):
    def _predict(self):
        self.output['Xy'] = pd.concat(self.input, axis=1)


class CustomCombinationStep(Step):
    def _predict(self):
        self.output['classification'] = np.where(self.input['dominant'] < 0, self.input['dominant'],
                                                 self.input['other'])


class CustomPaellaStep(Step):
    def _fit(self):
        self.sklearn_object = Paella(**self.kargs)
        self.sklearn_object.fit(**self.input)
        self.predict()

    def _predict(self):
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
