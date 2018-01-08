# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import networkx as nx
from paella import Paella

from abc import ABC, abstractmethod


##############################
# GRAPH
##############################
class ComputationalGraph(ABC):
    """
    Protocol definition. Use PipeGraph instead.
    """

    @abstractmethod
    def __iter__(self):
        """
        Part of the iterator protocol
        """
        pass

    @abstractmethod
    def fit(self):
        """
        Iterates over the graph steps in topological order and executes their fit method
        """
        pass

    @abstractmethod
    def run(self):
        """
        Iterates over the graph steps in topological order and executes their run method
        """
        pass

    @abstractmethod
    def nodes(self):
        """
        Returns the graph nodes
        """
        pass

    @abstractmethod
    def get_fit_steps(self):
        """
        Returns an iterator over those steps with active fit status
        """
        pass

    @abstractmethod
    def get_run_steps(self):
        """
        Returns an iterator over those steps with active run status
        """
        pass


class PipeGraph(ComputationalGraph):
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
        for step in self.get_fit_steps():
            print("Fitting: ", step.node_name)
            step.fit()

    def run(self):
        """
        Iterates over the graph steps in topological order and executes their run method
        """
        for step in self.get_run_steps():
            print("Running: ", step.node_name)
            step.run()

    @property
    def nodes(self):
        """
        Returns the graph nodes
        """
        return self.graph.nodes

    def get_fit_steps(self):
        """
        Returns: An iterator over those steps with active fit status
        """
        return FitIterator(self)

    def get_run_steps(self):
        """
        Returns: An iterator over those steps with active run status
        """
        return RunIterator(self)


#####################################
#   Iterators
#####################################
class StepIterator(ABC):
    """
    Protocol definition. se concrete classes instead, e.g. FitIterator and RunIterator
    """

    @abstractmethod
    def __init__(self, pipeGraph):
        """
        Args:
            pipeGraph: Reference to the PipeGraph object over witch iteration is performed
        """
        pass

    @abstractmethod
    def __next__(self):
        """
        Part of the Iterator protocol
        """
        pass

    def __iter__(self):
        """
        Part of the Iterator protocol.
        Returns: An iterator over the PipeGraph object stored in self.pipeGraph
        """
        return self


class FitIterator(StepIterator):
    """
    An iterator over those steps in PipeGraph object with fit in use_for attribute
    """

    def __init__(self, pipe_graph):
        """
        Args:
            pipeGraph: Reference to the PipeGraph object over witch iteration is performed
        """
        self.pipeGraph = pipe_graph
        self.ordered_nodes_generator = (name for name in nx.topological_sort(self.pipeGraph.graph)
                                        if 'fit' in self.pipeGraph.graph.node[name]['step'].use_for)

    def __next__(self):
        """
        Part of the Iterator protocol
        Returns: An iterator over the nodes with fit in use_for attribute
        """
        next_node = next(self.ordered_nodes_generator)
        return self.pipeGraph.graph.node[next_node]['step']


class RunIterator(StepIterator):
    """
    An iterator over those steps in PipeGraph object with run in use_for attribute
    """

    def __init__(self, pipe_graph):
        """
        Args:
            pipeGraph: Reference to the PipeGraph object over witch iteration is performed
        """
        self.pipeGraph = pipe_graph
        self.ordered_nodes_generator = (name for name in nx.topological_sort(self.pipeGraph.graph)
                                        if 'run' in self.pipeGraph.graph.node[name]['step'].use_for)

    def __next__(self):
        """
        Part of the Iterator protocol
        Returns: An iterator over the nodes with run in use_for attribute
        """
        next_node = next(self.ordered_nodes_generator)
        return self.pipeGraph.graph.node[next_node]['step']


################################
#  STEPS
################################
class Step(ABC):

    def fit(self):
        """
        Template method for fitting
        """
        self.input = self._read_connections()
        self._pre_fit()
        self._fit()
        self._post_fit()
        self._update_graph_data(self.output)

    def run(self):
        """
        Template method for running
        """
        self.input = self._read_connections()
        self._pre_run()
        self._run()
        self._post_run()
        self._update_graph_data(self.output)

    @abstractmethod
    def _pre_fit(self):
        ''' Preprocessing '''
        pass

    @abstractmethod
    def _fit(self):
        ''' Fit operations '''
        pass

    @abstractmethod
    def _post_fit(self):
        ''' Postprocessing '''
        pass

    @abstractmethod
    def _pre_run(self):
        ''' Preprocessing '''
        pass

    @abstractmethod
    def _run(self):
        ''' Run operations '''
        pass

    @abstractmethod
    def _post_run(self):
        ''' Postprocessing '''
        pass

    @abstractmethod
    def _read_connections(self):
        pass

    def _update_graph_data(self, data_dict):
        self.pipegraph.data.update(
            {(self.node_name, variable): value for variable, value in data_dict.items()}
        )


class CustomStep(Step):
    """
    """

    def __init__(self, pipegraph, node_name, connections, use_for=['fit', 'run'], sklearn_class=None, **kargs):
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

    def _read_connections(self):
        if isinstance(self, FirstStep):
            input_data = self.connections
        else:
            input_data = {inner_variable: self.data[node_and_outer_variable_tuple]
                          for inner_variable, node_and_outer_variable_tuple in self.connections.items()}
        return input_data

    def _pre(self):
        pass

    def _post(self):
        pass

    def _pre_fit(self):
        self._pre()

    def _fit(self):
        pass

    def _post_fit(self):
        self._post()

    def _pre_run(self):
        self._pre()

    def _run(self):
        pass

    def _post_run(self):
        self._post()


class FirstStep(CustomStep):
    """
    This class implements the First step doing nothing but publishing its input data in the graph data attribute
    """

    def _post(self):
        self.output = self.input


class LastStep(CustomStep):
    """
    This class implements the Last step doing nothing but publishing its input data in the graph data attribute
    """

    def _post(self):
        self.output = self.input


class SkLearnFitPredictRegularStep(CustomStep):
    def _fit(self):
        self.sklearn_object.fit(**self.input)

    def _post(self):
        predict_dict = dict(self.input)

        if 'y' in predict_dict:
            del predict_dict['y']

        if 'sample_weight' in predict_dict:
            del predict_dict['sample_weight']

        self.output['prediction'] = self.sklearn_object.predict(**predict_dict)


class SkLearnFitPredictAlternativeStep(CustomStep):
    def _post(self):
        self.output['prediction'] = self.sklearn_object.fit_predict(**self.input)


class CustomConcatenationStep(CustomStep):
    def _post_fit(self):
        self.output['Xy'] = pd.concat(self.input, axis=1)


class CustomCombinationStep(CustomStep):
    def _post_fit(self):
        self.output['classification'] = np.where(self.input['dominant'] < 0, self.input['dominant'],
                                                 self.input['other'])


class CustomPaellaStep(CustomStep):

    def _pre_fit(self):
        self.sklearn_object = Paella(**self.kargs)

    def _fit(self):
        self.sklearn_object.fit(**self.input)

    def _post_fit(self):
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
