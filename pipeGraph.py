# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import networkx as nx
import itertools

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
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(graph_description)
        for current_name, info in graph_description.items():
            current_node = self.graph.node[current_name]
            if info['kargs'] is None:
                info['kargs'] = dict()
            current_node['step'] = info['step'](pipegraph=self,
                                                node_name=current_name,
                                                input=info['input'],
                                                use_for=info['use_for'],
                                                sklearn_class=info['sklearn_class'],
                                                **info['kargs'])

            if current_name != 'First':
                for ascendant in set(name for name, attribute in info['input']):
                    self.graph.add_edge(ascendant, current_name)

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
    """
    Protocol definition. Use concrete classes instead.
    """
    def __init__(self, pipegraph, node_name, input, use_for=['fit', 'run'], sklearn_class=None, **kargs):
        """

        Args:
            pipegraph:
            node_name:
            input:
            use_for:
            sklearn_class:
            kargs:
        """
        self.pipegraph = pipegraph
        self.data = self.pipegraph.data
        self.node_name = node_name
        self.input = input
        self.use_for = use_for
        if sklearn_class is not None:
            self.sklearn_object = sklearn_class(**kargs)

    # TODO: Add None assignment case with a pass similar

    # @abstractmethod
    def fit(self):
        """

        """
        pass

    # @abstractmethod
    def run(self):
        """

        """
        pass

    def update_graph_data(self, data_dict):
        """

        Args:
            data_dict:
        """
        self.pipegraph.data.update(
            {(self.node_name, variable): value for variable, value in data_dict.items()}
        )

class FirstStep(Step):
    """
    Graphs will have a First and a Last step.
    This class implements the former doing nothing but publishing its input data in the graph data attribute
    """
    def fit(self):
        """
        Fit as in scikit-learn fit
        """
        self.run()

    def run(self):
        """
        Run as in scikit-learn predict, transform; essentially doing any other thing different from fit.
        """
        self.update_graph_data(self.input)


class ConcatenateInputStep(Step):
    """
    """
    def fit(self):
        """

        """
        input_values = [self.data[node_variable] for node_variable in self.input]
        input_data = pd.concat(input_values, axis=1)
        self.sklearn_object.fit(input_data)
        self.update_graph_data({'output': self.sklearn_object.predict(input_data)})

    def run(self):
        """

        """
        input_values = [self.data[node_variable] for node_variable in self.input]
        input_data = pd.concat(input_values, axis=1)
        self.update_graph_data({'output': self.sklearn_object.predict(input_data)})


class CustomDBSCANStep(Step):
    """
    """
    def fit(self):
        """

        """
        self.run()

    def run(self):
        """

        """
        input_values = [self.data[node_variable] for node_variable in self.input]
        input_data = pd.concat(input_values, axis=1)
        self.update_graph_data({'output': self.sklearn_object.fit_predict(input_data)})


class CustomCombinationStep(Step):
    """
    """
    def fit(self):
        """

        """
        self.run()

    def run(self):
        """

        """
        v0 = self.data['Dbscan', 'output']
        v1 = self.data['Gaussian_Mixture', 'output']
        classification = np.where(v0 < 0, v0, v1)
        self.update_graph_data({'output': classification})


class PaellaStep(Step):
    """
    """
    def fit(self):
        """

        """
        X = self.data['First', 'X']
        y = self.data['First', 'y']
        classification = self.data['Combine_Clustering', 'output']

        self.sklearn_object.fit(X=X, y=y, classification=classification)
        sample_weight = self.sklearn_object.transform(X, y)
        self.update_graph_data({'sample_weight': sample_weight})

    def run(self):
        """

        """
        data = self.input['Concatenate_Xy__data']
        sample_weight = self.sklearn_object.transform(data)
        self.update_graph_data({'sample_weight': sample_weight})


class RegressorStep(Step):
    """
    """
    def fit(self):
        """

        """
        X = self.data['First', 'X']
        y = self.data['First', 'y']
        sample_weight = self.data['Paella', 'sample_weight']
        self.sklearn_object.fit(X=X, y=y, sample_weight=sample_weight)
        y_pred = self.sklearn_object.predict(X)
        self.update_graph_data({'prediction': y_pred})

    def run(self):
        """

        """
        X = self.data['First', 'X']
        y_pred = self.sklearn_object.predict(X)
        self.update_graph_data({'prediction': y_pred})


class LastStep(Step):
    """
    """
    def fit(self):
        """

        """
        self.run()

    def run(self):
        """

        """
        self.update_graph_data({node_variable[1]: self.data[node_variable] for node_variable in self.input})
