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
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def nodes(self):
        pass

    @abstractmethod
    def get_fit_steps(self):
        pass

    @abstractmethod
    def get_run_steps(self):
        pass


class PipeGraph(ComputationalGraph):
    def __init__(self, graph_description):
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
        return self.graph.__iter__()

    def fit(self):
        for step in self.get_fit_steps():
            print("Fitting: ", step.node_name)
            step.fit()

    def run(self):
        for step in self.get_run_steps():
            print("Running: ", step.node_name)
            step.run()

    @property
    def nodes(self):
        return self.graph.nodes

    def get_fit_steps(self):
        return FitIterator(self)

    def get_run_steps(self):
        return RunIterator(self)


#####################################
#   Iterators
#####################################
class StepIterator(ABC):
    @abstractmethod
    def __init__(self, pipeGraph):
        pass

    @abstractmethod
    def __next__(self):
        pass

    def __iter__(self):
        return self


class FitIterator(StepIterator):
    def __init__(self, pipe_graph):
        self.pipeGraph = pipe_graph
        self.ordered_nodes_generator = itertools.filterfalse(
                                  lambda name: not 'fit' in self.pipeGraph.graph.node[name]['step'].use_for,
                                  nx.topological_sort(self.pipeGraph.graph))
    def __next__(self):
        next_node = next(self.ordered_nodes_generator)
        return self.pipeGraph.graph.node[next_node]['step']


class RunIterator(StepIterator):
    def __init__(self, pipe_graph):
        self.pipeGraph = pipe_graph
        self.ordered_nodes_generator = itertools.filterfalse(
                                  lambda name: not 'run' in self.pipeGraph.graph.node[name]['step'].use_for,
                                  nx.topological_sort(self.pipeGraph.graph))
    def __next__(self):
        next_node = next(self.ordered_nodes_generator)
        return self.pipeGraph.graph.node[next_node]['step']


################################
#  STEPS
################################
class Step(ABC):
    def __init__(self, pipegraph, node_name, input, use_for=['fit', 'run'], sklearn_class=None, **kargs):
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
        pass

    # @abstractmethod
    def run(self):
        pass

    def update_graph_data(self, data_dict):
        self.pipegraph.data.update(
            {(self.node_name, variable): value for variable, value in data_dict.items()}
        )

class FirstStep(Step):
    def fit(self):
        self.run()

    def run(self):
        self.update_graph_data(self.input)


class ConcatenateInputStep(Step):
    def fit(self):
        input_values = [self.data[node_variable] for node_variable in self.input]
        input_data = pd.concat(input_values, axis=1)
        self.sklearn_object.fit(input_data)
        self.update_graph_data({'output': self.sklearn_object.predict(input_data)})

    def run(self):
        input_values = [self.data[node_variable] for node_variable in self.input]
        input_data = pd.concat(input_values, axis=1)
        self.update_graph_data({'output': self.sklearn_object.predict(input_data)})


class CustomDBSCANStep(Step):
    def fit(self):
        self.run()

    def run(self):
        input_values = [self.data[node_variable] for node_variable in self.input]
        input_data = pd.concat(input_values, axis=1)
        self.update_graph_data({'output': self.sklearn_object.fit_predict(input_data)})


class CustomCombinationStep(Step):
    def fit(self):
        self.run()

    def run(self):
        v0 = self.data['Dbscan', 'output']
        v1 = self.data['Gaussian_Mixture', 'output']
        classification = np.where(v0 < 0, v0, v1)
        self.update_graph_data({'output': classification})


class PaellaStep(Step):
    def fit(self):
        X = self.data['First', 'X']
        y = self.data['First', 'y']
        classification = self.data['Combine_Clustering', 'output']

        self.sklearn_object.fit(X=X, y=y, classification=classification)
        sample_weight = self.sklearn_object.transform(X, y)
        self.update_graph_data({'sample_weight': sample_weight})

    def run(self):
        data = self.input['Concatenate_Xy__data']
        sample_weight = self.sklearn_object.transform(data)
        self.update_graph_data({'sample_weight': sample_weight})


class RegressorStep(Step):
    def fit(self):
        X = self.data['First', 'X']
        y = self.data['First', 'y']
        sample_weight = self.data['Paella', 'sample_weight']
        self.sklearn_object.fit(X=X, y=y, sample_weight=sample_weight)
        y_pred = self.sklearn_object.predict(X)
        self.update_graph_data({'prediction': y_pred})

    def run(self):
        X = self.data['First', 'X']
        y_pred = self.sklearn_object.predict(X)
        self.update_graph_data({'prediction': y_pred})


class LastStep(Step):
    def fit(self):
        self.run()

    def run(self):
        self.update_graph_data({node_variable[1]: self.data[node_variable] for node_variable in self.input})
