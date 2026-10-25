.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PipeGraph!
============================================

.. note::
   **Project status (October 2026): maintenance mode.** Release 0.3.4 keeps PipeGraph
   working with Python 3.12–3.14 and scikit-learn 1.9, but no new features are planned.
   Most of PipeGraph's use cases are now covered by scikit-learn itself (stacking,
   metadata routing, ``ColumnTransformer``), by `skrub Data Ops
   <https://skrub-data.org/stable/data_ops.html>`_ and by `scikit-lego
   <https://koaning.github.io/scikit-lego/>`_. The project README on GitHub explains
   which tool to use for each case and lists the known limitations of PipeGraph:
   https://github.com/mcasl/PipeGraph#should-you-use-pipegraph-in-2026

Introduction
------------

Scikit-Learn's Pipeline is a very useful tool to bind a sequence of transformers and a final estimator
in a single unit capable of working itself as an estimator. This eases managing the fit and predict
of the complete sequence of steps through a simple call to their fit and predict methods,
and in combination with the GridSearchCV function allows to tune the hyperparameters of these steps
while in the search for the best model.

PipeGraph extends the concept of Pipeline by using a graph structure that can handle Scikit-Learn's
objects in imaginative layouts. This allows the user to:

- Express simple and complex sequences of steps, from linear workflows to elaborated graphs
- To have access to all the data generated along the intermediate steps
- To use GridSearchCV on those graphs

Let's see a few examples to understand how we can express these situations through PipeGraph objects.


    .. toctree::
       :maxdepth: 2

       api
       User_Guide
       auto_examples/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

