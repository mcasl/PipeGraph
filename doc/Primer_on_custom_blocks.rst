A primer on building PipeGraph's custom blocks
==============================================

Wrappers for Scikit-Learn standard objects
------------------------------------------

Consider the following Scikit-Learn common objects:

.. code:: ipython3

    import sklearn
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import DBSCAN
    
    classifier = GaussianNB()
    scaler = MinMaxScaler() 
    dbscanner = DBSCAN()

And let's load some data to run the examples:

.. code:: ipython3

    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target

Now, let's fit each of the above defined sklearn objects and get the
output produced afterwards by using the corresponding method (predict,
fit\_predict, transform):

.. code:: ipython3

    classifier.fit(X, y)
    scaler.fit(X);
    dbscanner.fit(X, y)




.. parsed-literal::

    DBSCAN(algorithm='auto', eps=0.5, leaf_size=30, metric='euclidean',
        metric_params=None, min_samples=5, n_jobs=1, p=None)



.. code:: ipython3

    classifier.predict(X)




.. parsed-literal::

    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           --- omited ---


.. code:: ipython3

    classifier.predict_proba(X)




.. parsed-literal::

    array([[  1.00000000e+000,   1.38496103e-018,   7.25489025e-026],
           [  1.00000000e+000,   1.48206242e-017,   2.29743996e-025],
           [  1.00000000e+000,   1.07780639e-018,   2.35065917e-026],
           [  1.00000000e+000,   1.43871443e-017,   2.89954283e-025],
    --- omited ---



.. code:: ipython3

    classifier.predict_log_proba(X)




.. parsed-literal::

    array([[  0.00000000e+00,  -4.11208597e+01,  -5.78855367e+01],
           [  0.00000000e+00,  -3.87505119e+01,  -5.67328319e+01],
           [  0.00000000e+00,  -4.13716038e+01,  -5.90125166e+01],
           [  0.00000000e+00,  -3.87801966e+01,  -5.65000742e+01],
         --- omited ---


.. code:: ipython3

    scaler.transform(X)




.. parsed-literal::

    array([[ 0.22222222,  0.625     ,  0.06779661,  0.04166667],
           [ 0.16666667,  0.41666667,  0.06779661,  0.04166667],
           [ 0.11111111,  0.5       ,  0.05084746,  0.04166667],
           [ 0.08333333,  0.45833333,  0.08474576,  0.04166667],
         --- omited ---


.. code:: ipython3

    dbscanner.fit_predict(X)




.. parsed-literal::

    array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,
            1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1,
           --- omited ---


As it can be seen, in order to have access for each object's output, one
needs to call different methods. So as to offer a homogeneous interface
a collection of adapters is available in PipeGraph. Them all derive from
the ``AdapterForSkLearnLikeAdaptee`` baseclass. This class is an adapter
for Scikit-Learn objects in order to provide a common interface based on
fit and predict methods irrespectively of whether the adapted object
provided a ``transform``, ``fit_predict``, or ``predict interface``.

As it can be seen from the following code fragment, the ``fit`` and
``predict`` allow for an arbitrary number of positional and keyword
based parameters. These will have to be coherent with the adaptees
expectations, but at least we are not imposing hard constrains to the
adapter's interface.

::

    class AdapterForSkLearnLikeAdaptee(BaseEstimator):
        def fit(self, *pargs, **kwargs):
           ...
        def predict(self, *pargs, **kwargs):
           ...

Those sklearn objects following the ``predict`` protocol can be wrapped
into the class ``AdapterForFitPredictAdaptee``:

.. code:: ipython3

    from pipegraph.adapters import AdapterForFitPredictAdaptee
    
    wrapped_classifier = AdapterForFitPredictAdaptee(classifier)
    y_pred = wrapped_classifier.predict(X=X)
    y_pred




.. parsed-literal::

    {'predict': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            --- omited ---

     'predict_log_proba': array([[  0.00000000e+00,  -4.11208597e+01,  -5.78855367e+01],
            [  0.00000000e+00,  -3.87505119e+01,  -5.67328319e+01],
            [  0.00000000e+00,  -4.13716038e+01,  -5.90125166e+01],
            [  0.00000000e+00,  -3.87801966e+01,  -5.65000742e+01],
            [  0.00000000e+00,  -4.22118362e+01,  -5.87821546e+01],
            --- omited ---

     'predict_proba': array([[  1.00000000e+000,   1.38496103e-018,   7.25489025e-026],
            [  1.00000000e+000,   1.48206242e-017,   2.29743996e-025],
            [  1.00000000e+000,   1.07780639e-018,   2.35065917e-026],
            [  1.00000000e+000,   1.43871443e-017,   2.89954283e-025],
            [  1.00000000e+000,   4.65192224e-019,   2.95961100e-026],
             --- omited ---


As you can see the wrapper provides its output as a dictionary
containing the outputs provided by ``predict``, ``predict_proba``, and
``predict_log_proba`` where these methods are available.

.. code:: ipython3

    list(y_pred.keys())




.. parsed-literal::

    ['predict', 'predict_proba', 'predict_log_proba']



Those sklearn objects following the ``transform`` protocol can be
wrapped into the class ``AdapterForFitTransformAdaptee``:

.. code:: ipython3

    from pipegraph.adapters import AdapterForFitTransformAdaptee
    
    wrapped_scaler = AdapterForFitTransformAdaptee(scaler)
    y_pred=wrapped_scaler.predict(X)
    y_pred




.. parsed-literal::

    {'predict': array([[ 0.22222222,  0.625     ,  0.06779661,  0.04166667],
            [ 0.16666667,  0.41666667,  0.06779661,  0.04166667],
            [ 0.11111111,  0.5       ,  0.05084746,  0.04166667],
            [ 0.08333333,  0.45833333,  0.08474576,  0.04166667],
            [ 0.19444444,  0.66666667,  0.06779661,  0.04166667],
        --- omited ---


The adapter for transformers doesn't have to provide so many methods'
output, only the value provided by calling ``trasform`` method on the
adaptee, which for homogeneity is provided as a dictionary with
'predict' as key:

.. code:: ipython3

    list(y_pred.keys())




.. parsed-literal::

    ['predict']



Those sklearn objects following the ``fit_predict`` protocol can be
wrapped into the class ``AdapterForAtomicFitPredictAdaptee``:

.. code:: ipython3

    from pipegraph.adapters import AdapterForAtomicFitPredictAdaptee
    
    wrapped_dbscanner = AdapterForAtomicFitPredictAdaptee(dbscanner)
    y_pred = wrapped_dbscanner.predict(X=X)
    y_pred




.. parsed-literal::

    {'predict': array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,
             1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1,
            --- omited ---

Again, this adapter provides a dictionary with the values of calling
``fit_predict`` under the key 'predict'.

Special wrappers
----------------

Besides of the three families of objects provided by Scikit-Learn, it is
sometimes convenient to provide custom objects whose ``predict`` method
returns multiple outputs. In this case, a dictionary can be used as
well, with the name of the outputs as keys. In order to comply with this
kind of output, the class
``AdapterForCustomFitPredictWithDictionaryOutputAdaptee`` is provided:

.. code:: ipython3

    from pipegraph.standard_blocks import Demultiplexer
    from pipegraph.adapters import AdapterForCustomFitPredictWithDictionaryOutputAdaptee
    
    demultiplexer = Demultiplexer()
    wrapped_demultiplexer = AdapterForCustomFitPredictWithDictionaryOutputAdaptee(demultiplexer)
    output = wrapped_demultiplexer.predict(X=X, selection=y)
    output




.. parsed-literal::

    {'X_0':
           0    1    2    3
     0   5.1  3.5  1.4  0.2
     1   4.9  3.0  1.4  0.2
     2   4.7  3.2  1.3  0.2
     3   4.6  3.1  1.5  0.2
     4   5.0  3.6  1.4  0.2
     --- omited ---

    'X_1':
           0    1    2    3
     50  7.0  3.2  4.7  1.4
     51  6.4  3.2  4.5  1.5
     52  6.9  3.1  4.9  1.5
     53  5.5  2.3  4.0  1.3
     54  6.5  2.8  4.6  1.5
     --- omited ---

     'X_2':
             0    1    2    3
     100  6.3  3.3  6.0  2.5
     101  5.8  2.7  5.1  1.9
     102  7.1  3.0  5.9  2.1
     103  6.3  2.9  5.6  1.8
     104  6.5  3.0  5.8  2.2
     --- omited ---


.. code:: ipython3

    list(output.keys())




.. parsed-literal::

    ['X_0', 'X_1', 'X_2']



As it can be seen, this adapter's ``predict`` method provides the
dictionary of outputs provided by the adaptee with its original keys.

Wrapping your custom blocks
===========================

PipeGraph uses the
``wrap_adaptee_in_process(adaptee, strategy_class=None)`` function to
wrap the objects passed to its constructor's ``steps`` parameters
accordingly to these rules: - If the ``strategy_class`` parameter is
passed, this class is used as adapter - Else, if the adaptee's class is
in ``pipegraph.base.strategies_for_custom_adaptees`` dictionary, the
value class there is used. - Else, if the adaptee has a ``predict``
method, the ``AdapterForFitPredictAdaptee`` class is used. - Else, if
the adaptee has a ``transform`` method, the
``AdapterForFitTransformAdaptee`` class is used. - Else, if the adaptee
has a ``fit_predict`` method, the ``AdapterForAtomicFitPredictAdaptee``

.. code:: ipython3

    from pipegraph.base import wrap_adaptee_in_process
    
    wrapped_scaler = wrap_adaptee_in_process(scaler)
    wrapped_scaler.predict(X)




.. parsed-literal::

    {'predict': array([[ 0.22222222,  0.625     ,  0.06779661,  0.04166667],
            [ 0.16666667,  0.41666667,  0.06779661,  0.04166667],
            [ 0.11111111,  0.5       ,  0.05084746,  0.04166667],
            [ 0.08333333,  0.45833333,  0.08474576,  0.04166667],
            [ 0.19444444,  0.66666667,  0.06779661,  0.04166667],
            --- omited ---


.. code:: ipython3

    wrapped_demultiplexer = wrap_adaptee_in_process(demultiplexer)
    wrapped_demultiplexer.predict(X=X, selection=y)




.. parsed-literal::

    {'X_0':
           0    1    2    3
     0   5.1  3.5  1.4  0.2
     1   4.9  3.0  1.4  0.2
     2   4.7  3.2  1.3  0.2
     3   4.6  3.1  1.5  0.2
     4   5.0  3.6  1.4  0.2
     --- omited ---

     'X_1':
            0    1    2    3
     50  7.0  3.2  4.7  1.4
     51  6.4  3.2  4.5  1.5
     52  6.9  3.1  4.9  1.5
     53  5.5  2.3  4.0  1.3
     --- omited ---
     'X_2':
             0    1    2    3
     100  6.3  3.3  6.0  2.5
     101  5.8  2.7  5.1  1.9
     102  7.1  3.0  5.9  2.1
     103  6.3  2.9  5.6  1.8
     --- omited ---


Those users implementing their own custom blocks may find useful the
option of providing their own custom class to th
``wrap_adaptee_in_process``, as in:

.. code:: ipython3

    wrapped_demultiplexer = wrap_adaptee_in_process(adaptee=demultiplexer,
                                                    strategy_class=AdapterForCustomFitPredictWithDictionaryOutputAdaptee)
    wrapped_demultiplexer.predict(X=X, selection=y)




.. parsed-literal::

    {'X_0':
           0    1    2    3
     0   5.1  3.5  1.4  0.2
     1   4.9  3.0  1.4  0.2
     2   4.7  3.2  1.3  0.2
     3   4.6  3.1  1.5  0.2
     4   5.0  3.6  1.4  0.2
     --- omited ---

     'X_1':
            0    1    2    3
     50  7.0  3.2  4.7  1.4
     51  6.4  3.2  4.5  1.5
     52  6.9  3.1  4.9  1.5
     53  5.5  2.3  4.0  1.3
     54  6.5  2.8  4.6  1.5
     --- omited ---

     'X_2':
             0    1    2    3
     100  6.3  3.3  6.0  2.5
     101  5.8  2.7  5.1  1.9
     102  7.1  3.0  5.9  2.1
     103  6.3  2.9  5.6  1.8
     104  6.5  3.0  5.8  2.2
     --- omited ---


Passing an already wrapped object to PipeGraph's constructor ``steps``
parameter by using the ``wrap_adaptee_in_process`` as describe above may
be useful for those custom blocks built by users, thus avoiding the need
to modify the ``pipegraph.base.strategies_for_custom_adaptees``
dictionary.
