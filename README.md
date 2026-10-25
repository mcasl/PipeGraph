# PipeGraph — graph-shaped workflows for scikit-learn

[![PyPI version](https://badge.fury.io/py/pipeGraph.svg)](https://pypi.org/project/pipeGraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)

> **Project status (October 2026): maintenance mode.**
> Release 0.3.4 makes sure the package installs and its test suite passes on
> Python 3.12–3.14 with scikit-learn 1.9, numpy 2.x, pandas 3.x and networkx 3.x.
> No new features are planned. Most of what PipeGraph was created for in 2018 is
> now available in scikit-learn itself or in actively maintained libraries built
> by the scikit-learn team — see [*Should you use PipeGraph in 2026?*](#should-you-use-pipegraph-in-2026)
> before adopting it.

**PipeGraph** extends scikit-learn's `Pipeline` from a linear sequence of steps to a
directed acyclic graph. Steps are the usual `(name, estimator)` tuples; *connections*
say which output of which step feeds which input of which other step. Any output can be
routed to any input — `X`, `y`, `sample_weight`, or any custom keyword — the graph can
differ between `fit` and `predict`, every intermediate result stays accessible, and the
whole graph is an estimator that works with `GridSearchCV`.

```python
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from pipegraph import PipeGraph

# Choose the number of clusters by how well a supervised model can reproduce them.
model = PipeGraph(steps=[('clustering', KMeans()),
                         ('classification', LinearDiscriminantAnalysis())])
model.inject(sink='clustering',     sink_var='X', source='_External', source_var='X')
model.inject(sink='classification', sink_var='X', source='_External', source_var='X')
model.inject(sink='classification', sink_var='y', source='clustering', source_var='predict')

search = GridSearchCV(model, param_grid={'clustering__n_clusters': range(2, 20)}, cv=5)
search.fit(X)
```

The output of the clustering step becomes the *target* of the classification step, and
`n_clusters` is tuned by cross-validation. This is the kind of workflow a `Pipeline`
cannot express.

## Should you use PipeGraph in 2026?

When PipeGraph was written (2018), scikit-learn's `Pipeline` had no way to route
`sample_weight` to a specific step, no stacking, no dataframe output and no notion of a
graph. Since then the ecosystem has caught up with almost every use case in our gallery
of examples. Our recommendation today:

| If you need… | Use… |
|---|---|
| A linear sequence of transformers and a final estimator, tuned with `GridSearchCV` | [`sklearn.pipeline.Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) |
| The predictions of several models (plus, optionally, the original features) fed to a meta-model | [`StackingClassifier` / `StackingRegressor`](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization) (scikit-learn ≥ 0.22). They use out-of-fold predictions; PipeGraph's `Concatenator` feeds in-sample predictions to the meta-model, which leaks training information |
| `sample_weight`, `groups` or any other metadata delivered to specific steps, correctly split in cross-validation | scikit-learn [metadata routing](https://scikit-learn.org/stable/metadata_routing.html) (≥ 1.4, still opt-in via `sklearn.set_config(enable_metadata_routing=True)`), and `Pipeline(transform_input=[...])` (≥ 1.6) to push metadata such as a validation set through the preceding steps |
| Different transformers for different columns, or several transformers side by side | [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) / [`FeatureUnion`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html), with `set_output(transform="pandas")` or `"polars"` to keep column names |
| A full DAG with any number of inputs, tunable choices anywhere in the graph (including `sample_weight` computed from the data), inspection of intermediate results, different behaviour at fit and predict time, several tables | [skrub Data Ops](https://skrub-data.org/stable/data_ops.html), from the scikit-learn team. `skb.apply(..., fit_kwargs={"sample_weight": weights})` covers our *varying sample weights* example end to end |
| One model per group or segment of the data, with a global fallback | [`scikit-lego`'s `GroupedPredictor`](https://koaning.github.io/scikit-lego/api/meta/) — this replaces our `Demultiplexer` / `Multiplexer` examples |
| A fitted predictor used as a transformer in the middle of a pipeline | `scikit-lego`'s `EstimatorTransformer`, or stacking as above |

What none of the above offers, and PipeGraph still does, is using the **output of a step
as the target (`y`) of another step while tuning the producing step by cross-validation**,
as in the example above. skrub requires the target to be marked with `mark_as_y()` and
does not tune choices upstream of it; a `Pipeline` only passes `X` between steps. If that
is your situation, PipeGraph works, with the limitations listed below. A dedicated
meta-estimator of a few dozen lines (clusterer + classifier, `n_clusters` as a parameter)
is usually the more robust solution.

## Known limitations

These are the reasons we chose not to keep developing the current code base. Fixing them
properly would mean rewriting the core.

- **Fitted `PipeGraph` objects cannot be pickled** (so they cannot be saved with `joblib`,
  nor returned by `cross_validate(..., return_estimator=True)` with `n_jobs > 1`). At fit
  time each step's class is replaced by a dynamically created adapter class
  (`add_mixins_to_step`), which `pickle` cannot locate. For the same reason, **`fit`
  changes the class of the step objects you pass in**.
- **Not compliant with `sklearn.utils.estimator_checks.check_estimator`** (19 of 41 checks
  fail with scikit-learn 1.9.1): `fit_connections` is filled in during `fit`, no
  `NotFittedError` is raised before fitting, `n_features_in_` is not set,
  `decision_function` applies the last step to the raw `X`, and `ValueError`s raised by a
  step are printed instead of propagated.
- **No metadata-routing support** (`get_metadata_routing` is not implemented); extra fit
  arguments are routed by PipeGraph's own connection mechanism only.
- `predict_dict` only exposes the `predict`, `predict_proba` and `predict_log_proba`
  outputs of the *last* step; other intermediate results are reachable through the
  private `_fit_data` / `_predict_data` dictionaries.

## Installation

```
pip install pipegraph
```

Requires Python ≥ 3.12, scikit-learn ≥ 1.5, numpy ≥ 1.26, pandas ≥ 2.1 and networkx ≥ 3.0
(`pip install "pipegraph[examples]"` adds matplotlib, seaborn and scipy for the gallery).
The 0.3.4 test suite passes both with the newest versions at release time (Python 3.14,
scikit-learn 1.9.1, numpy 2.5, pandas 3.1, networkx 3.7) and with the oldest supported ones
(Python 3.12, scikit-learn 1.5.2, numpy 1.26.4, pandas 2.1.4, networkx 3.0).

For development the repository ships a [uv](https://docs.astral.sh/uv/) project:

```
uv sync            # creates .venv with the locked versions
uv run pytest      # runs the test suite
```

## Documentation

API reference, user guide and gallery of examples:
https://mcasl.github.io/PipeGraph/

The examples in [`examples/`](examples/) are still runnable and illustrate the
connection syntax (`inject`, `fit_connections`, `predict_connections`), the custom blocks
(`Concatenator`, `ColumnSelector`, `Demultiplexer`, `Multiplexer`) and the use of
`GridSearchCV` on a graph.

## Citing

The framework behind PipeGraph is described in:

> Castejón-Limas, M.; Fernández-Robles, L.; Alaiz-Moretón, H.; Cifuentes-Rodríguez, J.;
> Fernández-Llamas, C. *A Framework for the Optimization of Complex Cyber-Physical Systems
> via Directed Acyclic Graph.* Sensors 2022, 22(4), 1490.
> https://doi.org/10.3390/s22041490

## License

MIT. Copyright (c) 2018 Laura Fernández Robles, Héctor Alaiz Moretón, Jaime
Cifuentes-Rodríguez, Javier Alfonso-Cendón, Camino Fernández-Llamas and Manuel
Castejón-Limas. See [LICENSE.txt](LICENSE.txt).
