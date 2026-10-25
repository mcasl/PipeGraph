# Changelog

## 0.3.4 — 2026-10-25

Maintenance release. No functional changes to the graph engine.

- Compatibility with Python 3.12–3.14, scikit-learn 1.5–1.9, numpy 1.26–2.x,
  pandas 2.1–3.x and networkx 3.x (`graph.nodes` instead of the removed `graph.node`;
  `Concatenator` makes column names unique, as recent scikit-learn versions require).
  The test suite was run at both ends of these ranges.
- Project tooling moved to `uv` (`pyproject.toml`, `uv.lock`); `requirements.txt`
  and the AppVeyor configuration removed; CircleCI configuration updated.
- README rewritten: the project is in maintenance mode; it now documents which
  scikit-learn, skrub and scikit-lego features replace each of PipeGraph's use cases,
  what remains PipeGraph-specific, and the known limitations of the current code
  (fitted graphs cannot be pickled, partial `check_estimator` compliance, no
  metadata routing).
- Packaging: all metadata consolidated in `pyproject.toml` (`setup.py` is now only
  the CircleCI version-check shim; `setup.cfg` and `README.rst` removed). The long
  description on PyPI is taken from `README.md`. `matplotlib`, `seaborn` and `scipy`
  are no longer runtime dependencies: install `pipegraph[examples]` to run the gallery.

## 0.3.3 — 2019-10-09

- Changes in `Demultiplexer` and `Multiplexer`.

Earlier releases: see the git history.
