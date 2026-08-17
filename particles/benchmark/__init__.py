"""Extraction-quality benchmark harness (techspec §13.3).

This package implements the reference runner for the frozen
:class:`BenchmarkSuite` schema. The runner answers *"given this
gold-standard suite, how well does extractor X perform?"* with three
metrics: ``precision``, ``recall``, ``calibration_error``.

The harness is *report-only* in this revision — it never modifies the
particle store, never auto-writes ``calibration_history`` entries, and
never gates extractor registration. Temperature-scaling calibration
(the consumer of ``calibration_error``) is a separate, deferred ADR.

Public surface:

* :mod:`.schema` — :class:`BenchmarkSuite`, :class:`BenchmarkCase`,
  :class:`ExpectedParticle`, :class:`RequiredMetric` (verbatim
  techspec §13.3)
* :mod:`.loader` — :func:`load_suite`, :func:`discover_suites`
* :mod:`.equivalence` — :func:`match_emitted_to_expected`
* :mod:`.metrics` — :func:`compute_precision`, :func:`compute_recall`,
  :func:`compute_calibration_error`

The runner that ties these together lives in :mod:`.runner` (
commit 2/3); the CLI lands alongside it.
"""
