"""Precision / recall / Expected Calibration Error.

These functions consume the equivalence result (which emitted particles
matched which expected ones) and emit the three required metrics from
techspec §13.3. Pure functions; no I/O, no SDK seams.
"""

from __future__ import annotations

from particles.core.schema import Particle
from particles.extraction.calibration import expected_calibration_error


def compute_precision(matched_emitted: int, total_emitted: int) -> float:
    """Fraction of emitted particles that matched an expected particle.

    ``total_emitted == 0`` is treated as precision = 1.0 — vacuously
    no spurious particles. This avoids a division-by-zero blow-up when
    an extractor produces nothing against a particular case, and is
    the convention scikit-learn's ``precision_score`` uses for
    ``zero_division=1``. The runner separately surfaces the
    ``particles_emitted`` count so an operator can spot the degenerate
    case in the report.
    """
    if total_emitted == 0:
        return 1.0
    return matched_emitted / total_emitted


def compute_recall(matched_required: int, total_required: int) -> float:
    """Fraction of required expected particles that the extractor emitted.

    ``total_required == 0`` is treated as recall = 1.0 — the suite
    placed no recall obligation on the extractor for this case, so the
    extractor cannot fail at it. Optional-only suites that want a
    non-trivial recall floor should mark at least one particle
    ``required: true``.
    """
    if total_required == 0:
        return 1.0
    return matched_required / total_required


def compute_calibration_error(
    matched_ids: set[str],
    emitted: list[Particle],
    *,
    bins: int = 10,
) -> float:
    """Expected Calibration Error over equal-width confidence bins.

    For each bin ``b``:
      * ``conf_b`` = mean of ``particle.confidence.value`` for emitted
        particles whose stated confidence falls in the bin
      * ``acc_b`` = fraction of those particles that matched an expected
        particle (``id in matched_ids``)
      * the bin contributes ``(|b| / N) × |conf_b - acc_b|`` to ECE

    ``ECE = 0.0`` is perfect calibration; the worst case (all-wrong
    extractor that reports 1.0 on every particle) is ``ECE = 1.0``.
    Returns 0.0 on an empty ``emitted`` list — no claims is
    technically perfect calibration (and the operator sees from
    ``particles_emitted`` that nothing was produced).

    ``bins=10`` is the convention from Guo et al. 2017 ("On
    Calibration of Modern Neural Networks"). Coarser bins (5) hide
    over/under-confidence in the tails; finer bins (20) start to
    over-fit on small per-case populations.

    Adapter over the canonical
    :func:`particles.extraction.calibration.expected_calibration_error`
    : it maps each emitted particle to
    ``(confidence.value, matched?)`` and delegates the binning math, so the
    benchmark harness and the extractor calibration tooling share one
    implementation.
    """
    confidences = [p.confidence.value for p in emitted]
    correctness = [p.id in matched_ids for p in emitted]
    return expected_calibration_error(confidences, correctness, bins=bins)
