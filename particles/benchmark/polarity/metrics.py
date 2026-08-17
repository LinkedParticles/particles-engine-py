"""Pure claim-polarity metrics.

Functions over aligned ``(expected, emitted)`` :class:`ClaimPolarity` pairs —
one pair per gold label the equivalence judge matched to an emitted claim. No
I/O, no SDK seams; the runner accumulates the pairs and calls these to roll up
the report. Distinct from :mod:`particles.benchmark.metrics` (content
precision/recall/ECE) and :mod:`particles.benchmark.modality.metrics`
(``assertion_modality`` quality): these measure whether the document's *framing*
of each claim — ``ASSERTED`` / ``DECLINED`` / ``HYPOTHETICAL`` (
cap. 1) — was classified correctly.

The headline is the :func:`wrong_declined_rate` — a real current decision
(``ASSERTED``) mis-classified ``DECLINED`` is silently dropped from the default
factual surface (query / projection / export), so it is the precision risk that
bears directly on README-projection trust. :func:`wrong_hidden_rate` is its
superset (``DECLINED`` *or* ``HYPOTHETICAL`` — both hide a claim) and equals
``1 - polarity_recall(pairs, ASSERTED)`` whenever any ``ASSERTED`` label aligned.

The vacuous-denominator convention matches the content and modality harnesses:
precision / recall of a polarity with zero relevant pairs is ``1.0``. Because
that can read as a falsely-perfect score when support is zero, the runner
surfaces per-polarity support (via the confusion cells) alongside the rate so a
``1.0`` on no evidence is visible rather than flattering.
"""

from __future__ import annotations

from collections import Counter

from particles.benchmark.polarity.schema import ClaimPolarity

# (expected_polarity, emitted_polarity)
PolarityPair = tuple[ClaimPolarity, ClaimPolarity]

# The non-asserted classes — both are excluded from the default factual surface
# (cap. 1), so an ASSERTED claim landing in either is hidden.
_NON_ASSERTED = (ClaimPolarity.DECLINED, ClaimPolarity.HYPOTHETICAL)


def confusion_counts(pairs: list[PolarityPair]) -> dict[PolarityPair, int]:
    """Tally aligned pairs into a ``(expected, emitted) -> count`` confusion map."""
    counts: Counter[PolarityPair] = Counter(pairs)
    return dict(counts)


def polarity_precision(pairs: list[PolarityPair], polarity: ClaimPolarity) -> float:
    """Precision for ``polarity`` — of claims *emitted as* it, the fraction correct.

    ``TP / (TP + FP)``. Returns ``1.0`` when the extractor emitted nothing with
    this polarity (no false positives are possible); read it together with the
    polarity's emitted-support count.
    """
    predicted = sum(1 for _exp, emi in pairs if emi == polarity)
    if predicted == 0:
        return 1.0
    correct = sum(1 for exp, emi in pairs if exp == polarity and emi == polarity)
    return correct / predicted


def polarity_recall(pairs: list[PolarityPair], polarity: ClaimPolarity) -> float:
    """Recall for ``polarity`` — of claims that *should be* it, the fraction caught.

    ``TP / (TP + FN)``. Returns ``1.0`` when no gold label carries this polarity
    (the extractor cannot miss what was never expected); read it together with
    the polarity's expected-support count.
    """
    actual = sum(1 for exp, _emi in pairs if exp == polarity)
    if actual == 0:
        return 1.0
    correct = sum(1 for exp, emi in pairs if exp == polarity and emi == polarity)
    return correct / actual


def wrong_declined_rate(pairs: list[PolarityPair]) -> float:
    """The headline danger: real decisions wrongly classified ``DECLINED``.

    Of the aligned claims whose *expected* polarity is ``ASSERTED``, the
    fraction the extractor tagged ``DECLINED`` — i.e. a real current decision
    wrongly read as a rejected / superseded alternative and thereby **silently
    hidden** from the default factual surface (query / projection / export). This
    is the specific precision risk that bears on README-projection trust, so it
    is surfaced as the headline. Returns ``0.0`` when no gold label is
    ``ASSERTED`` (no such miss is possible).
    """
    expected_asserted = sum(1 for exp, _emi in pairs if exp == ClaimPolarity.ASSERTED)
    if expected_asserted == 0:
        return 0.0
    declined = sum(
        1 for exp, emi in pairs if exp == ClaimPolarity.ASSERTED and emi == ClaimPolarity.DECLINED
    )
    return declined / expected_asserted


def wrong_hidden_rate(pairs: list[PolarityPair]) -> float:
    """Superset of :func:`wrong_declined_rate` — real decisions hidden any way.

    Of the aligned claims whose *expected* polarity is ``ASSERTED``, the
    fraction the extractor tagged **non**-``ASSERTED`` (``DECLINED`` *or*
    ``HYPOTHETICAL``) — both are excluded from the default surface, so either
    misclassification hides a real decision. This is the complement of
    ``polarity_recall(pairs, ASSERTED)`` whenever any ``ASSERTED`` label aligned,
    surfaced explicitly alongside the ``DECLINED``-specific headline. Returns
    ``0.0`` when no gold label is ``ASSERTED``.
    """
    expected_asserted = sum(1 for exp, _emi in pairs if exp == ClaimPolarity.ASSERTED)
    if expected_asserted == 0:
        return 0.0
    hidden = sum(1 for exp, emi in pairs if exp == ClaimPolarity.ASSERTED and emi in _NON_ASSERTED)
    return hidden / expected_asserted
