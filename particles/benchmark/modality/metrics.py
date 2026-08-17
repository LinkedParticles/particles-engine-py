"""Pure modality-classification metrics.

Functions over aligned ``(expected, emitted)`` ``assertion_modality`` pairs —
one pair per gold label the equivalence judge matched to an emitted claim. No
I/O, no SDK seams; the runner accumulates the pairs and calls these to roll up
the report. Distinct from :mod:`particles.benchmark.metrics` (content
precision/recall/ECE): those measure whether the *right facts* were emitted,
these measure whether emitted facts got the *right modality*.

The vacuous-denominator convention matches the content harness
(:func:`particles.benchmark.metrics.compute_precision`): precision / recall of
a modality with zero relevant pairs is ``1.0``. Because that can read as a
falsely-perfect score when support is zero, the runner surfaces per-modality
support (via the confusion cells) alongside the rate so a ``1.0`` on no
evidence is visible rather than flattering.
"""

from __future__ import annotations

from collections import Counter

from particles.core.schema import AssertionModality

# (expected_modality, emitted_modality)
ModalityPair = tuple[AssertionModality, AssertionModality]


def confusion_counts(pairs: list[ModalityPair]) -> dict[ModalityPair, int]:
    """Tally aligned pairs into a ``(expected, emitted) -> count`` confusion map."""
    counts: Counter[ModalityPair] = Counter(pairs)
    return dict(counts)


def modality_precision(pairs: list[ModalityPair], modality: AssertionModality) -> float:
    """Precision for ``modality`` — of claims *emitted as* it, the fraction correct.

    ``TP / (TP + FP)``. Returns ``1.0`` when the extractor emitted nothing with
    this modality (no false positives are possible); read it together with the
    modality's emitted-support count.
    """
    predicted = sum(1 for _exp, emi in pairs if emi == modality)
    if predicted == 0:
        return 1.0
    correct = sum(1 for exp, emi in pairs if exp == modality and emi == modality)
    return correct / predicted


def modality_recall(pairs: list[ModalityPair], modality: AssertionModality) -> float:
    """Recall for ``modality`` — of claims that *should be* it, the fraction caught.

    ``TP / (TP + FN)``. Returns ``1.0`` when no gold label carries this modality
    (the extractor cannot miss what was never expected); read it together with
    the modality's expected-support count.
    """
    actual = sum(1 for exp, _emi in pairs if exp == modality)
    if actual == 0:
        return 1.0
    correct = sum(1 for exp, emi in pairs if exp == modality and emi == modality)
    return correct / actual


def false_non_falsifiable_rate(pairs: list[ModalityPair]) -> float:
    """The dangerous-miss rate the inverted journal default raises.

    Of the aligned claims whose *expected* modality is ``FALSIFIABLE``, the
    fraction the extractor tagged **non**-``FALSIFIABLE`` — i.e. a real
    world-fact wrongly exempted from contradiction-arbitration. This is the
    complement of ``modality_recall(pairs, FALSIFIABLE)``, surfaced explicitly
    because it is the specific error this suite exists to bound. Returns ``0.0``
    when no gold label is ``FALSIFIABLE`` (no such miss is possible).
    """
    expected_falsifiable = sum(1 for exp, _emi in pairs if exp == AssertionModality.FALSIFIABLE)
    if expected_falsifiable == 0:
        return 0.0
    demoted = sum(
        1
        for exp, emi in pairs
        if exp == AssertionModality.FALSIFIABLE and emi != AssertionModality.FALSIFIABLE
    )
    return demoted / expected_falsifiable


def narrative_emission_rate(expected_cases: int, emitted_cases: int) -> float:
    """Fraction of entries that should have a ``NARRATIVE`` and got one.

    ``expected_cases`` is the number of cases with ``narrative_expected=True``;
    ``emitted_cases`` is how many of those actually emitted a ``NARRATIVE``
    candidate. Returns ``1.0`` when no case expects a narrative (vacuous, per
    the harness convention).
    """
    if expected_cases == 0:
        return 1.0
    return emitted_cases / expected_cases
