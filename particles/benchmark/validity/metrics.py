"""Pure event-anchored-validity metrics.

Functions over aligned ``(expected_boundary, emitted_boundary)`` pairs — one pair
per gold label the equivalence judge matched to an emitted claim, each side a
``date`` or ``None`` (``None`` = "no validity boundary"). No I/O, no SDK seams;
the runner accumulates the pairs and calls these to roll up the report. Distinct
from the content harness (precision/recall/ECE), the modality harness
(``assertion_modality`` quality), and the polarity harness (framing quality):
these measure whether the extractor put a **validity boundary** on the right
claims, at roughly the right date.

The headline is :func:`wrong_expiry_rate` — of the aligned claims whose gold is
**durable** (no boundary), the fraction the extractor wrongly assigned a
``valid_until``. Because a ``valid_until`` can flip a particle out of ACTIVE (the
§9.3 staleness lint retires it as ``VALIDITY_EXPIRED``), a durable fact wrongly
bounded is **silently retired** — the over-eager-expiry failure mode this whole
feature is built to keep rare, so it is surfaced as the headline (the direct
analog of the polarity harness's ``wrong_declined_rate``).

The vacuous-denominator convention matches the sibling harnesses: precision /
recall / date-accuracy over zero relevant pairs is ``1.0``, and the danger rate
over zero durable pairs is ``0.0``. Because a ``1.0`` (or ``0.0``) on no evidence
reads as falsely perfect, the runner surfaces per-class support counts alongside
the rates so an empty denominator is visible rather than flattering.
"""

from __future__ import annotations

from datetime import date

# (expected_boundary, emitted_boundary); ``None`` means "no valid_until".
ValidityPair = tuple[date | None, date | None]


def wrong_expiry_rate(pairs: list[ValidityPair]) -> float:
    """The headline danger: durable facts wrongly assigned a ``valid_until``.

    Of the aligned claims whose *gold* boundary is ``None`` (durable), the
    fraction the extractor gave any ``valid_until`` — i.e. a permanently-true fact
    that the §9.3 staleness lint will later **silently retire** as
    ``VALIDITY_EXPIRED``. This is the over-eager-expiry failure mode; lower is
    better. Returns ``0.0`` when no gold label is durable (no such error is
    possible).
    """
    durable = sum(1 for exp, _emi in pairs if exp is None)
    if durable == 0:
        return 0.0
    wrongly_bounded = sum(1 for exp, emi in pairs if exp is None and emi is not None)
    return wrongly_bounded / durable


def expiry_precision(pairs: list[ValidityPair]) -> float:
    """Existence precision — of claims *given* a boundary, the fraction gold agrees.

    ``TP / (TP + FP)`` where a positive is "the extractor emitted a
    ``valid_until``" and correct means "gold says this claim is date-bounded".
    Measures whether a boundary belongs at all, separately from whether its date
    is right (:func:`date_accuracy`). Returns ``1.0`` when the extractor emitted
    no boundary (no false positives possible); read it with the emitted-support
    count.
    """
    predicted = sum(1 for _exp, emi in pairs if emi is not None)
    if predicted == 0:
        return 1.0
    correct = sum(1 for exp, emi in pairs if emi is not None and exp is not None)
    return correct / predicted


def expiry_recall(pairs: list[ValidityPair]) -> float:
    """Existence recall — of claims that *should* be bounded, the fraction caught.

    ``TP / (TP + FN)``. Returns ``1.0`` when no gold label is date-bounded (the
    extractor cannot miss what was never expected); read it with the
    expected-support count. Missing a boundary is the *safe* failure direction
    (the claim keeps decay treatment), so this is the secondary axis to the
    headline danger rate.
    """
    actual = sum(1 for exp, _emi in pairs if exp is not None)
    if actual == 0:
        return 1.0
    caught = sum(1 for exp, emi in pairs if exp is not None and emi is not None)
    return caught / actual


def date_accuracy(pairs: list[ValidityPair], tolerance_days: int) -> float:
    """Of correctly-flagged bounded claims, the fraction whose date is close.

    Over the pairs where **both** gold and emitted carry a boundary, the fraction
    whose emitted date is within ``tolerance_days`` of gold (absolute). Decoupled
    from existence precision/recall so "should there be a boundary" and "is the
    date right" are measured independently. Returns ``1.0`` when no pair has both
    boundaries set (nothing to score); read it with the both-bounded support.
    """
    both = [(exp, emi) for exp, emi in pairs if exp is not None and emi is not None]
    if not both:
        return 1.0
    close = sum(1 for exp, emi in both if abs((emi - exp).days) <= tolerance_days)
    return close / len(both)


def support_counts(pairs: list[ValidityPair]) -> dict[str, int]:
    """Class supports so a vacuous ``1.0`` / ``0.0`` rate is visible, not flattering.

    Returns the aligned-pair counts the report renders beside the rates:
    ``durable`` (gold has no boundary), ``bounded`` (gold has one),
    ``emitted`` (extractor set one), and ``both`` (gold and emitted both set).
    """
    return {
        "durable": sum(1 for exp, _emi in pairs if exp is None),
        "bounded": sum(1 for exp, _emi in pairs if exp is not None),
        "emitted": sum(1 for _exp, emi in pairs if emi is not None),
        "both": sum(1 for exp, emi in pairs if exp is not None and emi is not None),
    }
