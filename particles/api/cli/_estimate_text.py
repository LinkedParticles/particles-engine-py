# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""How an extraction price reads on the terminal.

Shared by ``extract --all-pending`` (its estimate and its budget stop) and
``import vault`` (the price of the extraction it queued), so the two verbs
state one figure the same way.
"""

from __future__ import annotations

from particles.operations.spend_estimate import ExtractionRange


def usd(value: float | None) -> str:
    """A dollar figure, or ``US$?`` when it cannot be priced."""
    return "US$?" if value is None else f"US${value:,.2f}"


def duration_text(seconds: float) -> str:
    """An estimated wall time, in hours from one hour up."""
    if seconds >= 3600:
        return f"about {seconds / 3600:.1f} h"
    return f"about {max(1, round(seconds / 60))} min"


def range_text(band: ExtractionRange, *, name_model: bool = True) -> str:
    """One run's price band at list price, or why it has none."""
    if band.high_usd is None:
        return f"unpriced ({band.model} has no llm.price_per_mtok entry)"
    text = f"about {usd(band.low_usd)}–{usd(band.high_usd)} (ceiling {usd(band.ceiling_usd)})"
    return f"{text} at {band.model} list price" if name_model else text
