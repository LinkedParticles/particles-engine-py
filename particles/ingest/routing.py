# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Route one particle through the rungs above the §6.6 ladder (D2).

The two write paths, the extract pass and ``reconcile_and_insert`` (the
assertion and interchange-import path), decide what to do with a particle by
the same rungs in the same order:

1. **Off the conflict surface** (DOCUMENT_META; non-asserted):
   insert ACTIVE without conflict-checking.
2. **Exact duplicate of an ACTIVE claim**: append this source's
   provenance to the claim already held.
3. **Twin of a judgment-retired claim or an open hold**: hold the
   re-assertion for review, or absorb it into the open hold.
4. **No live conflict**, including a conflict an earlier candidate of this pass
   already retired: insert ACTIVE.
5. **Otherwise** the §6.6 ladder against the conflict.

"Duplicate" must mean one thing on both paths, so the order lives
here once. :func:`route_particle` is pure: each path gathers its own indexes
and conflict, calls it, and applies the route.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from enum import Enum

from particles.core.schema import Particle
from particles.extraction.polarity import is_non_asserted
from particles.extraction.scope import is_excluded_document_meta
from particles.ingest.duplicate_suppression import DuplicateIndex


class Route(Enum):
    """What a write path does with one particle."""

    UNCHECKED = "unchecked"
    """Off the conflict surface: insert ACTIVE."""
    SUPPRESS = "suppress"
    """Append provenance to the ACTIVE twin."""
    RETIRED_TWIN = "retired_twin"
    """Hold or absorb against the retired twin or open hold."""
    INSERT = "insert"
    """No live conflict: insert ACTIVE."""
    LADDER = "ladder"
    """Run the §6.6 ladder against the conflict."""


def skips_conflict_resolution(properties: dict[str, object] | None) -> bool:
    """True for particles §6.6 conflict resolution must ignore.

    Two off-the-conflict-surface classes share the same treatment, kept out of
    the candidate set and written straight to ACTIVE without conflict-checking:

    * **DOCUMENT_META**: claims about a document's own apparatus,
      not about the world.
    * **non-asserted** (cap. 1): a document's rejected / superseded /
      deferred / counterfactual prose (``polarity`` DECLINED / HYPOTHETICAL). A
      rejected alternative must not manufacture an ``INCONSISTENCY`` against the
      chosen decision.

    Both stay stored and ACTIVE (label, never delete); the query / lint / export
    layers apply the visibility exclusion.
    """
    return is_excluded_document_meta(properties) or is_non_asserted(properties)


def route_particle(
    particle: Particle,
    *,
    duplicates: DuplicateIndex,
    retired_values: DuplicateIndex,
    conflict: Particle | None,
    retired_ids: AbstractSet[str],
) -> tuple[Route, Particle | None]:
    """Pick the rung that handles ``particle``, and the particle it acts on.

    Args:
        particle: The candidate to route.
        duplicates: ACTIVE claims by identity key. Pass an empty
            index when suppression is disabled.
        retired_values: Judgment-retired claims and open holds by the same key.
            Pass an empty index when the quarantine is disabled.
        conflict: The §6.6 conflict candidate, or ``None`` when there is none
            or the search has not run yet.
        retired_ids: Ids retired earlier in this pass. A conflict
            among them is no longer live.

    Returns:
        The route, with its target: the ACTIVE twin for ``SUPPRESS``, the
        retired twin for ``RETIRED_TWIN``, the conflict for ``LADDER``, and
        ``None`` for ``UNCHECKED`` and ``INSERT``.
    """
    if skips_conflict_resolution(particle.properties):
        return Route.UNCHECKED, None
    duplicate_of = duplicates.find(particle)
    if duplicate_of is not None:
        return Route.SUPPRESS, duplicate_of
    retired_twin = retired_values.find(particle)
    if retired_twin is not None:
        return Route.RETIRED_TWIN, retired_twin
    if conflict is None or conflict.id in retired_ids:
        return Route.INSERT, None
    return Route.LADDER, conflict
