"""Query-time source-trust evaluation.

Loads the operator's source-trust policy — ``SourceTrustStatement`` rows
(§6.4) and URL trust rules — into an in-process ``TrustPolicy``
snapshot once per query, then evaluates it against each candidate's
provenance with no per-particle DB round trips. This mirrors the extractor
trust-weight cache (``populate_trust_cache``): in ``query_federated`` the
snapshot is loaded from the **viewer's** store and applied to every store's
candidates, so "consensus is per-viewer trust at query time".

Resolve-or-None semantics: an explicitly asserted rank applies
unchanged; total absence of applicable policy yields ``None`` and the caller
treats the factor as neutral (1.0). The §6.6 conflict ladder's 0.50
no-information baseline never fires here — a store with no trust
configuration is byte-for-byte unaffected by query-time source trust.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Particle, TrustLensDefinition
from particles.extraction.registry import infer_domain
from particles.store.lens_store import get_adopted_lenses
from particles.store.trust_store import SourceTrustRow, TrustStatementRow


@dataclass(frozen=True)
class TrustPolicy:
    """One store's source-trust policy, snapshotted for in-process evaluation.

    Attributes:
        statements: ``{(domain, source_ref_type, source_ref_value): trust_rank}``
            from ``SourceTrustStatement`` rows; for duplicate keys the most
            recently asserted statement wins (matching ``get_trust_rank``).
        domain_scores: explicitly configured domain baseline rows,
            including the wildcard ``"*"`` row when present.
        url_patterns: compiled URL-pattern modifier rules.
    """

    statements: dict[tuple[str, str, str], float]
    domain_scores: dict[str, float]
    url_patterns: tuple[tuple[re.Pattern[str], float], ...]

    def evaluate(
        self,
        entry_id: str | None,
        source_type: str,
        uri_r: str | None,
        author_id: str | None = None,
    ) -> float | None:
        """Return the asserted source_trust_rank for one candidate, or None.

        Walks the same layers as ``get_layered_trust_rank``,
        in the same order, so the query path and the §6.6 ladder cannot
        disagree on precedence:

          1. ``CORPUS_ENTRY``-scoped statement (most specific)
          2. ``AUTHOR``-scoped statement (the snapshot's ``author_id``, §6.5)
          3. ``SOURCE_TYPE``-scoped statement
          4. URL rules — explicit rows only
          5. nothing configured → ``None`` (caller applies 1.0)

        The knowledge domain for layers 1–3 is derived exactly as the ladder
        derives it: ``infer_domain(source_type)``; a None domain (e.g.
        WEB_PAGE, PDF) skips the statement layers. A candidate with no
        ``author_id`` (non-UGC source) skips layer 2.
        """
        domain = infer_domain(source_type)
        if domain is not None:
            if entry_id is not None:
                rank = self.statements.get((domain, "CORPUS_ENTRY", entry_id))
                if rank is not None:
                    return rank
            if author_id is not None:
                rank = self.statements.get((domain, "AUTHOR", author_id))
                if rank is not None:
                    return rank
            rank = self.statements.get((domain, "SOURCE_TYPE", source_type))
            if rank is not None:
                return rank
        return self._url_rank(uri_r)

    def _url_rank(self, uri_r: str | None) -> float | None:
        """Evaluate the URL layer, explicit rules only.

        Unlike ``resolve_trust_score`` there is no synthetic 0.50 fallback:
        local / unfetched sources (no URI, ``file://``) and URIs no rule
        matches return ``None``. A URL-pattern modifier with no configured
        domain baseline applies against neutral 1.0 — an explicit operator
        assertion bites even without a baseline row.
        """
        if uri_r is None or uri_r.startswith("file://"):
            return None
        netloc = urlparse(uri_r).netloc or uri_r
        base = self.domain_scores.get(netloc)
        if base is None:
            base = self.domain_scores.get("*")

        modifier_total = 0.0
        modifier_matched = False
        for pattern, modifier in self.url_patterns:
            if pattern.search(uri_r):
                modifier_total += modifier
                modifier_matched = True

        if base is None and not modifier_matched:
            return None
        return max(0.0, min(1.0, (base if base is not None else 1.0) + modifier_total))


#: Neutral policy — every evaluation returns None (factor 1.0).
EMPTY_TRUST_POLICY = TrustPolicy(statements={}, domain_scores={}, url_patterns=())


def _is_empty(policy: TrustPolicy) -> bool:
    return not (policy.statements or policy.domain_scores or policy.url_patterns)


async def _load_local_parts(
    session: AsyncSession,
) -> tuple[
    dict[tuple[str, str, str], float],
    dict[str, float],
    list[tuple[re.Pattern[str], float]],
    set[str],
]:
    """Build the **local** policy components (no lens overlay).

    Returns ``(statements, domain_scores, url_patterns, local_pattern_keys)`` —
    the store's own ``SourceTrustStatement`` rows and URL rules only.
    Shared by :func:`load_trust_policy` (which then overlays adopted lenses) and
    :func:`load_local_trust_policy` (which uses the local policy standalone, for
    the contestedness member set).
    """
    stmt_rows = (
        (
            await session.execute(
                select(TrustStatementRow).order_by(TrustStatementRow.asserted_at.asc())
            )
        )
        .scalars()
        .all()
    )
    # Ascending order so the most recently asserted statement overwrites
    # earlier ones for the same (domain, ref_type, ref_value) key.
    statements: dict[tuple[str, str, str], float] = {
        (r.domain, r.source_ref_type, r.source_ref_value): r.trust_rank for r in stmt_rows
    }

    rule_rows = (await session.execute(select(SourceTrustRow))).scalars().all()
    domain_scores: dict[str, float] = {}
    url_patterns: list[tuple[re.Pattern[str], float]] = []
    local_pattern_keys: set[str] = set()
    for rule in rule_rows:
        if rule.scope == "domain" and rule.score is not None:
            domain_scores[rule.pattern] = rule.score
        elif rule.scope == "url_pattern" and rule.modifier is not None:
            # Mirror resolve_trust_score: an invalid regex never matches.
            with contextlib.suppress(re.error):
                url_patterns.append((re.compile(rule.pattern), rule.modifier))
                local_pattern_keys.add(rule.pattern)
    return statements, domain_scores, url_patterns, local_pattern_keys


async def load_local_trust_policy(session: AsyncSession) -> TrustPolicy:
    """Snapshot the store's **local** policy alone — no lens overlay.

    The local-policy member of the contestedness policy set: the store's own
    trust statements and URL rules with resolve-or-None neutrality, exactly as
    :func:`load_trust_policy` evaluates them before any lens composition. A
    member even when empty — the neutral policy *is* the viewer's operative
    rendering.
    """
    statements, domain_scores, url_patterns, _ = await _load_local_parts(session)
    return TrustPolicy(
        statements=statements,
        domain_scores=domain_scores,
        url_patterns=tuple(url_patterns),
    )


def lens_to_trust_policy(lens: TrustLensDefinition) -> TrustPolicy:
    """Convert one lens's portable layers into a **standalone** TrustPolicy.

    A per-lens member of the contestedness policy set: the lens's own statements
    and URL rules only, no local overlay and no cross-lens min — each lens is
    evaluated in full and alone, because composition would collapse the very
    spread being measured. Mirrors the per-lens overlay shape in
    :func:`load_trust_policy`: statements key on ``SOURCE_TYPE``; within the lens
    duplicate keys take the minimum rank, and url-pattern modifiers sum.
    """
    statements: dict[tuple[str, str, str], float] = {}
    for s in lens.statements:
        key = (s.domain, "SOURCE_TYPE", s.source_type)
        current = statements.get(key)
        statements[key] = s.trust_rank if current is None else min(current, s.trust_rank)

    domain_scores: dict[str, float] = {}
    modifier_sums: dict[str, float] = {}
    for r in lens.url_rules:
        if r.scope == "domain" and r.score is not None:
            current = domain_scores.get(r.pattern)
            domain_scores[r.pattern] = r.score if current is None else min(current, r.score)
        elif r.scope == "url_pattern" and r.modifier is not None:
            modifier_sums[r.pattern] = modifier_sums.get(r.pattern, 0.0) + r.modifier

    url_patterns: list[tuple[re.Pattern[str], float]] = []
    for pattern, total in modifier_sums.items():
        with contextlib.suppress(re.error):
            url_patterns.append((re.compile(pattern), total))

    return TrustPolicy(
        statements=statements,
        domain_scores=domain_scores,
        url_patterns=tuple(url_patterns),
    )


async def load_trust_policy(session: AsyncSession) -> TrustPolicy:
    """Snapshot one store's trust policy: local rows composed over adopted lenses.

    Called once per query — from the queried store in ``query``, from the
    **viewer's** store in ``query_federated``.

    Composition: the store's **local** statements and the rules always win per key; for keys local policy does not assert, adopted
    lenses contribute **most-skeptical-wins** — the minimum rank/score across
    lenses (for URL-pattern modifiers: per-lens sums, minimum of the sums).
    A key neither local policy nor any lens asserts stays absent — the
    silence-is-neutral rule is unchanged by adoption.
    """
    statements, domain_scores, url_patterns, local_pattern_keys = await _load_local_parts(session)

    # overlay adopted lenses under the local policy.
    lenses = await get_adopted_lenses(session)
    if lenses:
        lens_statements: dict[tuple[str, str, str], float] = {}
        lens_domains: dict[str, float] = {}
        lens_modifier_sums: dict[str, float] = {}
        for lens in lenses:
            for s in lens.statements:
                key = (s.domain, "SOURCE_TYPE", s.source_type)
                current = lens_statements.get(key)
                lens_statements[key] = (
                    s.trust_rank if current is None else min(current, s.trust_rank)
                )
            per_lens_sums: dict[str, float] = {}
            for r in lens.url_rules:
                if r.scope == "domain" and r.score is not None:
                    current = lens_domains.get(r.pattern)
                    lens_domains[r.pattern] = r.score if current is None else min(current, r.score)
                elif r.scope == "url_pattern" and r.modifier is not None:
                    # Modifiers sum within a lens; min-of-sums across lenses.
                    per_lens_sums[r.pattern] = per_lens_sums.get(r.pattern, 0.0) + r.modifier
            for pattern, total in per_lens_sums.items():
                current_sum = lens_modifier_sums.get(pattern)
                lens_modifier_sums[pattern] = (
                    total if current_sum is None else min(current_sum, total)
                )

        for key, rank in lens_statements.items():
            statements.setdefault(key, rank)
        for pattern, score in lens_domains.items():
            domain_scores.setdefault(pattern, score)
        for pattern, total in lens_modifier_sums.items():
            if pattern not in local_pattern_keys:
                with contextlib.suppress(re.error):
                    url_patterns.append((re.compile(pattern), total))

    return TrustPolicy(
        statements=statements,
        domain_scores=domain_scores,
        url_patterns=tuple(url_patterns),
    )


async def load_source_trust_ranks(
    session: AsyncSession,
    particles: list[Particle],
) -> dict[str, float]:
    """Return ``{particle_id: asserted source_trust_rank}`` for one store.

    Convenience composition of :func:`load_trust_policy` and the batched
    provenance walk for read surfaces that compute effective confidence
    outside the query orchestrator (the anki / wiki exporters,
    which skip recency decay but must still apply source trust). Particles
    with no applicable policy are **absent** from the map — the caller
    treats absence as the neutral 1.0 factor.

    Short-circuits to ``{}`` without the corpus-entry round trips when the
    store has no trust configuration at all, preserving the cheap path the
    deck/synthesis exporters were designed around.
    """
    policy = await load_trust_policy(session)
    if _is_empty(policy) or not particles:
        return {}

    from particles.operations.query.source_info import load_source_rows

    rows = await load_source_rows(session, particles)
    ranks: dict[str, float] = {}
    for p in particles:
        _, source_type, entry_id, uri_r, author_id = rows.get(p.id, (None, "", None, None, None))
        rank = policy.evaluate(entry_id, source_type, uri_r, author_id)
        if rank is not None:
            ranks[p.id] = rank
    return ranks
