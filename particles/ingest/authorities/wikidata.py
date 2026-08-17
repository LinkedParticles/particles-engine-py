"""Wikidata subject authority.

The Wikidata live-lookup path, migrated **verbatim** from ``subject_resolver``
into a registered :class:`SubjectAuthority`. The module-level helpers
(``_wikidata_search``, ``_wikidata_aliases``, ``_wikidata_link_confidence``,
``_is_prefix_expansion``) are kept at module scope so existing tests patch them
at ``particles.ingest.authorities.wikidata.*``.

``WikidataAuthority.resolve`` performs the live lookup and the QID-dedup
*read* (preserving the alias-fetch-skip optimization), but never writes:
inserts / alias-merges / cache stay in ``subject_resolver`` via the neutral
:class:`AuthorityResolution`.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from particles.config import get_config
from particles.core.schema import ApplicabilityClause, ExternalRef
from particles.embeddings import cosine_similarity, get_embedding_model
from particles.http import get_capped, particles_client
from particles.ingest.authorities._shared import _RateLimiter, get_limiter
from particles.ingest.authorities.registry import AuthorityResolution
from particles.store.subject_store import find_by_external_ref
from particles.store.wikidata_cache import get_label

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


def _wikidata_limiter() -> _RateLimiter:
    return get_limiter("wikidata", get_config().subjects.wikidata_rate_limit_rps)


# ---------------------------------------------------------------------------
# Link confidence — moved verbatim from subject_resolver
# ---------------------------------------------------------------------------


#: One WARNING per process for the encoder-free link scorer. A
#: resolution pass calls the scorer once per candidate QID, so warning per call
#: would bury the message it is trying to deliver.
_unscored_warning_emitted = False


def _warn_unscored_once() -> None:
    """Disclose that link scoring is unavailable, once per process."""
    global _unscored_warning_emitted
    if _unscored_warning_emitted:
        return
    _unscored_warning_emitted = True
    log.warning(
        "No embedding model: Wikidata links cannot be scored, so every candidate "
        "attaches at the 0.5 unscoreable sentinel. The abstention cannot "
        "fire and the L-SEM-03 lint cannot flag them, so a plausible-but-wrong "
        "QID will be attached silently. Re-resolve these subjects with the model "
        "available to get real scores."
    )


def _wikidata_link_confidence(description: str, particle_content: str | None) -> float:
    """Score a Wikidata link by cosine similarity between entity description and particle content.

    Returns 1.0 when no particle content is available (conservative: display the link).
    Returns the cosine similarity score [0.0, 1.0] otherwise.

    The ``0.5`` sentinel means *could not be scored*, and the scorer deliberately
    lets it attach: the abstain floor sits strictly below it so that only
    scored-and-low links (the plausible-but-wrong mislinks) are dropped. That
    stays true here — but a bug found the missing-encoder case reaching the
    same sentinel through a `log.debug`, which made "the scorer is unavailable,
    so nothing can be abstained and nothing can be linted" indistinguishable
    from "this particular pair scored 0.5". The outcome is unchanged; the
    silence is not.
    """
    if not particle_content or not description:
        return 0.5  # uncertain but not suppressed by default threshold

    try:
        model = get_embedding_model()
        if model is None:
            _warn_unscored_once()
            return 0.5

        vecs = model.encode(
            [description, particle_content], convert_to_numpy=True, normalize_embeddings=True
        )
        # normalized cosine clamped to [0, 1] — the abstain cutoff this
        # feeds lives on that scale.
        return cosine_similarity(vecs[0], vecs[1])
    except Exception as exc:
        log.debug("Wikidata link confidence computation failed: %s", exc)
        return 0.5


# ---------------------------------------------------------------------------
# Wikidata API — moved verbatim from subject_resolver
# ---------------------------------------------------------------------------


def _is_prefix_expansion(query: str, label: str) -> bool:
    """True if ``label`` looks like an expansion of ``query`` into something else.

    Wikidata's ``wbsearchentities`` does prefix matching, so short common-name
    queries (software, projects, tools) often match the wrong entity:

    1. **Paper-title pattern** — ``query`` then a title separator
       (``:`` ``-`` ``–`` ``—``) then a subtitle. E.g.
       "FlashAttention" → "FlashAttention: Data-centric Interaction…".

    2. **Word-continuation pattern** — ``query`` is a stem; ``label`` extends it
       character-by-character into a different word. E.g. "micrograd"
       → "Microgradients of microbial oxygen consumption…".

    Either pattern adopts a wrong canonical name, so reject both. Legitimate
    longer labels keep a non-alphanumeric, non-separator break after the
    query (e.g. "OpenAI" → "OpenAI Inc.", "PyTorch" → "PyTorch (framework)").
    """
    if not query or not label:
        return False
    if len(label) <= len(query):
        return False
    if not label.lower().startswith(query.lower()):
        return False

    next_char = label[len(query)]

    # Word continuation: the next char extends the query into a longer word.
    if next_char.isalnum():
        return True

    # Title separator (optionally with leading whitespace).
    rest = label[len(query) :].lstrip()
    return bool(rest) and rest[0] in (":", "-", "–", "—")


async def _wikidata_search(name: str) -> dict[str, object] | None:
    """Search Wikidata for an entity by name. Returns the first usable hit.

    Bumps the API limit to 5 and skips prefix-expansion candidates (paper
    titles of the form "<query>: <subtitle>") so a short project name like
    "FlashAttention" doesn't get adopted as a scholarly article's full title.
    Falls through to ``None`` (and thus a bare local Subject) when no
    candidate survives the filter.
    """
    await _wikidata_limiter().acquire()
    try:
        async with particles_client(timeout=10.0) as client:
            resp = await get_capped(
                client,
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbsearchentities",
                    "search": name,
                    "language": "en",
                    "format": "json",
                    "limit": "5",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("search", [])
            for hit in results:
                label = str(hit.get("label", ""))
                if _is_prefix_expansion(name, label):
                    log.info(
                        "Wikidata: skipping prefix-expansion candidate %s (%r) for query %r",
                        hit.get("id"),
                        label,
                        name,
                    )
                    continue
                return hit  # type: ignore[no-any-return]
            return None
    except Exception as exc:
        log.warning("Wikidata search failed for %r: %s", name, exc)
        return None


async def _wikidata_aliases(qid: str) -> list[str]:
    """Fetch English labels and aliases for a Wikidata QID."""
    await _wikidata_limiter().acquire()
    try:
        async with particles_client(timeout=10.0) as client:
            resp = await get_capped(
                client,
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "ids": qid,
                    "props": "labels|aliases",
                    "languages": "en",
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            entity = data.get("entities", {}).get(qid, {})
            aliases: list[str] = []
            # English label
            label = entity.get("labels", {}).get("en", {}).get("value")
            if label:
                aliases.append(label)
            # English aliases
            for a in entity.get("aliases", {}).get("en", []):
                v = a.get("value")
                if v and v not in aliases:
                    aliases.append(v)
            return aliases
    except Exception as exc:
        log.warning("Wikidata alias fetch failed for %s: %s", qid, exc)
        return []


# ---------------------------------------------------------------------------
# The authority
# ---------------------------------------------------------------------------

# Pattern matching the old _NAMESPACE_PATTERNS wikidata row. NB: the captured id
# is the digits only (group 1) — recognize stores "123456", while the live path
# (resolve) stores the full "Q123456". This digits-vs-Qxxx asymmetry is
# pre-existing and preserved verbatim (move-not-rewrite).
_Q_PATTERN = re.compile(r"\bQ(\d{4,})\b")


class WikidataAuthority:
    """General-purpose live authority backed by the Wikidata API.

    The grandfathered broad-applicability authority (``APPLICABILITY = []`` ⇒
    applies to every domain). The § Constrained rule bars *new* unconditioned
    recognizers, not this one — Wikidata is the general fallback.
    """

    NAMESPACE = "wikidata"
    PRIORITY = 30
    LIVE = True
    DEFAULT_LINK_CONFIDENCE = 1.0
    APPLICABILITY: list[ApplicabilityClause] = []  # broad: applies to all domains

    def uri_for(self, external_id: str) -> str | None:
        # external_id is the full "Qxxx" form produced by resolve().
        return f"https://www.wikidata.org/wiki/{external_id}"

    def recognize(self, name: str) -> ExternalRef | None:
        m = _Q_PATTERN.search(name)
        if m:
            # id = digits only (parity with old _detect_namespace_pattern).
            return ExternalRef(namespace=self.NAMESPACE, id=m.group(1))
        return None

    async def resolve(
        self,
        session: AsyncSession,
        name: str,
        *,
        particle_content: str | None,
        domain: str | None,
    ) -> AuthorityResolution | None:
        wikidata_result = await _wikidata_search(name)
        if not wikidata_result:
            return None

        qid: str = str(wikidata_result.get("id", ""))
        description = str(wikidata_result.get("description", ""))

        # QID-dedup read (preserves the alias-fetch-skip optimization): if the
        # QID is already stored, hand the existing subject back and let the
        # resolver own the alias-merge write.
        by_qid = await find_by_external_ref(session, self.NAMESPACE, qid)
        if by_qid:
            return AuthorityResolution(existing=by_qid)

        aliases = await _wikidata_aliases(qid)
        if name not in aliases:
            aliases.append(name)
        canonical = aliases[0] if aliases else name
        link_confidence = _wikidata_link_confidence(description, particle_content)
        if link_confidence < 1.0:
            log.debug(
                "Wikidata link %s (%s) confidence=%.2f for content %r",
                qid,
                description[:60],
                link_confidence,
                (particle_content or "")[:60],
            )
        return AuthorityResolution(
            external_ref=ExternalRef(
                namespace=self.NAMESPACE,
                id=qid,
                uri=self.uri_for(qid),
                confidence=link_confidence,
            ),
            canonical_name=canonical,
            aliases=aliases[1:],  # canonical is first; rest are aliases
            description=description or None,
        )

    async def canonical_name_for(self, session: AsyncSession, external_id: str) -> str | None:
        # external_id is the digits-only id from recognize(); reconstruct the
        # QID key for the label cache (matches the old bare-local fallback).
        qid_key = f"Q{external_id}"
        cached_label = await get_label(session, qid_key)
        if cached_label and cached_label != qid_key:
            return cached_label
        return None
