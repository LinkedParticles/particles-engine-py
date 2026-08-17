"""Notion exporter — the first API-target exporter.

Unlike every other shipped exporter, the Notion exporter does not write to the
local filesystem (``output`` is ``None``). It projects the store into **one
operator-provided Notion database**: each Subject becomes a database row (a
page) titled with its disambiguated display name, and each ACTIVE
particle the subject participates in becomes a block in that page's managed
body, carrying its effective confidence, status, and a link to its corpus
provenance.

Credential pattern (Part A):

* The integration token is a **secret**, read once via
  :func:`particles.secrets.get_notion_api_key` as the **first statement** of
  :meth:`NotionExporter.export` — before any store read or network call, so a
  missing token can never produce a half-written workspace. The class declares
  ``REQUIRES_SECRET = "NOTION_API_KEY"`` so the CLI pre-flight (a courtesy that
  fails faster) and a future ``export --list`` learn *that* a secret is needed
  and *which* env var names it, without ever reading the value.
* Non-secret parameters (target database id, property names, managed-heading
  text) live in ``config.notion`` (:class:`particles.config.NotionConfig`) and
  are overridable per-invocation via ``**options`` — never alongside the token.

Mapping & idempotency (Part B):

* One database row per Subject (``min_particles`` count check);
  one block per particle, multi-subject edges de-duplicated by particle id.
* Idempotent upsert: each page is stamped with the Particles subject id in the
  ``subject_id_property`` rich-text property; re-sync queries the database for
  an existing row with that id and updates it rather than creating a duplicate.
* The block body is reconciled by owning the managed block range under a
  sentinel heading (default): re-sync deletes the old managed blocks and writes
  the current set, so stale particles drop. ``--no-update-blocks`` opts into a
  conservative create-only mode (a page's blocks are written once, never
  rewritten — hand-edits below the heading survive).
* The cross-exporter quality filter (``min_particle_confidence``) and
  the ``min_particles`` count both apply before any page/block.

Dry-run: a dry run reads the store and computes the full plan
but makes **zero** Notion API writes and **skips the existence probe** — it
reports planned totals only (``pages_created``/``pages_updated`` are ``None``),
so it needs only a readable token and never mutates the workspace.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Particle, Subject
from particles.core.scoring.confidence import compute_effective_confidence
from particles.core.status import Status
from particles.exporters.summaries import NotionSummary
from particles.render.markdown import build_subject_naming, exclude_non_asserted

log = logging.getLogger(__name__)

_NOTION_API_BASE = "https://api.notion.com/v1"
# Pinned Notion API version. Notion requires this header on every request and
# breaking changes are gated behind a new date; pin it so a server-side default
# bump can't silently change our request shape.
_NOTION_VERSION = "2022-06-28"
# Notion caps a single block-children append at 100 blocks and a delete must be
# one block at a time. We batch appends to this size.
_APPEND_BATCH = 100


class NotionExportError(Exception):
    """Raised when the Notion API returns an error the exporter can't proceed past."""


class _NotionClient:
    """Thin async wrapper over the Notion REST API.

    Routes through the shared :func:`particles.http.particles_client` so the
    SSRF transport and the configured timeout / user-agent apply. The token is
    passed in the ``Authorization: Bearer`` header; ``Notion-Version`` is
    required on every request. Every call increments :attr:`api_calls` so the
    summary can report the real-run network cost.
    """

    def __init__(self, token: str, timeout: float | None = None) -> None:
        self._token = token
        self._timeout = timeout
        self.api_calls = 0

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        }

    async def _request(self, method: str, path: str, *, json: Any = None) -> dict[str, Any]:
        # Deferred import (branch-/resource-local, AGENTS.md case 2/4): the
        # shared client is only needed on the real-run path, and tests patch
        # ``particles.http.particles_client``.
        from particles.http import particles_client

        self.api_calls += 1
        url = f"{_NOTION_API_BASE}{path}"
        async with particles_client(timeout=self._timeout, extra_headers=self._headers()) as client:
            resp = await client.request(method, url, json=json)
        if resp.status_code >= 400:
            detail = ""
            try:
                detail = resp.json().get("message", "")
            except Exception:  # pragma: no cover — non-JSON error body
                detail = resp.text[:200]
            raise NotionExportError(
                f"Notion API {method} {path} failed ({resp.status_code}): {detail}"
            )
        body = resp.json()
        return body if isinstance(body, dict) else {}

    async def find_page_by_subject_id(
        self, database_id: str, id_property: str, subject_id: str
    ) -> str | None:
        """Return the page id of an existing row carrying ``subject_id``, else None."""
        body = await self._request(
            "POST",
            f"/databases/{database_id}/query",
            json={
                "filter": {
                    "property": id_property,
                    "rich_text": {"equals": subject_id},
                },
                "page_size": 1,
            },
        )
        results = body.get("results") or []
        if results and isinstance(results[0], dict):
            page_id = results[0].get("id")
            return str(page_id) if page_id else None
        return None

    async def create_page(self, database_id: str, properties: dict[str, Any]) -> str:
        body = await self._request(
            "POST",
            "/pages",
            json={"parent": {"database_id": database_id}, "properties": properties},
        )
        return str(body.get("id", ""))

    async def update_page_properties(self, page_id: str, properties: dict[str, Any]) -> None:
        await self._request("PATCH", f"/pages/{page_id}", json={"properties": properties})

    async def list_block_children(self, block_id: str) -> list[dict[str, Any]]:
        """Return the (first page of) child blocks of ``block_id``.

        Notion paginates at 100; for a single subject page the managed range is
        well under that in practice, but we follow ``next_cursor`` to be safe.
        """
        out: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            path = f"/blocks/{block_id}/children?page_size=100"
            if cursor:
                path += f"&start_cursor={cursor}"
            body = await self._request("GET", path)
            results = body.get("results") or []
            out.extend(b for b in results if isinstance(b, dict))
            if not body.get("has_more"):
                break
            cursor = body.get("next_cursor")
            if not cursor:
                break
        return out

    async def delete_block(self, block_id: str) -> None:
        await self._request("DELETE", f"/blocks/{block_id}")

    async def append_blocks(self, block_id: str, children: list[dict[str, Any]]) -> None:
        for start in range(0, len(children), _APPEND_BATCH):
            batch = children[start : start + _APPEND_BATCH]
            await self._request("PATCH", f"/blocks/{block_id}/children", json={"children": batch})


# ---------------------------------------------------------------------------
# Pure store → Notion plan mapping (no I/O — unit-testable in isolation)
# ---------------------------------------------------------------------------


def _rich_text(content: str) -> list[dict[str, Any]]:
    """A single rich-text run. Notion caps a text run at 2000 chars."""
    return [{"type": "text", "text": {"content": content[:2000]}}]


def _subject_properties(
    subject: Subject,
    display_name: str,
    particle_count: int,
    *,
    id_property: str,
) -> dict[str, Any]:
    """Build the database-row properties for a subject page.

    The page **title** is the disambiguated display name; the upsert-key
    ``id_property`` carries the Particles subject id; ``Subject Class`` /
    ``Aliases`` / ``External IDs`` / ``Particle Count`` carry the rest. We send
    only properties the exporter owns; an operator's extra columns are left
    untouched by ``update_page_properties`` (PATCH is a merge).
    """
    external = ", ".join(f"{ref.namespace}:{ref.id}" for ref in subject.external_ids)
    props: dict[str, Any] = {
        "Name": {"title": _rich_text(display_name)},
        id_property: {"rich_text": _rich_text(subject.id)},
        "Subject Class": {"rich_text": _rich_text(subject.subject_class or "")},
        "Aliases": {"rich_text": _rich_text(", ".join(subject.aliases))},
        "External IDs": {"rich_text": _rich_text(external)},
        "Particle Count": {"number": particle_count},
    }
    return props


def _particle_block(
    particle: Particle, effective_confidence: float, source_uri: str | None
) -> dict[str, Any]:
    """Render one particle as a bulleted-list block.

    The content carries the claim, its effective confidence and status inline
    (consistent with the Markdown renderer's callout shape), and — when the
    corpus entry has a public URL — a trailing link to the source so any claim
    is traceable to its provenance.
    """
    text = f"{particle.content}  (confidence {effective_confidence:.2f} · {particle.status})"
    runs: list[dict[str, Any]] = [{"type": "text", "text": {"content": text[:2000]}}]
    if source_uri:
        runs.append({"type": "text", "text": {"content": " "}})
        runs.append(
            {
                "type": "text",
                "text": {"content": "[source]", "link": {"url": source_uri}},
            }
        )
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": runs},
    }


def _managed_heading_block(heading_text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {"rich_text": _rich_text(heading_text)},
    }


def _is_managed_heading(block: dict[str, Any], heading_text: str) -> bool:
    """True when ``block`` is the sentinel heading that opens our managed range."""
    if block.get("type") != "heading_2":
        return False
    runs = (block.get("heading_2") or {}).get("rich_text") or []
    text = "".join(r.get("plain_text", (r.get("text") or {}).get("content", "")) for r in runs)
    return text.strip() == heading_text.strip()


class _SubjectPlan:
    """The fully-computed sync plan for one subject (pure data, no I/O)."""

    __slots__ = ("subject", "display_name", "properties", "blocks", "particle_count")

    def __init__(
        self,
        subject: Subject,
        display_name: str,
        properties: dict[str, Any],
        blocks: list[dict[str, Any]],
        particle_count: int,
    ) -> None:
        self.subject = subject
        self.display_name = display_name
        self.properties = properties
        self.blocks = blocks
        self.particle_count = particle_count


class NotionExporter:
    """Notion exporter plugin — store → one Notion database.

    Options accepted by :meth:`export` (all optional):

    * ``database_id`` (str) — override ``config.notion.database_id`` for this run.
    * ``min_particles`` (int) — minimum ACTIVE particle count for a subject to
      get a page. Defaults to 0 (every non-empty subject).
    * ``min_particle_confidence`` (float) — cross-exporter filter on
      effective confidence, applied before the count check and any write.
    * ``no_update_blocks`` (bool, default False) — create-only: a page's blocks
      are written once and never rewritten on re-sync.
    * ``dry_run`` (bool, default False) — read the store, compute the plan, make
      ZERO API writes and skip the existence probe.
    * ``include_non_asserted`` (bool, default False) — keep DECLINED /
      HYPOTHETICAL particles; excluded from the default surface.
    """

    FORMAT = "notion"
    REQUIRES_SECRET = "NOTION_API_KEY"

    async def export(
        self,
        session: AsyncSession,
        output: Path | None,
        **options: object,
    ) -> NotionSummary:
        # --- A3 (authoritative): credential read is the FIRST statement, before
        # any store read or network call, so a missing token fails loud with no
        # partial write. The CLI pre-flight is a courtesy that fails faster; this
        # is the single source of truth for callers that bypass the CLI. A dry
        # run still needs a (readable) token — Notion has no anonymous mode.
        from particles.secrets import get_notion_api_key

        token = get_notion_api_key()

        from particles.config import get_config

        cfg = get_config().notion
        common_cfg = get_config().exporter_common
        http_cfg = get_config().http

        dry_run = bool(options.get("dry_run", False))
        update_blocks = not bool(options.get("no_update_blocks", False))
        min_particles = int(options.get("min_particles", 0))  # type: ignore[call-overload]
        raw_mpc = options.get("min_particle_confidence")
        min_particle_confidence = (
            float(raw_mpc) if raw_mpc is not None else common_cfg.min_particle_confidence  # type: ignore[arg-type]
        )
        database_id = str(options.get("database_id") or cfg.database_id or "").strip()
        id_property = cfg.subject_id_property
        heading_text = cfg.managed_block_heading

        if not database_id:
            raise ValueError(
                "No Notion database id configured. Set `notion.database_id` in "
                "config.yaml or pass `--database-id`. Share that database with "
                "your integration in Notion first."
            )

        # --- Build the plan (pure store reads; identical on dry-run and real).
        plans, particles_dropped = await self._build_plans(
            session,
            min_particles=min_particles,
            min_particle_confidence=min_particle_confidence,
            include_non_asserted=bool(options.get("include_non_asserted", False)),
            id_property=id_property,
            heading_text=heading_text,
        )
        subjects_planned = len(plans)
        particles_synced = sum(p.particle_count for p in plans)

        if dry_run:
            # Zero writes, no existence probe: created-vs-updated
            # is unknown, so report planned totals only.
            log.info(
                "notion dry-run: %d subjects, %d particles planned for database %s",
                subjects_planned,
                particles_synced,
                database_id,
            )
            return NotionSummary(
                dry_run=True,
                subjects_planned=subjects_planned,
                particles_synced=particles_synced,
                particles_dropped_below_threshold=particles_dropped,
                min_particle_confidence=min_particle_confidence,
                database_id=database_id,
                update_blocks=update_blocks,
            )

        client = _NotionClient(token, timeout=http_cfg.timeout_seconds)
        pages_created = 0
        pages_updated = 0
        for plan in plans:
            existing_page_id = await client.find_page_by_subject_id(
                database_id, id_property, plan.subject.id
            )
            if existing_page_id is None:
                page_id = await client.create_page(database_id, plan.properties)
                await client.append_blocks(page_id, plan.blocks)
                pages_created += 1
                log.info("notion: created page for %s", plan.display_name)
            else:
                await client.update_page_properties(existing_page_id, plan.properties)
                if update_blocks:
                    await self._rewrite_managed_blocks(
                        client, existing_page_id, plan.blocks, heading_text
                    )
                pages_updated += 1
                log.info("notion: updated page for %s", plan.display_name)

        summary = NotionSummary(
            dry_run=False,
            subjects_planned=subjects_planned,
            particles_synced=particles_synced,
            particles_dropped_below_threshold=particles_dropped,
            min_particle_confidence=min_particle_confidence,
            database_id=database_id,
            update_blocks=update_blocks,
            pages_created=pages_created,
            pages_updated=pages_updated,
            api_calls=client.api_calls,
        )
        log.info("notion export summary: %s", summary.model_dump(exclude_none=True))
        return summary

    async def _build_plans(
        self,
        session: AsyncSession,
        *,
        min_particles: int,
        min_particle_confidence: float,
        include_non_asserted: bool,
        id_property: str,
        heading_text: str,
    ) -> tuple[list[_SubjectPlan], int]:
        """Compute the per-subject sync plan from the store (no Notion I/O).

        Returns ``(plans, particles_dropped_below_threshold)``. Reuses the shared
        store helpers named in ``particles/exporters/AGENTS.md`` — no new query
        shapes.
        """
        from particles.corpus.store import get_entry_uri_map
        from particles.operations.query.source_trust import load_source_trust_ranks
        from particles.store.extractor_store import (
            get_cached_trust_weight,
            get_trust_weight_map,
            populate_trust_cache,
        )
        from particles.store.particle_store import get_particles_by_status
        from particles.store.subject_store import (
            list_all_subjects,
            list_particle_subject_pairs,
        )

        all_subjects = await list_all_subjects(session)
        # disambiguation map from the FULL subject set (collisions
        # must be detected before min_particles drops any member).
        naming = build_subject_naming(all_subjects)

        active = await get_particles_by_status(session, Status.ACTIVE)
        # cap. 1: keep a document's rejected / deferred / counterfactual
        # prose off the default surface unless the caller opts in.
        active = exclude_non_asserted(active, {"include_non_asserted": include_non_asserted})
        active_by_id: dict[str, Particle] = {p.id: p for p in active}

        # Trust-weighted effective confidence per particle.
        populate_trust_cache(await get_trust_weight_map(session))
        source_ranks = await load_source_trust_ranks(session, active)
        eff_by_id: dict[str, float] = {}
        for p in active:
            extractor_id = p.extractor_ref.name if p.extractor_ref else ""
            trust = get_cached_trust_weight(extractor_id) if extractor_id else 1.0
            eff_by_id[p.id] = compute_effective_confidence(
                p.confidence.value,
                extractor_trust_weight=trust or 1.0,
                source_trust_rank=source_ranks.get(p.id, 1.0),
                calibration_source=p.confidence.calibration_source,
            )

        # Group ACTIVE particle ids by subject via the join table.
        by_subject: dict[str, list[str]] = defaultdict(list)
        for pid, sid in await list_particle_subject_pairs(session):
            if pid in active_by_id:
                by_subject[sid].append(pid)

        # Provenance URIs for every cited corpus entry (batch).
        needed_entry_ids: set[str] = set()
        for p in active:
            for ref in p.provenance:
                if ref.corpus_entry_id:
                    needed_entry_ids.add(ref.corpus_entry_id)
        corpus_uris = await get_entry_uri_map(session, needed_entry_ids)

        plans: list[_SubjectPlan] = []
        particles_dropped = 0
        for subject in all_subjects:
            # De-dup particle ids (a multi-subject edge appears once per page).
            pids = list(dict.fromkeys(by_subject.get(subject.id, [])))
            particles = sorted((active_by_id[pid] for pid in pids), key=lambda p: p.id)
            # drop below-threshold particles BEFORE the count check.
            kept: list[Particle] = []
            for p in particles:
                if eff_by_id[p.id] < min_particle_confidence:
                    particles_dropped += 1
                    continue
                kept.append(p)
            if len(kept) < min_particles or not kept:
                continue

            display_name = naming.display_name(subject)
            blocks: list[dict[str, Any]] = [_managed_heading_block(heading_text)]
            for p in kept:
                source_uri = _first_source_uri(p, corpus_uris)
                blocks.append(_particle_block(p, eff_by_id[p.id], source_uri))
            properties = _subject_properties(
                subject, display_name, len(kept), id_property=id_property
            )
            plans.append(_SubjectPlan(subject, display_name, properties, blocks, len(kept)))
        return plans, particles_dropped

    async def _rewrite_managed_blocks(
        self,
        client: _NotionClient,
        page_id: str,
        blocks: list[dict[str, Any]],
        heading_text: str,
    ) -> None:
        """Own the managed range: delete old managed blocks, append the current set.

        The exporter owns every block from the sentinel heading to the end of the
        page. On re-sync we delete that range and re-append, so a
        re-run is idempotent and stale particles drop. Blocks *above* the heading
        (operator-added content) are never touched.
        """
        existing = await client.list_block_children(page_id)
        deleting = False
        for block in existing:
            if not deleting and _is_managed_heading(block, heading_text):
                deleting = True
            if deleting:
                block_id = block.get("id")
                if block_id:
                    await client.delete_block(str(block_id))
        await client.append_blocks(page_id, blocks)


def _first_source_uri(particle: Particle, corpus_uris: dict[str, str | None]) -> str | None:
    """The public URL of the particle's first corpus provenance ref, if any."""
    for ref in particle.provenance:
        if ref.corpus_entry_id:
            uri = corpus_uris.get(ref.corpus_entry_id)
            if uri:
                return uri
    return None
