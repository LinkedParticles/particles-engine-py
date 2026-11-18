# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Which settings a thin client and its engine should agree on, and whether they do.

A thin client (``engine.base_url`` set) and the engine it talks to each load
their own config. Most of what the operator tunes (extraction, ranking, the
nightly maintenance passes, model choice) runs on the engine, so a change made
only in the client's file has no effect. This module classifies every
top-level section of :class:`~particles.config.ParticlesConfig` as *matched*
(the two sides should agree) or *per side* (they differ on purpose), and
reduces the matched settings to per-setting digests that the engine serves
and the client compares.

A *leaf* is a field whose type is not itself a config model: a scalar, a list,
or a dict such as ``llm.providers`` (one leaf, so an entry added on one side
is a difference, not a missing leaf). Only a schema difference between two
releases leaves a path on one side only. Store-free and pure, so it is Client
layer.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from particles.config import ParticlesConfig, env_override_fields

#: Sections the engine applies to work it does for its clients. A thin client
#: and its engine should agree on every leaf under them.
MATCHED_SECTIONS: frozenset[str] = frozenset(
    {
        "llm",
        "extraction",
        "extraction_scope",
        "extraction_modality",
        "extraction_stance",
        "extraction_polarity",
        "extraction_validity",
        "extraction_vision",
        "generation",
        "modality_regeneration",
        "validity_backfill",
        "structured_claim",
        "rdf",
        "document_supersession",
        "document_precedence",
        "journal_extractor",
        "web_clipper",
        "migration",
        "trust",
        "reconciliation",
        "content_age_decay",
        "utility",
        "owner_lens",
        "observer_scope",
        "confidence",
        "conformance",
        "deposit_date",
        "subjects",
        "subject_gate",
        "authorities",
        "wikidata",
        "reddit",
        "hackernews",
        "mastodon",
        "github",
        "query",
        "source_passage",
        "contestedness",
        "lint",
        "graph",
        "links_suggest",
        "vocabulary",
        "citation_signal",
        "curation",
        "audit",
        "reindex",
        "consolidation",
        "embeddings",
        "refetch_floors",
    }
)

#: Sections that describe the process or the host it runs on. They differ
#: between a client and its engine on purpose and are never compared.
PER_SIDE_SECTIONS: frozenset[str] = frozenset(
    {
        "storage",
        "host",
        "engine",
        "api",
        "build",
        "daemon",
        "observability",
        "agent_memory",
        "claude_code",
        "mcp",
        "codex",
        "cli",
        "metrics",
        "http",
        "network",
        "rule_sources",
        "local_refresh",
        "local_files",
        "inbox",
        "obsidian",
        "wiki",
        "logseq",
        "notion",
        "exporter_common",
        "import_vault",
        "import_project",
        "benchmark",
        "benchmark_memory",
        "benchmark_rot",
        "benchmark_relevance_floor",
        "benchmark_leakage",
        "benchmark_update_gate",
    }
)

#: Subsections of a per-side section that are matched all the same. The rest of
#: ``mcp`` is the MCP server's own (its tool defaults, recall resource and
#: memory façade), and on a thin client that server runs on the client; the
#: belief-write switch and its caps are enforced by the engine as well.
MATCHED_SUBSECTIONS: frozenset[str] = frozenset({"claude_code.observer_scope", "mcp.write"})

#: Settings inside matched sections that are per side by construction. ``*``
#: matches one path segment; a pattern that reaches inside a leaf (a dict such
#: as ``llm.providers``) removes that part of the leaf's value before the digest.
PER_SIDE_LEAVES: tuple[str, ...] = (
    # A local model is localhost to the client and host.docker.internal to a container.
    "llm.providers.*.base_url",
    # A Keychain profile names the client's own Keychain.
    "consolidation.credential_profile",
    # A path on the host that holds the manifests.
    "graph.manifest_root",
    # A display toggle for the process that loads the encoder.
    "embeddings.progress_bars",
)

#: How much of a client value a doctor line prints.
VALUE_PREVIEW_CHARS = 80


class TuningReport(BaseModel):
    """What ``GET /config/tuning`` serves: the engine's matched settings, as digests."""

    #: The engine's release, so a caller can say why a path is on one side only.
    version: str
    #: Each matched leaf's dotted path and the digest of its canonical JSON.
    digests: dict[str, str]
    #: The matched paths whose value the engine takes from an environment variable.
    env_overrides: list[str] = []
    #: The extraction-budget warnings, evaluated on the engine's config.
    warnings: list[str] = []


def _segments(path: str) -> list[str]:
    return path.split(".")


def _matches(pattern: Sequence[str], path: Sequence[str]) -> bool:
    """Whether ``pattern`` names ``path`` segment by segment (``*`` is any one segment)."""
    return len(pattern) == len(path) and all(
        p in ("*", s) for p, s in zip(pattern, path, strict=True)
    )


def _strip_inside(value: Any, rest: Sequence[str]) -> Any:
    """``value`` with every key ``rest`` names below it removed (a copy)."""
    if not rest or not isinstance(value, dict):
        return value
    head, tail = rest[0], rest[1:]
    out: dict[str, Any] = {}
    for key, item in value.items():
        if head in ("*", key):
            if not tail:
                continue
            out[key] = _strip_inside(item, tail)
        else:
            out[key] = item
    return out


def _is_model(annotation: object) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _walk(model: type[BaseModel], dumped: dict[str, Any], prefix: str) -> Iterator[tuple[str, Any]]:
    for name, info in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        value = dumped.get(name)
        if _is_model(info.annotation) and isinstance(value, dict):
            yield from _walk(info.annotation, value, path)  # type: ignore[arg-type]
        else:
            yield path, value


def _is_matched(path: str) -> bool:
    section = path.split(".", 1)[0]
    if section in MATCHED_SECTIONS:
        return True
    return any(path == sub or path.startswith(f"{sub}.") for sub in MATCHED_SUBSECTIONS)


def tuning_leaves(cfg: ParticlesConfig) -> dict[str, Any]:
    """Every matched leaf of ``cfg``, by dotted path, with the per-side parts removed."""
    excluded = [_segments(p) for p in PER_SIDE_LEAVES]
    leaves: dict[str, Any] = {}
    for path, value in _walk(type(cfg), cfg.model_dump(mode="json"), ""):
        if not _is_matched(path):
            continue
        segments = _segments(path)
        if any(_matches(pattern, segments) for pattern in excluded):
            continue
        for pattern in excluded:
            if len(pattern) > len(segments) and _matches(pattern[: len(segments)], segments):
                value = _strip_inside(value, pattern[len(segments) :])
        leaves[path] = value
    return leaves


def leaf_digest(value: Any) -> str:
    """The SHA-256 digest of ``value``'s canonical JSON."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tuning_digests(cfg: ParticlesConfig) -> dict[str, str]:
    """Each matched leaf of ``cfg`` and its digest."""
    return {path: leaf_digest(value) for path, value in tuning_leaves(cfg).items()}


def env_overridden_tuning(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The matched paths whose value this process takes from an environment variable."""
    env = os.environ if environ is None else environ
    return {path: var for var, path in env_override_fields(env).items() if _is_matched(path)}


def ignored_by(path: str, patterns: Iterable[str]) -> bool:
    """Whether an ``engine.tuning_drift_ignore`` entry names ``path``.

    An entry names the leaf it spells, every leaf under it, and the whole leaf
    it points inside (only the engine's digest of a leaf is known, so part of
    one cannot be compared).
    """
    segments = _segments(path)
    for raw in patterns:
        pattern = _segments(raw.strip())
        n = min(len(pattern), len(segments))
        if _matches(pattern[:n], segments[:n]):
            return True
    return False


@dataclass
class TuningComparison:
    """How a client's matched settings compare with the engine's digests."""

    #: Leaves present on both sides that agree.
    matching: int = 0
    #: Leaves present on both sides that differ, each with the client's value.
    differing: dict[str, Any] = field(default_factory=dict)
    #: Differing leaves an ``engine.tuning_drift_ignore`` entry declares.
    declared: list[str] = field(default_factory=list)
    #: Leaves present on one side only (the two releases' schemas differ).
    one_sided: int = 0


def compare_tuning(
    cfg: ParticlesConfig, engine_digests: dict[str, str], ignore: Iterable[str]
) -> TuningComparison:
    """Compare ``cfg``'s matched leaves with the engine's ``engine_digests``."""
    patterns = list(ignore)
    leaves = tuning_leaves(cfg)
    result = TuningComparison(one_sided=len(leaves.keys() ^ engine_digests.keys()))
    for path in sorted(leaves.keys() & engine_digests.keys()):
        value = leaves[path]
        if leaf_digest(value) == engine_digests[path]:
            result.matching += 1
        elif ignored_by(path, patterns):
            result.declared.append(path)
        else:
            result.differing[path] = value
    return result


def preview(value: Any) -> str:
    """A client value as one doctor line prints it, capped."""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= VALUE_PREVIEW_CHARS:
        return text
    return text[: VALUE_PREVIEW_CHARS - 1] + "…"
