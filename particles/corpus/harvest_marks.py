# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The marks only the owner's own harvesters put on a corpus entry.

The Claude Code hooks, ``particles audit``, ``rules sync`` and ``init
claude-code`` stamp their entries with a depositor id and a tag set that other
code reads as provenance: the update rung treats two local memory files as one
lineage, the observer scope reads the harness tag, and the agent-memory region
reads ``rule-file``. A mark any intake could write would let a caller join
those classes, so every other intake refuses or strips them (security review
2026-10-24-2, P5):

* the HTTP deposit routes refuse them unless the request carries a real
  bearer, which makes the caller the operator;
* ``particles deposit`` and the agent ``deposit_text`` refuse them;
* the web-clipper intake drops a reserved tag a capture's frontmatter names.

The lineage guard keys on the depositor (:func:`is_owner_harvest`), never on a
tag alone.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlparse

from particles.core.schema import CorpusEntry


@dataclass(frozen=True)
class HarnessSource:
    """One local harness's reserved source identity; never a publisher authority."""

    scheme: str
    tag: str
    depositor: str

    @property
    def session_prefix(self) -> str:
        """Canonical prefix shared by capture writers and Engine readers."""
        return f"{self.scheme}://session/"


HARNESS_SOURCES = MappingProxyType(
    {
        "claude-code": HarnessSource("claude-code", "claude-code", "claude-code-hook"),
        "codex": HarnessSource("codex", "codex", "codex-hook"),
    }
)
HARNESS_URI_SCHEMES = frozenset(source.scheme for source in HARNESS_SOURCES.values())


def harness_session_source(uri: str | None) -> HarnessSource | None:
    """Recognize a session namespace without treating its netloc as a web host."""
    return next(
        (
            source
            for source in HARNESS_SOURCES.values()
            if (uri or "").startswith(source.session_prefix)
        ),
        None,
    )


def is_harness_uri(uri: str | None) -> bool:
    """All URIs in a reserved harness scheme have no publisher authority."""
    return urlparse(uri or "").scheme in HARNESS_URI_SCHEMES


#: The tag the memory-file harvest stamps on every memory-file entry.
MEMORY_FILE_TAG = "memory-file"

#: The tag the harvest adds to the fold-and-archive file's entry.
MEMORY_ARCHIVE_TAG = "memory-archive"

#: The tag the rule-file intake puts on every rule document;
#: ``rule_sources.RULE_SOURCE_TAG`` is this value.
RULE_FILE_TAG = "rule-file"

#: Tags only a harvester stamps: the harness tag, the memory-file and archive
#: harvests, and the rule-file intake.
RESERVED_TAGS = frozenset(
    {MEMORY_FILE_TAG, MEMORY_ARCHIVE_TAG, RULE_FILE_TAG}
    | {source.tag for source in HARNESS_SOURCES.values()}
)

#: The tags whose entries the local refresh masks when it re-reads them: the
#: rule-file intake's and the memory-file harvest's (security review P48).
REFRESH_MASKED_TAGS = frozenset({"rule-file", "memory-file"})

#: The depositors whose memory-file entries the update rung reads as the
#: owner's lineage: the SessionEnd harvest and the first-run memory audit.
HARVEST_DEPOSITORS = frozenset({HARNESS_SOURCES["claude-code"].depositor, "audit"})

#: Depositor ids no intake but a harvester may claim: the two above plus the
#: rule-file intakes, whose entries the local refresh tier trusts
#: (:data:`particles.corpus.fetch.LOCAL_LAZY_DEPOSITORS`).
RESERVED_DEPOSITORS = (
    HARVEST_DEPOSITORS
    | {"rule-sync", "init"}
    | {source.depositor for source in HARNESS_SOURCES.values()}
)

#: The tags that mark a memory-file harvest (the update rung's carve-out).
_MEMORY_HARVEST_TAGS = frozenset({"memory-file", HARNESS_SOURCES["claude-code"].tag})


def reserved_marks(tags: Iterable[str] | None, deposited_by: str | None = None) -> list[str]:
    """The reserved tags and depositor id a deposit names, sorted; empty when none."""
    found = sorted(set(tags or []) & RESERVED_TAGS)
    if deposited_by is not None and deposited_by in RESERVED_DEPOSITORS:
        found.append(f"deposited_by={deposited_by}")
    return found


def reserved_marks_refusal(marks: list[str]) -> str | None:
    """The operator-facing refusal for a deposit naming ``marks``, or ``None`` when empty."""
    if not marks:
        return None
    return (
        f"refusing a deposit that names {', '.join(marks)}: only the owner's harvesters "
        "(the Claude Code/Codex hooks, `particles audit`, `rules sync`, `init claude-code`) "
        "write those marks. Drop them, or use a different tag or --deposited-by."
    )


def strip_reserved_tags(tags: Iterable[str]) -> list[str]:
    """``tags`` without the reserved ones, order kept."""
    return [tag for tag in tags if tag not in RESERVED_TAGS]


def is_owner_harvest(entry: CorpusEntry) -> bool:
    """Whether ``entry`` is a memory-file harvest by one of the owner's harvesters.

    Both halves are needed: the depositor says which process wrote the entry,
    and the tag says it is a memory file rather than, say, a rule file the same
    process enrolled.
    """
    return entry.deposited_by in HARVEST_DEPOSITORS and bool(
        set(entry.tags or []) & _MEMORY_HARVEST_TAGS
    )


def claude_project_slug(path: Path | str) -> str:
    """The ``~/.claude/projects/`` directory name Claude Code derives from a path.

    Every character outside ``[A-Za-z0-9]`` becomes ``-``. The platform does
    not document the rule; ``resolve_session_project`` therefore never trusts
    it blindly: it only re-keys a session whose transcript directory this
    function reproduces. It lives here, beside the harness registry, so the
    engine's route can check a pushed rule file's project against its path the
    way the client computed it.
    """
    return re.sub(r"[^A-Za-z0-9]", "-", str(path))
