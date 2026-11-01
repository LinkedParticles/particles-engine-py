# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""§7.5 Lazy re-fetch protocol for LAZY corpus entries.

Three-tier change detection, over two transports (local tier):

  Tier 1 — Source signal, no body:
             HTTP  — ETag / If-Modified-Since → 304 → REVISIT snapshot
             local — ``stat`` mtime vs the prior snapshot's ``last_modified``
                     → equal → no I/O, no row
  Tier 2 — Content hash: read + SHA-256 compare → REVISIT if unchanged,
           RESPONSE if changed
  Tier 3 — Manual override (``force=True``; ``deposit_url`` / ``corpus refresh``)

:func:`maybe_refetch` dispatches on the URI-R scheme so both transports share
one ladder — the policy gate, the per-source-type floor, the force override and
the RESPONSE/REVISIT distinction have a single home, and the future
backoff and decay have one cadence to tune rather than two.

An HTTP entry whose URI-R an importer claims (a Reddit permalink, a Hacker News
item page) is fetched by that importer's ``refetch``, not by a GET of the URI-R,
because the importer stored bytes from an API rather than the page the URI-R
names. That route has no tier 1: the importer returns a body and
tier 2 decides.

The local tier has **no network**: no DNS, no redirects, no ``particles_client``
and no SSRF guard, because that entire surface is absent rather than bypassed.
What it has instead is a root allowlist (:func:`is_refresh_allowed`): the tier
reads only files under the roots its legitimate depositors write from, because
an entry's ``file://`` URI-R is not proof the operator chose that file
(security review D2).
"""

from __future__ import annotations

import errno
import json
import logging
import os
import stat as stat_module
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import LoadedConfigFile, get_config, is_broad_root, loaded_config_file
from particles.core.schema import (
    ExtractionStatus,
    FetchPolicy,
    Snapshot,
    WarcRecordType,
)
from particles.corpus.deposit import save_blob, sha256
from particles.corpus.harvest_marks import REFRESH_MASKED_TAGS
from particles.corpus.local_read import LocalReadRefused, read_local_file
from particles.corpus.operator_only import (
    WRITABLE_BY_OTHERS,
    Exposure,
    ExposureKind,
    NodeFacts,
    gather_write_facts,
    judge_write_facts,
    writable_only_by_operator,
)
from particles.corpus.redaction import redact_secrets
from particles.corpus.rule_roots import effective_uid, list_rule_source_roots, owned_by_refresher
from particles.corpus.rule_sources import expand_path
from particles.corpus.store import (
    CorpusEntryRow,
    SnapshotRow,
    list_refreshable_local_entries,
    list_snapshots_for_entry,
)
from particles.corpus.worktree import verified_main_checkout
from particles.http import get_capped, particles_client

log = logging.getLogger(__name__)


class SourceImporter(Protocol):
    """The part of an importer the refetch ladder reads.

    Every ``ImporterPlugin`` satisfies it. The ladder takes importers as an
    argument rather than importing the registry, so ``corpus`` gains no edge
    into ``ingest``; the refresh operation passes ``get_importers()``.
    An importer that can re-fetch also defines ``refetch(uri_r) -> bytes``,
    found with ``getattr`` the way ``primary_url`` is.
    """

    def accepts_url(self, url: str) -> bool:
        """Return True if this importer handles the given URL."""
        ...


class RefetchUnsupported(RuntimeError):
    """An importer owns the entry's URI-R but cannot re-fetch it.

    Raised instead of a plain GET of the URI-R, which would store a page the
    importer's extractor cannot read.
    """


def _floor_seconds(source_type: str) -> int:
    """The per-source-type re-fetch floor, in seconds (``corpus.refetch_floors``).

    Read from config at call time per the AGENTS.md configuration rule. This
    function used to consult a private dict that shadowed the config section —
    a discrepancy nothing noticed because, until, no production code
    called into this module at all.
    """
    return get_config().refetch_floors.get(source_type, 3600)


def path_from_file_uri(uri_r: str) -> Path:
    """The local path a ``file://`` URI-R names.

    ``deposit_file`` records ``path.resolve().as_uri()``, so the round trip is
    exact for anything this SDK deposited. Percent-escapes (spaces, non-ASCII)
    are decoded; a ``file://host/...`` authority other than ``localhost`` is
    rejected rather than silently reinterpreted as a local path.
    """
    parsed = urlparse(uri_r)
    if parsed.netloc not in ("", "localhost"):
        raise ValueError(f"Refusing non-local file URI with authority {parsed.netloc!r}: {uri_r}")
    return Path(unquote(parsed.path))


#: The depositors that enrol a ``LAZY`` ``file://`` entry from the engine's own
#: disk: ``rules sync`` (``rule-sync``), ``init claude-code`` (``init``), the
#: SessionEnd harvest (``claude-code-hook``) and ``particles deposit FILE``
#: (``operator``). A derived root admits only their entries (security review
#: N21): before 1.210.4 the HTTP route could enrol any engine path as ``LAZY``
#: (its default identity is ``api-user``), and nothing else tells such an entry
#: from a local one. A new local depositor of ``LAZY`` entries joins this set.
LOCAL_LAZY_DEPOSITORS = frozenset({"rule-sync", "init", "claude-code-hook", "operator"})

#: The depositors the ``~/.claude/projects`` memory-file root admits: the set
#: above plus ``particles audit`` (``audit``). The audit deposits the memory
#: files it reads ``NEVER``; the SessionEnd harvest later enrols the same entry
#: ``LAZY`` and the entry keeps its first depositor. ``audit`` is scoped to this
#: one root rather than added above because ``deposited_by`` is a free string
#: over HTTP: so scoped, a forged ``audit`` reaches nothing a forged
#: ``claude-code-hook`` does not.
MEMORY_FILE_DEPOSITORS = LOCAL_LAZY_DEPOSITORS | {"audit"}


class RefreshRefusal(StrEnum):
    """Why the local tier refuses a ``file://`` entry.

    Each cause has its own fix, so every report names which one it counts.
    """

    #: The path is under no allowed root. ``rules sync`` from the project
    #: that holds it records its root.
    OUTSIDE_ROOTS = "outside_roots"
    #: The path is under a derived root, but the entry's first depositor is
    #: not one the local tier trusts there (security review N21). Re-enrolling
    #: keeps the first depositor, so only ``local_refresh.allowed_roots``,
    #: which admits any depositor, changes the answer.
    UNTRUSTED_DEPOSITOR = "untrusted_depositor"
    #: The store resolves against the working directory, lies inside a git
    #: work tree, or is another user's file, so it may be one a cloned
    #: repository shipped, and its entries choose no host file (
    #: security review 2026-10-26-1, W15). Naming a store outside any
    #: repository with an absolute path is the fix.
    CWD_STORE = "cwd_store"
    #: The store file is the operator's, but its group or other users may
    #: write it, so rows another user wrote could choose the host files the
    #: refresh reads (security review 2026-10-27-1, G16). Removing group and
    #: other write permission from the store file is the fix.
    STORE_WRITABLE_BY_OTHERS = "store_writable_by_others"
    #: The store passes, but the config file this process loaded is one another
    #: user may write, or sits in a directory they may write into, and it
    #: chooses ``local_refresh.allowed_roots``, which admits any file under it.
    #: Removing their write access, or moving the config to a
    #: directory only the operator may write, is the fix.
    CONFIG_WRITABLE_BY_OTHERS = "config_writable_by_others"
    #: The path is under a recorded root, or the main checkout derived from
    #: one, whose directory another user owns. Naming the
    #: directory in ``local_refresh.allowed_roots`` trusts it; otherwise
    #: ``rules forget`` the root.
    ROOT_NOT_OWNED = "root_not_owned"
    #: The file lies under a recorded root, or the main checkout derived from
    #: one, but another user owns the file itself. The read
    #: refuses it on the descriptor; this cause lets the counts say so
    #: (security review 2026-10-25-2, R42).
    FILE_NOT_OWNED = "file_not_owned"
    #: The file has more than one link, which a contained read refuses since
    #: no path can say where its other names lie (security review K10, P18).
    HARD_LINKED = "hard_linked"
    #: The file lies under a recorded root, or the main checkout derived from
    #: one, and its group or other users may write it, by its mode or an ACL
    #: entry, or may write into a directory that holds it, so its owner is not
    #: who wrote its bytes (security review 2026-10-26-1, W23; 2026-10-27-1,
    #: G20).
    WRITABLE_BY_OTHERS = "writable_by_others"
    #: The name is a symlink and ``local_refresh.follow_symlinks`` is off, so
    #: the read refuses it (security review 2026-10-26-1, W66).
    SYMLINKED = "symlinked"
    #: The name is not a regular file, or the file is over
    #: ``local_files.max_bytes``, so the read refuses it (R42).
    UNREADABLE = "unreadable"

    @property
    def note(self) -> str:
        """The cause as a short clause, for a per-entry line."""
        if self is RefreshRefusal.OUTSIDE_ROOTS:
            return "outside the allowed roots"
        if self is RefreshRefusal.CWD_STORE:
            return "the store may be one a repository shipped"
        if self is RefreshRefusal.STORE_WRITABLE_BY_OTHERS:
            return "other users can write the store"
        if self is RefreshRefusal.CONFIG_WRITABLE_BY_OTHERS:
            return "other users can write the config file"
        if self is RefreshRefusal.ROOT_NOT_OWNED:
            return "its recorded root is owned by another user"
        if self is RefreshRefusal.FILE_NOT_OWNED:
            return "the file is owned by another user"
        if self is RefreshRefusal.HARD_LINKED:
            return "the file is hard-linked"
        if self is RefreshRefusal.WRITABLE_BY_OTHERS:
            return "other users can write the file"
        if self is RefreshRefusal.SYMLINKED:
            return "the file is a symlink"
        if self is RefreshRefusal.UNREADABLE:
            return "the file is not a regular file within the size cap"
        return "enrolled by an untrusted depositor"


@dataclass(frozen=True)
class RefusalCounts:
    """How many ``LAZY`` ``file://`` entries the local tier refuses, by cause."""

    outside_roots: int = 0
    untrusted_depositor: int = 0
    cwd_store: int = 0
    store_writable_by_others: int = 0
    config_writable_by_others: int = 0
    root_not_owned: int = 0
    file_not_owned: int = 0
    hard_linked: int = 0
    writable_by_others: int = 0
    symlinked: int = 0
    unreadable: int = 0

    @classmethod
    def tally(cls, reasons: Iterable[RefreshRefusal]) -> RefusalCounts:
        """Count a stream of refusals by cause."""
        counts = Counter(reasons)
        return cls(
            outside_roots=counts[RefreshRefusal.OUTSIDE_ROOTS],
            untrusted_depositor=counts[RefreshRefusal.UNTRUSTED_DEPOSITOR],
            cwd_store=counts[RefreshRefusal.CWD_STORE],
            store_writable_by_others=counts[RefreshRefusal.STORE_WRITABLE_BY_OTHERS],
            config_writable_by_others=counts[RefreshRefusal.CONFIG_WRITABLE_BY_OTHERS],
            root_not_owned=counts[RefreshRefusal.ROOT_NOT_OWNED],
            file_not_owned=counts[RefreshRefusal.FILE_NOT_OWNED],
            hard_linked=counts[RefreshRefusal.HARD_LINKED],
            writable_by_others=counts[RefreshRefusal.WRITABLE_BY_OTHERS],
            symlinked=counts[RefreshRefusal.SYMLINKED],
            unreadable=counts[RefreshRefusal.UNREADABLE],
        )

    @property
    def total(self) -> int:
        """Every refused entry, whatever the cause."""
        return (
            self.outside_roots
            + self.untrusted_depositor
            + self.cwd_store
            + self.store_writable_by_others
            + self.config_writable_by_others
            + self.root_not_owned
            + self.file_not_owned
            + self.hard_linked
            + self.writable_by_others
            + self.symlinked
            + self.unreadable
        )

    def lines(self) -> list[str]:
        """One sentence per nonzero cause: the count and the fix that matches it.

        ``hook doctor``, the bare ``particles rules`` verb and the refresh
        loop's warning all print these, so the advice reads the same
        everywhere.
        """
        out: list[str] = []
        if self.cwd_store:
            out.append(
                f"{self.cwd_store} LAZY local source(s) are not refreshed because this store "
                "resolves against the working directory, lies inside a git work tree, or is "
                "another user's file, so it may be one a cloned repository shipped. Name a "
                "store of your own outside any repository with an absolute DATABASE_URL to "
                "refresh it."
            )
        if self.store_writable_by_others:
            out.append(
                f"{self.store_writable_by_others} LAZY local source(s) are not refreshed "
                "because other users can write this store file, one of its SQLite sidecars "
                "(-wal, -shm, -journal) or its directory, so its rows may not be yours. "
                "Remove group and other write permission from each (chmod go-w), or move "
                "the store to a directory only you may write, to refresh them."
            )
        if self.config_writable_by_others:
            out.append(
                f"{self.config_writable_by_others} LAZY local source(s) are not refreshed "
                "because other users can write the config file this process loaded, or the "
                "directory holding it, and it chooses the roots the refresh reads. Remove "
                "group and other write permission from it (chmod go-w), or move it to a "
                "directory only you may write, to refresh them."
            )
        if self.outside_roots:
            out.append(
                f"{self.outside_roots} LAZY local source(s) lie outside every allowed "
                "root and are not refreshed. Run `particles rules sync` from the "
                "project that holds them, or add their directory to "
                "local_refresh.allowed_roots."
            )
        if self.untrusted_depositor:
            out.append(
                f"{self.untrusted_depositor} LAZY local source(s) were enrolled by a "
                "depositor the local refresh does not trust and are not refreshed. "
                "Re-enrolling keeps the first depositor, so `rules sync` does not "
                "change this; if the files are yours, add their directory to "
                "local_refresh.allowed_roots."
            )
        if self.root_not_owned:
            out.append(
                f"{self.root_not_owned} LAZY local source(s) lie under a recorded root "
                "another user owns and are not refreshed. If you trust that directory, "
                "add it to local_refresh.allowed_roots; otherwise run "
                "`particles rules forget <root>`."
            )
        if self.file_not_owned:
            out.append(
                f"{self.file_not_owned} LAZY local source(s) under a recorded root are "
                "owned by another user and are not refreshed. If you trust that file, add "
                "its directory to local_refresh.allowed_roots; otherwise remove it."
            )
        if self.hard_linked:
            out.append(
                f"{self.hard_linked} LAZY local source(s) are hard-linked and are not "
                "refreshed, since a hard link can name a file outside every root. Replace "
                "each with a copy of the file it links to."
            )
        if self.writable_by_others:
            out.append(
                f"{self.writable_by_others} LAZY local source(s) under a recorded root can "
                "be written by other users, or lie in a directory they can write into, and "
                "are not refreshed. Remove group and other write permission (chmod go-w) and "
                "any ACL entry granting write from each file and the directories above it."
            )
        if self.symlinked:
            out.append(
                f"{self.symlinked} LAZY local source(s) are symlinks and are not refreshed "
                "while local_refresh.follow_symlinks is off. Enroll the file the link "
                "names instead."
            )
        if self.unreadable:
            out.append(
                f"{self.unreadable} LAZY local source(s) are not regular files or are over "
                "local_files.max_bytes and are not refreshed."
            )
        return out


def _rule_file_root_sets(
    recorded_roots: Sequence[Path], *, check_owner: bool = True
) -> tuple[list[Path], list[Path]]:
    """The rule-file roots in two sets: those an operator named, and those derived.

    Named: ``rule_sources.paths`` and the recorded roots that are existing
    directories. Derived: the main checkout of each recorded
    worktree root, read from the root's own ``.git`` file and verified against
    the links git writes (:func:`~particles.corpus.worktree.verified_main_checkout`;
    security review K6, P12, P17), and ``~/.claude``. Only a
    ``rule_sources.paths`` entry may be a file admitted by equality
    (:func:`refresh_refusal`); every other root admits rule-named files under
    it and nothing else.
    A broad root (:func:`~particles.config.is_broad_root`: the
    filesystem root, the home directory, or an ancestor of either) admits
    nothing, whether named, recorded or derived (security review K9, P49).
    A recorded root, and the main checkout derived from one, counts only while
    the refreshing user owns its directory; ``check_owner``
    off drops that rule, which :func:`refresh_refusal` uses to tell a
    foreign-owned root from no root at all.
    """
    recorded = [
        r
        for r in recorded_directory_roots(recorded_roots, check_owner=check_owner)
        if not is_broad_root(r)
    ]
    named = configured_rule_roots() + recorded
    derived = [
        main
        for r in recorded
        if (main := verified_main_checkout(r)) is not None
        and not is_broad_root(main)
        and (not check_owner or owned_by_refresher(main))
    ]
    derived.append((Path.home() / ".claude").resolve())
    return named, derived


def rule_file_roots(recorded_roots: Sequence[Path]) -> list[Path]:
    """The roots under which the local tier admits rule files.

    ``rule_sources.paths``, the roots ``rules sync`` and ``init claude-code``
    recorded, the main checkout of each recorded worktree root (read from the
    root's own ``.git`` file and verified against the links git writes,
    :func:`~particles.corpus.worktree.verified_main_checkout`; security
    review K6), and ``~/.claude``. The refreshing process's
    working directory is never consulted, so a refresh admits the same files
    whichever directory runs it (security review N38). A broad root is
    dropped (:func:`_rule_file_root_sets`).
    """
    named, derived = _rule_file_root_sets(recorded_roots)
    return named + derived


def configured_rule_roots() -> list[Path]:
    """``rule_sources.paths``, the roots the operator's own config names.

    The only roots that admit a file equal to the root whatever its name.
    A broad one admits nothing (security review K9, P49).
    """
    return [
        r
        for p in get_config().rule_sources.paths
        if (r := expand_path(p)) is not None and not is_broad_root(r)
    ]


def recorded_directory_roots(
    recorded_roots: Sequence[Path], *, check_owner: bool = True
) -> list[Path]:
    """The recorded roots that are existing directories the refreshing user owns.

    A recorded file, or a path that no longer exists, admits nothing: a row
    names the place an operator enrolled, and a store carries its own rows
    (security review 2026-10-24-2, P2). A directory another user
    owns admits nothing either, since it may be one they created at the
    recorded path after the original went away.
    ``check_owner`` off keeps it.
    """
    out: list[Path] = []
    for raw in recorded_roots:
        root = Path(raw)
        try:
            if root.is_dir() and (not check_owner or owned_by_refresher(root)):
                out.append(root)
        except OSError:
            continue
    return out


def is_refresh_allowed(
    path: Path, *, recorded_roots: Sequence[Path] = (), deposited_by: str | None = None
) -> bool:
    """Whether the local tier may read ``path`` (security review D2).

    :func:`refresh_refusal` is ``None``; see it for the roots and the
    depositor rule.
    """
    return refresh_refusal(path, recorded_roots=recorded_roots, deposited_by=deposited_by) is None


def allowed_refresh_roots() -> list[tuple[str, Path]]:
    """``local_refresh.allowed_roots`` as ``(as written, expanded)`` pairs, broad ones dropped.

    A broad entry (:func:`~particles.config.is_broad_root`: the filesystem
    root, the home directory or an ancestor of either) admits nothing, as a
    broad rule-file root admits nothing (security review 2026-10-26-1, W64).
    ``particles config validate`` names each one it drops
    (:func:`broad_allowed_roots`).
    """
    pairs = [(raw, expand_path(raw)) for raw in get_config().local_refresh.allowed_roots]
    return [(raw, root) for raw, root in pairs if root is not None and not is_broad_root(root)]


def broad_allowed_roots() -> list[str]:
    """The ``local_refresh.allowed_roots`` entries that are broad and admit nothing."""
    return [
        raw
        for raw in get_config().local_refresh.allowed_roots
        if (root := expand_path(raw)) is not None and is_broad_root(root)
    ]


def refresh_refusal(
    path: Path, *, recorded_roots: Sequence[Path] = (), deposited_by: str | None = None
) -> RefreshRefusal | None:
    """Why the local tier may not read ``path``, or ``None`` when it may (security review D2).

    The resolved path must sit under one of three kinds of root, each the
    reach of one legitimate depositor of a ``LAZY`` ``file://`` entry:

    * ``local_refresh.allowed_roots`` admits any file under it: the operator's
      own ``particles deposit FILE --fetch-policy LAZY`` opt-ins. A broad
      entry admits nothing (:func:`allowed_refresh_roots`).
    * The rule-file roots (:func:`rule_file_roots`) admit files named in
      ``rule_sources.filenames`` beneath them (and a ``rule_sources.paths``
      entry that is itself a file, whatever its name; a recorded or derived
      root never admits by equality): what ``rules sync`` and
      ``init claude-code`` enrol. ``recorded_roots`` is
      what those verbs recorded (:func:`list_rule_source_roots`); a caller
      with a session passes it, and none means only the configured and
      user-level roots.
    * ``~/.claude/projects`` admits ``*.md``: the Claude Code memory files the
      SessionEnd harvest enrols.

    Every comparison is on resolved paths, so ``..`` and a symlinked component
    cannot step out of a root.

    ``deposited_by``, when given, is the entry's recorded depositor. The two
    derived roots then admit it only when it is one of
    :data:`LOCAL_LAZY_DEPOSITORS` (the memory-file root: one of
    :data:`MEMORY_FILE_DEPOSITORS`), and a path they hold under another
    depositor is :attr:`RefreshRefusal.UNTRUSTED_DEPOSITOR`.
    ``local_refresh.allowed_roots`` admits any depositor, since the operator
    named that directory (security review N21). A path only a recorded root
    another user owns would hold (or the main checkout derived from one) is
    :attr:`RefreshRefusal.ROOT_NOT_OWNED`. A path no root holds
    is :attr:`RefreshRefusal.OUTSIDE_ROOTS`.
    """
    cfg = get_config()
    # ``os.path.realpath`` rather than ``Path.resolve()``: on Python 3.11
    # ``resolve()`` also stats the result, and a refused entry is never
    # touched. Both walk the same symlinks.
    resolved = Path(os.path.realpath(path))
    if any(resolved.is_relative_to(root) for _raw, root in allowed_refresh_roots()):
        return None

    def _trusted(depositors: frozenset[str]) -> RefreshRefusal | None:
        if deposited_by is None or deposited_by in depositors:
            return None
        return RefreshRefusal.UNTRUSTED_DEPOSITOR

    projects = (Path.home() / ".claude" / "projects").resolve()
    if resolved.suffix == ".md" and resolved.is_relative_to(projects):
        return _trusted(MEMORY_FILE_DEPOSITORS)

    filenames = set(cfg.rule_sources.filenames)

    def _held(check_owner: bool) -> bool:
        named, derived = _rule_file_root_sets(recorded_roots, check_owner=check_owner)
        return resolved.name in filenames and any(
            resolved.is_relative_to(root) for root in (*named, *derived)
        )

    # A root that is itself a file admits that file only when the operator's
    # config names it; a recorded or derived root admits rule files beneath it
    # and never a file by being equal to it.
    if any(resolved == root for root in configured_rule_roots()) or _held(check_owner=True):
        return _trusted(LOCAL_LAZY_DEPOSITORS)
    if _held(check_owner=False):
        return RefreshRefusal.ROOT_NOT_OWNED
    return RefreshRefusal.OUTSIDE_ROOTS


class AdmissionKind(StrEnum):
    """Which kind of route admits a path to the local tier (:func:`admission`)."""

    ALLOWED_ROOT = "allowed_root"
    MEMORY_FILE = "memory_file"
    CONFIGURED = "configured"
    USER_CLAUDE = "user_claude"
    RECORDED = "recorded"
    DERIVED = "derived"
    OTHER = "other"

    @property
    def needs_owner(self) -> bool:
        """Whether a file read through this route must be the refreshing user's.

        The routes a verb recorded, not the ones the operator names in config.
        """
        return self in (AdmissionKind.RECORDED, AdmissionKind.DERIVED)


def admission(
    path: Path, *, recorded_roots: Sequence[Path] = (), deposited_by: str | None = None
) -> tuple[AdmissionKind, str] | None:
    """The route that admits ``path`` to the local tier and its name, or ``None`` when refused.

    The routes :func:`refresh_refusal` checks, in the same order. The name is a
    short phrase for an operator: a config key, the memory-file rule,
    ``~/.claude``, or the root itself.
    """
    if refresh_refusal(path, recorded_roots=recorded_roots, deposited_by=deposited_by) is not None:
        return None
    cfg = get_config()
    resolved = Path(os.path.realpath(path))
    for raw, root in allowed_refresh_roots():
        if resolved.is_relative_to(root):
            return AdmissionKind.ALLOWED_ROOT, f"local_refresh.allowed_roots ({raw})"
    projects = (Path.home() / ".claude" / "projects").resolve()
    if resolved.suffix == ".md" and resolved.is_relative_to(projects):
        return AdmissionKind.MEMORY_FILE, "the ~/.claude/projects memory-file rule"
    configured = [
        r
        for p in cfg.rule_sources.paths
        if (r := expand_path(p)) is not None and not is_broad_root(r)
    ]
    for root in configured:
        if resolved == root or resolved.is_relative_to(root):
            return AdmissionKind.CONFIGURED, f"rule_sources.paths ({root})"
    user_claude = (Path.home() / ".claude").resolve()
    if resolved.is_relative_to(user_claude):
        return AdmissionKind.USER_CLAUDE, "~/.claude"
    named, derived = _rule_file_root_sets(recorded_roots)
    for root in named:
        if resolved == root or resolved.is_relative_to(root):
            return AdmissionKind.RECORDED, f"the recorded root {root}"
    for root in derived:
        if resolved.is_relative_to(root):
            return (
                AdmissionKind.DERIVED,
                f"the main checkout of a recorded worktree ({root})",
            )
    return AdmissionKind.OTHER, "another allowed root"


def admission_route(
    path: Path, *, recorded_roots: Sequence[Path] = (), deposited_by: str | None = None
) -> str | None:
    """Which route admits ``path`` to the local tier, or ``None`` when it is refused.

    The name :func:`admission` gives, so ``particles rules forget`` can say
    which other route still reads a file once a root is removed.
    A route whose read would refuse the file itself (:func:`file_refusal`)
    does not admit it.
    """
    found = admission(path, recorded_roots=recorded_roots, deposited_by=deposited_by)
    if found is None or file_refusal(path, found[0]) is not None:
        return None
    return found[1]


def file_refusal(path: Path, kind: AdmissionKind) -> RefreshRefusal | None:
    """What the read's descriptor checks would refuse in the file ``path`` names, or ``None``.

    The checks :func:`read_local_file` makes on the refresh read: a symlink
    while ``local_refresh.follow_symlinks`` is off, a non-regular file or one
    over ``local_files.max_bytes``, the owner and others-may-write checks a
    ``kind`` that :attr:`~AdmissionKind.needs_owner` makes, and
    the hard-link check every contained read makes (security review K10,
    P18). A ``stat`` of the name, so the counts ``particles rules``, ``hook
    doctor`` and ``rules forget`` print agree with what the refresh reads
    (security review 2026-10-25-2, R42; 2026-10-26-1, W23, W66); the read
    itself judges its own descriptor. A file that cannot be stat'ed is not
    refused here: the ladder reports it missing.
    """
    cfg = get_config()
    try:
        st = os.stat(path) if cfg.local_refresh.follow_symlinks else os.lstat(path)
    except OSError:
        return None
    if stat_module.S_ISLNK(st.st_mode):
        return RefreshRefusal.SYMLINKED
    if not stat_module.S_ISREG(st.st_mode) or st.st_size > cfg.local_files.max_bytes:
        return RefreshRefusal.UNREADABLE
    uid = effective_uid()
    if uid is None:
        # The read refuses every route on such a host, the ones that need no
        # owner included, so the counts do too (2026-10-30-1, AB59).
        return RefreshRefusal.UNREADABLE
    if kind.needs_owner:
        # The name as enrolled, so the directories its symlinks live in are
        # judged too.
        exposure = writable_only_by_operator(path, named_ahead=False, operator_uid=uid)
        if exposure is not None:
            return rule_file_refusal(exposure)
    if st.st_nlink > 1:
        return RefreshRefusal.HARD_LINKED
    return None


def rule_file_refusal(exposure: Exposure) -> RefreshRefusal:
    """The refusal cause a rule file under a recorded or derived root gets."""
    return {
        ExposureKind.FILE_NOT_OWNED: RefreshRefusal.FILE_NOT_OWNED,
        ExposureKind.FILE_WRITABLE: RefreshRefusal.WRITABLE_BY_OTHERS,
        ExposureKind.FILE_UNJUDGEABLE: RefreshRefusal.UNREADABLE,
        ExposureKind.PLACE_OPEN: RefreshRefusal.WRITABLE_BY_OTHERS,
    }[exposure.kind]


def dir_open_to_others(real: Path) -> Path | None:
    """The first directory holding ``real`` that other users may write into, or ``None``.

    The place half of the one predicate (part 2) for a file read
    once it exists, so the sticky bit closes each directory: the directory
    holding ``real`` and each one above it up to the first broad root. A
    directory another user than the refresher and root owns counts as open, and
    so does one that cannot be opened and judged.
    """
    facts = gather_write_facts(real)
    facts = replace(facts, file=NodeFacts(real, exists=False, opened=False), sidecars=())
    exposure = judge_write_facts(facts, operator_uid=effective_uid(), named_ahead=False)
    return None if exposure is None else exposure.at


def local_refusal(
    path: Path, *, recorded_roots: Sequence[Path] = (), deposited_by: str | None = None
) -> RefreshRefusal | None:
    """Why the refresh would not read ``path``: its roots, then the file itself.

    :func:`refresh_refusal`, and when that admits the path,
    :func:`file_refusal` on the route :func:`admission` names.
    """
    reason = refresh_refusal(path, recorded_roots=recorded_roots, deposited_by=deposited_by)
    if reason is not None:
        return reason
    found = admission(path, recorded_roots=recorded_roots, deposited_by=deposited_by)
    return None if found is None else file_refusal(path, found[0])


async def maybe_refetch(
    session: AsyncSession,
    entry_id: str,
    force: bool = False,
    *,
    importers: Sequence[SourceImporter] = (),
) -> Snapshot | None:
    """Check whether entry needs re-fetching; create a new snapshot if content changed.

    Returns the latest snapshot (new RESPONSE/REVISIT or existing) or None if
    the entry has no URI-R, has fetch_policy=NEVER, or names a local file that
    no longer exists.

    Dispatches on the URI-R scheme: a ``file://`` entry takes the
    local tier (``stat`` then hash, no network) and everything else takes the
    HTTP ladder. Within the HTTP ladder, the first of ``importers`` that accepts
    the URI-R fetches it with its ``refetch``; a URI-R no importer accepts is
    fetched with a GET.

    Raises:
        RefetchUnsupported: an importer accepts the URI-R but defines no
            ``refetch``.
    """
    entry_row = await session.get(CorpusEntryRow, entry_id)
    if entry_row is None:
        raise ValueError(f"Entry {entry_id} not found")

    if entry_row.fetch_policy != FetchPolicy.LAZY.value and not force:
        return None  # nothing to do
    if entry_row.uri_r is None:
        return None

    is_local = entry_row.uri_r.startswith("file://")

    snapshots = await list_snapshots_for_entry(session, entry_id)
    if not snapshots:
        # No prior snapshot — capture a fresh one.
        if is_local:
            return await _do_local_refresh(session, entry_row, prior=None)
        return await _do_remote_refresh(session, entry_row, importers, prior=None)

    latest = max(snapshots, key=lambda s: s.captured_at)

    # Floor check
    floor = _floor_seconds(entry_row.source_type)
    now = datetime.now(UTC)
    age_seconds = (
        now - latest.captured_at.replace(tzinfo=UTC)
        if latest.captured_at.tzinfo is None
        else (now - latest.captured_at)
    ).total_seconds()
    if not force and age_seconds < floor:
        log.debug(
            "Entry %s last fetched %.0fs ago (floor %ds); skipping",
            entry_id,
            age_seconds,
            floor,
        )
        return latest

    if is_local:
        return await _do_local_refresh(session, entry_row, prior=latest)
    return await _do_remote_refresh(session, entry_row, importers, prior=latest)


async def _do_remote_refresh(
    session: AsyncSession,
    entry_row: CorpusEntryRow,
    importers: Sequence[SourceImporter],
    prior: Snapshot | None,
) -> Snapshot:
    """Fetch a non-file entry through its owning importer, or by GET when none owns it."""
    uri_r = entry_row.uri_r
    assert uri_r is not None
    owner = next((imp for imp in importers if imp.accepts_url(uri_r)), None)
    if owner is None:
        return await _do_fetch(session, entry_row, prior)
    refetch: Callable[[str], Awaitable[bytes]] | None = getattr(owner, "refetch", None)
    if refetch is None:
        raise RefetchUnsupported(
            f"{uri_r} belongs to the {type(owner).__name__}, which cannot refresh an "
            f"entry, and fetching the URL directly would store a different document. "
            f"Deposit the URL again to capture a new snapshot."
        )

    # The importer fetches the way its deposit does, so an unchanged source
    # hashes to the prior snapshot's content_hash and becomes a REVISIT. Its own
    # client carries the SSRF guard and the egress policy.
    content = await refetch(uri_r)
    return await _record_body(session, entry_row.entry_id, content, prior, datetime.now(UTC))


#: Tags of the entries the Claude Code harness enrols (``rules sync`` and the
#: SessionEnd harvest). Their deposits are masked (:func:`redact_secrets`), so
#: the refresh masks what it re-reads of them too (security review P48).
_HARNESS_SOURCE_TAGS = REFRESH_MASKED_TAGS


def _redact_harness_source(entry_row: CorpusEntryRow, content: bytes) -> bytes | None:
    """The masked bytes of a harness-enrolled source, or ``None`` when nothing was masked.

    A file of another entry is the operator's own deposit, archived as its
    bytes, and is never rewritten.
    """
    if not _HARNESS_SOURCE_TAGS & set(json.loads(entry_row.tags_json or "[]")):
        return None
    text = content.decode("utf-8", errors="replace")
    masked = redact_secrets(text)
    return masked.encode("utf-8") if masked != text else None


async def _do_local_refresh(
    session: AsyncSession,
    entry_row: CorpusEntryRow,
    prior: Snapshot | None,
) -> Snapshot | None:
    """The local tier: ``stat`` mtime, then content hash. No network.

    Returns the resulting snapshot, or ``None`` when the file no longer exists,
    cannot be read, or lies outside the root allowlist (:func:`is_refresh_allowed`;
    a refused path is never opened or stat'ed). A vanished rule file is an
    *outcome*, not an error: it must not crash the nightly cycle, and
    (deliberately) it must not retire its beliefs either. "The file is gone"
    is not "the claims are false", and a stat taken mid-``git``-operation is
    not evidence of deletion. Retiring on absence is decay, not
    this ladder's job.
    """
    uri_r = entry_row.uri_r
    assert uri_r is not None

    try:
        path = path_from_file_uri(uri_r)
    except ValueError as exc:
        log.info("Entry %s: local source unavailable (%s)", entry_row.entry_id[:8], exc)
        return None
    if effective_uid() is None:
        # Nothing on this host says whose a file is, so no route reads one,
        # the ones that need no owner included.
        log.warning(
            "Entry %s: this host reports no effective uid, so no host file is read.",
            entry_row.entry_id[:8],
        )
        return None
    if (store := cwd_relative_store(session)) is not None:
        # The refresh loop refuses these first, with the count; this keeps the
        # tier itself closed for any other caller.
        log.info(
            "Entry %s: the store %s may not be the operator's own; not read.",
            entry_row.entry_id[:8],
            store,
        )
        return None
    if (config := config_file_judgement()) is not None:
        # The config chooses the roots, so one others may write chooses no
        # host file either.
        log.info(
            "Entry %s: the config file %s may not be the operator's own; not read.",
            entry_row.entry_id[:8],
            config[0],
        )
        return None
    depositor = entry_row.deposited_by
    recorded = await list_rule_source_roots(session)

    def _allowed(candidate: Path) -> bool:
        return is_refresh_allowed(candidate, recorded_roots=recorded, deposited_by=depositor)

    if not _allowed(path):
        # One warning per run, with the count, comes from the refresh loop
        # (operations/corpus_refresh.py); this line is per entry.
        log.info(
            "Entry %s: %s is outside the local refresh roots for depositor %r; not read.",
            entry_row.entry_id[:8],
            path,
            depositor,
        )
        return None

    cfg = get_config()
    try:
        follow = cfg.local_refresh.follow_symlinks
        if not follow and path.is_symlink():
            raise ValueError(f"symlinked source and local_refresh.follow_symlinks is off: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
    except (OSError, ValueError) as exc:
        log.info("Entry %s: local source unavailable (%s)", entry_row.entry_id[:8], exc)
        return None

    mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)

    # Tier 1 — source signal, no read. `last_modified` already means "when the
    # source last changed"; a local mtime is that same quantity over a different
    # transport, which is why this reuses the column rather than adding one.
    #
    # The comparison is INEQUALITY, not recency: a restore from backup or an
    # `rsync -t` can move an mtime backwards while changing content, so "differs"
    # — in either direction — means re-read. Stated plainly: an unchanged mtime
    # is a heuristic, not proof. Where a 304 is the server's assertion, this is
    # an inference; `force=True` is the escape hatch.
    if (
        prior is not None
        and prior.last_modified is not None
        and _as_utc(prior.last_modified) == mtime
    ):
        log.debug("Entry %s: mtime unchanged; no read", entry_row.entry_id[:8])
        return prior

    # The bytes come from one descriptor, and that descriptor's own path is
    # judged again: the check above resolved the name, and a parent swapped for
    # a symlink since then would otherwise be followed here (security review N22).
    # A file read through a recorded root, or the main checkout derived from
    # one, must be the refreshing user's, judged on the same descriptor: a
    # rule file another user wrote beneath the root is refused however the
    # root came to hold it. The route is the one the opened
    # file's own path takes, never the name's: a directory swapped between a
    # lookup on the name and the open would otherwise choose the check
    # (security review 2026-10-25-2, R15).
    def _owner(real: Path) -> int | None:
        route = admission(real, recorded_roots=recorded, deposited_by=depositor)
        if route is not None and not route[0].needs_owner:
            return None
        if (uid := effective_uid()) is None:
            # Nothing says whose the file is, so it is not read.
            raise LocalReadRefused(
                errno.EPERM, "refusing a file whose owner this host cannot report", str(real)
            )
        return uid

    try:
        read = read_local_file(
            path,
            follow_symlinks=follow,
            max_bytes=cfg.local_files.max_bytes,
            allowed=_allowed,
            owner=_owner,
        )
    except LocalReadRefused as exc:
        level = logging.WARNING if exc.errno == errno.EPERM else logging.INFO
        log.log(level, "Entry %s: local source refused (%s)", entry_row.entry_id[:8], exc)
        return None
    except OSError as exc:
        log.info("Entry %s: local source unreadable (%s)", entry_row.entry_id[:8], exc)
        return None
    # The read judged the directories above the file too, on the same
    # descriptor: a directory others may write into lets them rename an
    # owner-written file into a rule file's (G20).
    content = read.content
    mtime = datetime.fromtimestamp(read.stat.st_mtime, tz=UTC)

    # A credential written into a harness-enrolled file after its enrolment
    # is masked here as the harness masks it at deposit (security review P48).
    # The snapshot then no longer holds the file's bytes, so the entry leaves
    # this byte-identity tier and the harness, which applies
    # the same mask, keeps it current.
    masked = _redact_harness_source(entry_row, content)
    if masked is not None:
        content = masked
        entry_row.fetch_policy = FetchPolicy.NEVER.value
        log.warning(
            "Entry %s: %s holds a credential shape; stored masked and left to the harvest "
            "(fetch_policy NEVER).",
            entry_row.entry_id[:8],
            path,
        )

    now = datetime.now(UTC)
    content_hash = sha256(content)

    # Tier 2 — content hash, authoritative. SHA-256 is the canonical snapshot
    # identity (techspec §7.2/§7.3: the blob-store key and the cross-operator
    # shared-archive identity), so change detection reuses it rather than
    # introducing a second, non-cryptographic hash for the
    # measurement behind that call.
    if prior is not None and content_hash == prior.content_hash:
        snap = await _write_revisit(session, entry_row.entry_id, prior, now, last_modified=mtime)
        log.info(
            "Entry %s: local content unchanged (hash match) → REVISIT %s",
            entry_row.entry_id[:8],
            snap.snapshot_id[:8],
        )
        return snap

    archive_path = save_blob(content, content_hash)
    snap = Snapshot(
        captured_at=now,
        content_hash=content_hash,
        last_modified=mtime,
        warc_record_type=WarcRecordType.RESPONSE,
        archive_path=archive_path,
        extraction_status=ExtractionStatus.PENDING,
    )
    session.add(SnapshotRow.from_model(snap, entry_id=entry_row.entry_id))
    await session.flush()

    # No staleness cascade here: the MUTABLE generation cascade runs
    # AFTER the new snapshot is extracted (particles/ingest/generation.py), so
    # carry-forward — which looks up ACTIVE particles — still sees the
    # prior generation and can keep unchanged chunks rather than re-paying for
    # them and minting duplicates.
    log.info(
        "Entry %s: local content changed → RESPONSE snapshot %s (PENDING)",
        entry_row.entry_id[:8],
        snap.snapshot_id[:8],
    )
    return snap


def _as_utc(dt: datetime) -> datetime:
    """Naive datetimes out of SQLite are UTC by construction; stamp them so."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


async def _do_fetch(
    session: AsyncSession,
    entry_row: CorpusEntryRow,
    prior: Snapshot | None,
) -> Snapshot:
    """Execute the actual HTTP fetch and return the resulting snapshot."""
    uri_r = entry_row.uri_r
    assert uri_r is not None

    # Re-validate on every refetch — DNS can move between deposit and refetch,
    # so the URL that was safe months ago may now resolve to something internal.
    from particles.url_safety import validate_fetch_url

    validate_fetch_url(uri_r)

    extra_headers: dict[str, str] = {}
    if prior:
        if prior.etag:
            extra_headers["If-None-Match"] = prior.etag
        if prior.last_modified:
            import calendar
            from email.utils import formatdate

            extra_headers["If-Modified-Since"] = formatdate(
                calendar.timegm(prior.last_modified.timetuple()), usegmt=True
            )

    async with particles_client(extra_headers=extra_headers) as client:
        resp = await get_capped(client, uri_r)

    now = datetime.now(UTC)

    # Tier 1: 304 Not Modified → REVISIT
    if resp.status_code == 304 and prior:
        snap = await _write_revisit(session, entry_row.entry_id, prior, now)
        log.info("Entry %s: 304 Not Modified → REVISIT %s", entry_row.entry_id, snap.snapshot_id)
        return snap

    resp.raise_for_status()
    return await _record_body(
        session,
        entry_row.entry_id,
        resp.content,
        prior,
        now,
        etag=resp.headers.get("etag"),
        last_modified=_parse_last_modified(resp.headers.get("last-modified")),
    )


async def _record_body(
    session: AsyncSession,
    entry_id: str,
    content: bytes,
    prior: Snapshot | None,
    now: datetime,
    *,
    etag: str | None = None,
    last_modified: datetime | None = None,
) -> Snapshot:
    """Tier 2 for a fetched body: REVISIT on a hash match, else a RESPONSE snapshot."""
    content_hash = sha256(content)

    if prior and content_hash == prior.content_hash:
        snap = await _write_revisit(session, entry_id, prior, now)
        log.info(
            "Entry %s: content unchanged (hash match) → REVISIT %s",
            entry_id,
            snap.snapshot_id,
        )
        return snap

    # New content: write RESPONSE snapshot
    archive_path = save_blob(content, content_hash)
    snap = Snapshot(
        captured_at=now,
        content_hash=content_hash,
        etag=etag,
        last_modified=last_modified,
        warc_record_type=WarcRecordType.RESPONSE,
        archive_path=archive_path,
        extraction_status=ExtractionStatus.PENDING,
    )
    snap_row = SnapshotRow.from_model(snap, entry_id=entry_id)
    session.add(snap_row)
    await session.flush()

    # The MUTABLE staleness cascade used to run here, BEFORE the new snapshot was
    # extracted. This was moved after extraction
    # (particles/ingest/generation.py): demoting first blinds the
    # carry-forward — which looks up ACTIVE particles — so it re-paid the LLM for
    # every unchanged paragraph and minted duplicates of claims that never
    # changed. The bug was latent only because this module had no caller.
    log.info(
        "Entry %s: content changed → RESPONSE snapshot %s",
        entry_id,
        snap.snapshot_id,
    )
    return snap


async def _write_revisit(
    session: AsyncSession,
    entry_id: str,
    prior: Snapshot,
    captured_at: datetime,
    last_modified: datetime | None = None,
) -> Snapshot:
    """A zero-byte REVISIT recording "as of ``captured_at``, still snapshot ``prior``".

    ``last_modified`` stamps the source signal observed alongside the match (the
    local tier's mtime), so the *next* run's tier 1 can short-circuit without a
    read. Without it a file whose mtime moved but whose bytes did not would be
    re-read every night forever.
    """
    snap = Snapshot(
        captured_at=captured_at,
        content_hash=prior.content_hash,
        last_modified=last_modified if last_modified is not None else prior.last_modified,
        warc_record_type=WarcRecordType.REVISIT,
        archive_path=None,
        refers_to=prior.snapshot_id,
        extraction_status=ExtractionStatus.COMPLETE,  # no new extraction needed
    )
    session.add(SnapshotRow.from_model(snap, entry_id=entry_id))
    await session.flush()
    return snap


def _parse_last_modified(value: str | None) -> datetime | None:
    if not value:
        return None
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(value)
    except Exception:
        return None


def is_entry_refused(row: CorpusEntryRow, recorded_roots: Sequence[Path]) -> bool:
    """Whether the local tier refuses this entry's ``file://`` source.

    :func:`entry_refusal` is not ``None``.
    """
    return entry_refusal(row, recorded_roots) is not None


def entry_refusal(
    row: CorpusEntryRow,
    recorded_roots: Sequence[Path],
    *,
    cwd_store: bool | RefreshRefusal = False,
) -> RefreshRefusal | None:
    """Why the local tier refuses this entry's ``file://`` source, or ``None``.

    The same judgement :func:`_do_local_refresh` makes, over the entry's own
    path and depositor and the file it names (:func:`local_refusal`), so the
    refresh loop can count a refusal before the ladder runs. ``cwd_store``
    (:func:`cwd_relative_store`) refuses every ``file://`` entry first,
    with the cause :func:`store_refusal` names when one is passed. An
    entry that is not ``file://``, or
    whose URI-R does not parse, is not refused here; the ladder reports it.
    """
    if row.uri_r is None or not row.uri_r.startswith("file://"):
        return None
    if cwd_store:
        return cwd_store if isinstance(cwd_store, RefreshRefusal) else RefreshRefusal.CWD_STORE
    try:
        path = path_from_file_uri(row.uri_r)
    except ValueError:
        return None
    return local_refusal(path, recorded_roots=recorded_roots, deposited_by=row.deposited_by)


async def count_refused_local_entries(session: AsyncSession) -> RefusalCounts:
    """How many ``LAZY`` ``file://`` entries the local tier would refuse to read, by cause.

    ``particles rules`` and ``hook doctor`` report it, so the window between an
    upgrade and the first ``rules sync`` is never silent, and
    the advice they print matches the cause.
    """
    recorded = await list_rule_source_roots(session)
    cwd_store = store_refusal(session) or False
    reasons: list[RefreshRefusal] = []
    for entry_id, _uri_r in await list_refreshable_local_entries(session):
        row = await session.get(CorpusEntryRow, entry_id)
        if row is None:
            continue
        if (reason := entry_refusal(row, recorded, cwd_store=cwd_store)) is not None:
            reasons.append(reason)
    return RefusalCounts.tally(reasons)


def cwd_relative_store(session: AsyncSession) -> Path | None:
    """The session's SQLite file when it may be one a repository shipped, else ``None``.

    A store that resolves against the working directory is whatever the
    directory holds, which a cloned repository can ship, so the local tier
    reads no host file on its say-so (security review 2026-10-24-2,
    P2). A store named by an absolute path is refused the same way when the
    file lies inside a git work tree, which is where a clone's shipped store
    sits however it came to be named (an install pin written from inside the
    clone, an inherited ``DATABASE_URL``), or when another user owns the file
    (security review 2026-10-26-1, W15, W65), or when its group or other users
    may write it (security review 2026-10-27-1, G16). A ``.git`` in the home
    directory or one of its ancestors does not count, since every store under
    home would otherwise be refused. The config has already expanded a leading
    ``~`` (P13). In-memory and non-SQLite stores are never refused.
    :func:`store_refusal` names which cause applies.
    """
    store = _sqlite_store_file(session)
    if store is None:
        return None
    return store if _store_file_refusal(store) is not None else None


def store_refusal(session: AsyncSession) -> RefreshRefusal | None:
    """Why the session's store, or the config, chooses no host file, or ``None``.

    :attr:`RefreshRefusal.STORE_WRITABLE_BY_OTHERS` when the operator's store
    file, one of its SQLite sidecars or the directory holding it is one other
    users may write, :attr:`RefreshRefusal.CWD_STORE` for every other cause the
    store has (see :func:`cwd_relative_store`), and, when the store passes,
    :attr:`RefreshRefusal.CONFIG_WRITABLE_BY_OTHERS` when the config file this
    process loaded fails the same predicate (:func:`config_file_judgement`).
    """
    store = _sqlite_store_file(session)
    if store is not None and (refusal := _store_file_refusal(store)) is not None:
        return refusal
    return None if config_file_judgement() is None else RefreshRefusal.CONFIG_WRITABLE_BY_OTHERS


def config_file_judgement() -> tuple[Path, StoreFileJudgement] | None:
    """The config file this process loaded and why it chooses no host file, or ``None``.

    The config names ``local_refresh.allowed_roots``, which admits any file
    under it, so the refresh judges it with the one predicate, as it judges
    the store: no other user may write it or create a name in the directory
    holding it. The file judged is the one the settings were
    loaded from (:func:`~particles.config.loaded_config_file`), never a fresh
    lookup: a name that no longer holds that file is refused, since the
    settings stay in force whatever the name holds now, and so is one that
    others could write, or that sat in a directory they could write, when the
    settings were read from it, or that changed since (security review
    2026-10-30-1, AB13). A config inside a git work tree is refused as the
    doctor reports it, since a clone can supply it.
    Compiled defaults load no file and pass. The cause is worded as the
    store's is.
    """
    loaded = loaded_config_file()
    if loaded is None:
        return None
    path = loaded.path
    name = Path(os.path.abspath(path))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(name, flags)
    except OSError:
        fd = None
    try:
        st = None if fd is None else os.fstat(fd)
        if st is None or (st.st_dev, st.st_ino) != (loaded.device, loaded.inode):
            return path, StoreFileJudgement(
                RefreshRefusal.CONFIG_WRITABLE_BY_OTHERS,
                "no longer the file this process loaded its settings from (restart the "
                "process once the config is a file only this account may write)",
            )
        exposure = writable_only_by_operator(
            name, fd=fd, named_ahead=True, operator_uid=effective_uid()
        )
    finally:
        if fd is not None:
            os.close(fd)
    if exposure is not None:
        worded = _store_exposure_judgement(exposure)
        return path, StoreFileJudgement(RefreshRefusal.CONFIG_WRITABLE_BY_OTHERS, worded.cause)
    if (loaded_open := _config_open_when_loaded(loaded, st)) is not None:
        return path, StoreFileJudgement(
            RefreshRefusal.CONFIG_WRITABLE_BY_OTHERS,
            f"{loaded_open} (restart the process once the config is a file only this "
            "account may write)",
        )
    if (tree := _work_tree_holding(name)) is not None:
        return path, StoreFileJudgement(
            RefreshRefusal.CONFIG_WRITABLE_BY_OTHERS,
            f"a file inside the git work tree {tree}, which a clone can supply",
        )
    return None


def _config_open_when_loaded(loaded: LoadedConfigFile, now: os.stat_result) -> str | None:
    """Why the state the settings were read in is not trusted, or ``None``.

    The settings stay in force after the file is closed, so a file, or the
    directory holding it, that others could write when it was read is refused
    whatever it says now, and so is a file whose mode, ACL or bytes changed
    since (its ``ctime``), since its settings no longer say what the file does
    (security review 2026-10-30-1, AB13).
    """
    if now.st_ctime_ns != loaded.ctime_ns:
        return "changed since this process loaded its settings from it"
    uid = effective_uid()
    if loaded.uid != uid or loaded.mode & WRITABLE_BY_OTHERS:
        return "a file other users could write when this process loaded its settings from it"
    if (
        loaded.dir_mode is None
        or loaded.dir_uid not in (uid, 0)
        or loaded.dir_mode & WRITABLE_BY_OTHERS
    ):
        return (
            "in a directory other users could write when this process loaded its settings from it"
        )
    return None


def _work_tree_holding(path: Path) -> Path | None:
    """The git work tree the resolved ``path`` lies in, below the first broad root, or ``None``."""
    for parent in Path(os.path.realpath(path)).parents:
        if is_broad_root(parent):
            break
        if os.path.lexists(parent / ".git"):
            return parent
    return None


def _sqlite_store_file(session: AsyncSession) -> Path | None:
    """The session's SQLite database file as named, or ``None`` for any other store."""
    bind = session.bind
    url = getattr(bind, "url", None)
    if url is None or not str(url.drivername).startswith("sqlite"):
        return None
    database = url.database
    if not database or database == ":memory:":
        return None
    return Path(database)


#: The files SQLite reads beside a store as part of it: the write-ahead log
#: every connection runs (``journal_mode=WAL``), its shared-memory index, and
#: the rollback journal a store last opened in another mode may leave.
STORE_SIDECARS = ("-wal", "-shm", "-journal")


@dataclass(frozen=True)
class StoreFileJudgement:
    """Why a named store (or config) file is not trusted as the operator's own.

    ``refusal`` is the refresh's cause; ``cause`` words it as a clause that
    follows the file's name (``"<file>, <cause>"``), naming the path at fault
    and, where one applies, the fix, so ``hook doctor``, ``init claude-code``
    and the refresh say the same thing.
    """

    refusal: RefreshRefusal
    cause: str


def _store_file_refusal(store: Path) -> RefreshRefusal | None:
    """Why a named store file chooses no host file, or ``None`` (see :func:`cwd_relative_store`)."""
    judgement = store_file_judgement(store)
    return None if judgement is None else judgement.refusal


def store_file_judgement(
    store: Path, *, sidecars: Sequence[str] = STORE_SIDECARS
) -> StoreFileJudgement | None:
    """Judge a named store file as the refresh does, or ``None`` when it is trusted.

    The one operator-only-writable predicate over the file, each
    sidecar SQLite reads as part of the store, and the directories its name
    resolves through: a user who may create a name in the directory holding the
    store can plant a ``-wal`` file whose rows SQLite serves as the store's own,
    so that directory is judged with the sticky bit counting for nothing
    (security review 2026-10-28-1, O18, O23). A store inside a git work tree is
    refused too, since a clone can supply it. The install and the
    doctor judge a config pin with no ``sidecars``.
    """
    if not store.is_absolute():
        return StoreFileJudgement(
            RefreshRefusal.CWD_STORE, "a path resolved against the working directory"
        )
    exposure = writable_only_by_operator(
        store, sidecars=sidecars, named_ahead=True, operator_uid=effective_uid()
    )
    if exposure is not None:
        return _store_exposure_judgement(exposure)
    if (tree := _work_tree_holding(store)) is not None:
        return StoreFileJudgement(
            RefreshRefusal.CWD_STORE,
            f"a file inside the git work tree {tree}, which a clone can supply",
        )
    return None


def _store_exposure_judgement(exposure: Exposure) -> StoreFileJudgement:
    """The store's refusal cause and its wording for one exposure."""
    what = f"whose SQLite sidecar {exposure.at} is a file" if exposure.sidecar else "a file"
    if exposure.no_uid:
        return StoreFileJudgement(
            RefreshRefusal.CWD_STORE,
            f"{what} whose owner this host cannot be asked, since it reports no effective uid",
        )
    if exposure.owners_ignored:
        return StoreFileJudgement(
            RefreshRefusal.CWD_STORE,
            f"{what} on a volume that ignores ownership, where every local user may write "
            "it (keep it on a volume that keeps ownership)",
        )
    if exposure.kind is ExposureKind.FILE_UNJUDGEABLE:
        return StoreFileJudgement(
            RefreshRefusal.CWD_STORE, f"{what} this account cannot open to judge whose it is"
        )
    if exposure.kind is ExposureKind.FILE_NOT_OWNED:
        return StoreFileJudgement(RefreshRefusal.CWD_STORE, f"{what} another user owns")
    if exposure.kind is ExposureKind.FILE_WRITABLE:
        return StoreFileJudgement(
            RefreshRefusal.STORE_WRITABLE_BY_OTHERS,
            f"{what} other users may write (remove their write access: chmod go-w "
            f"{exposure.at}, and any ACL entry that grants it)",
        )
    return StoreFileJudgement(
        RefreshRefusal.STORE_WRITABLE_BY_OTHERS,
        f"in the directory {exposure.at}, where other users may create or replace "
        "files (move it to a directory only this account may write)",
    )
