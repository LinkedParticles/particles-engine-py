# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The rule-source set — durable operating rules as tracked sources.

A *rule source* is a local document that states how work is done here:
``AGENTS.md``, ``CLAUDE.md``, an agent-memory note. A measurement showed why they
need their own intake path: the store held 34 ACTIVE particles mentioning the
never-prepend-``export PATH`` rule and **not one stating it** — every one was a
third-person report *about* the rule, mined from a conversation that discussed
it. Conversations about rules yield claims about rules; only the rule document
yields the rule.

Two halves live here:

* **Resolution** (:func:`resolve_rule_sources`, :func:`discover_default_roots`)
  — pure, store-free path work. Walks the registered roots for the configured
  filenames under a depth cap, a file cap, and a directory denylist.
* **Sync** (:func:`sync_rule_sources`) — deposits each resolved file
  ``LOCAL_MARKDOWN`` / ``MUTABLE`` / **``LAZY``**.

``LAZY`` is the whole integration with the refresh loop. Change detection, the
REVISIT/RESPONSE split, the ``MUTABLE`` generation cascade, consolidation pass
0.5 and ``particles corpus refresh`` are all 0206's and are not reimplemented
here — this module only decides *membership*, which is why its config section
(``rule_sources``) sits beside 0206's ``local_refresh`` rather than inside it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.observer_scope import project_tag
from particles.core.schema import CorpusEntry, FetchPolicy, Mutability
from particles.corpus.local_read import read_local_file
from particles.corpus.rule_roots import record_rule_source_roots

log = logging.getLogger(__name__)

#: Tag every rule-source entry carries, so the set is addressable after deposit
#: (``query --tag``, ``select.allow`` pins, the ``rules`` report).
RULE_SOURCE_TAG = "rule-file"

#: ``source_type`` for a rule document. This is floored at 0 s, so the
#: nightly cadence is the only rate limit on re-checking it.
RULE_SOURCE_TYPE = "LOCAL_MARKDOWN"


def is_rule_source(entry: CorpusEntry) -> bool:
    """True when ``entry`` was taken in as a rule file (``AGENTS.md``, ``CLAUDE.md``, …)."""
    return RULE_SOURCE_TAG in (entry.tags or [])


@dataclass(frozen=True)
class RuleSourceResolution:
    """What a resolution pass found, including what it had to drop.

    ``truncated`` is carried rather than logged because a design consequence
    makes disclosure a requirement: a cap that silently swallows files reads as
    "the set is complete" when it is not.
    """

    files: list[Path]
    roots: list[Path]
    discovered: bool
    truncated: int = 0
    missing: list[Path] = field(default_factory=list)
    escaped: list[Path] = field(default_factory=list)
    """Rule-file names that resolve outside their root, or onto a file that is not
    a rule file, and so were not taken in (security review D23)."""


@dataclass
class RuleSyncReport:
    """Outcome of one :func:`sync_rule_sources` pass."""

    resolution: RuleSourceResolution
    deposited: list[tuple[Path, str]] = field(default_factory=list)
    unchanged: list[tuple[Path, str]] = field(default_factory=list)
    skipped_empty: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return len(self.deposited)


# ---------------------------------------------------------------------------
# Resolution — pure path work, no store, no I/O beyond stat/iterdir
# ---------------------------------------------------------------------------


def discover_default_roots(cwd: Path | None = None) -> list[Path]:
    """The two roots used when ``rule_sources.paths`` is empty.

    The nearest ancestor of ``cwd`` containing a ``.git`` entry (the project
    root), plus ``~/.claude`` (the user-level agent config). Deliberately two
    named roots rather than a search of the home directory: discovery that
    cannot be predicted from the rule cannot be reviewed before it writes.

    A ``.git`` *file* counts as well as a directory — that is what a worktree
    or submodule checkout has.
    """
    roots: list[Path] = []
    start = (cwd or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            roots.append(candidate)
            break
    user_claude = Path.home() / ".claude"
    if user_claude.is_dir():
        roots.append(user_claude)
    return roots


def expand_path(raw: str) -> Path:
    """``~`` and ``$VAR`` expansion, then absolutise."""
    return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()


def rule_source_roots(paths: list[str] | None = None, *, cwd: Path | None = None) -> list[Path]:
    """The roots a resolution walks: ``paths``, else ``rule_sources.paths``, else discovery.

    Shared with the local refresh tier's root allowlist (``corpus/fetch.py``),
    so a file ``rules sync`` enrols is a file the refresh will read.
    """
    raw = list(paths) if paths else list(get_config().rule_sources.paths)
    return [expand_path(p) for p in raw] if raw else discover_default_roots(cwd)


def resolve_rule_sources(
    paths: list[str] | None = None,
    *,
    cwd: Path | None = None,
) -> RuleSourceResolution:
    """Resolve the configured (or supplied) rule-source set to concrete files.

    ``paths`` overrides ``rule_sources.paths`` for one call — the
    ``particles rules sync PATH…`` form. When both are empty, the roots come
    from :func:`discover_default_roots`.

    A registered *file* is taken as-is regardless of its name (an operator who
    names a path means that path). A registered *directory* is walked for
    ``rule_sources.filenames``, skipping any directory named in
    ``exclude_dirs`` at any level. Results are de-duplicated by resolved path
    and sorted, so the set is stable across runs and across root ordering.
    """
    cfg = get_config().rule_sources
    discovered = not (paths or cfg.paths)
    roots = rule_source_roots(paths, cwd=cwd)

    filenames = set(cfg.filenames)
    exclude = set(cfg.exclude_dirs)
    found: set[Path] = set()
    missing: list[Path] = []
    escaped: list[Path] = []

    for root in roots:
        if root.is_file():
            found.add(root)
        elif root.is_dir():
            found.update(_walk(root, filenames, exclude, cfg.max_depth, escaped))
        else:
            missing.append(root)

    ordered = sorted(found)
    truncated = 0
    if cfg.max_files and len(ordered) > cfg.max_files:
        truncated = len(ordered) - cfg.max_files
        ordered = ordered[: cfg.max_files]

    return RuleSourceResolution(
        files=ordered,
        roots=roots,
        discovered=discovered,
        truncated=truncated,
        missing=missing,
        escaped=escaped,
    )


def _walk(
    root: Path,
    filenames: set[str],
    exclude: set[str],
    max_depth: int,
    escaped: list[Path] | None = None,
) -> list[Path]:
    """Breadth-first walk of ``root`` collecting ``filenames``, depth-capped.

    Written as an explicit queue rather than ``rglob`` because the denylist has
    to prune *subtrees* — ``rglob`` would still descend into a 40k-file
    ``node_modules`` and filter afterwards.

    A match is kept as its resolved path only when that path is still a rule
    file under ``root`` (security review D23): a cloned repository's
    ``CLAUDE.md`` symlinked to any file the user can read would otherwise be
    deposited. ``CLAUDE.md`` linked to the same repository's ``AGENTS.md`` is
    kept. The rule is the refresh tier's (:func:`particles.corpus.fetch.is_refresh_allowed`),
    so an intake never takes in a file the refresh would refuse. A refused
    match lands in ``escaped``.
    """
    real_root = root.resolve()
    out: list[Path] = []
    queue: list[tuple[Path, int]] = [(root, 0)]
    while queue:
        directory, depth = queue.pop(0)
        try:
            children = sorted(directory.iterdir())
        except OSError as exc:
            log.debug("rule_sources: cannot list %s (%s)", directory, exc)
            continue
        for child in children:
            if child.is_dir():
                if depth < max_depth and child.name not in exclude and not child.is_symlink():
                    queue.append((child, depth + 1))
            elif child.name in filenames:
                target = child.resolve()
                if target.name in filenames and target.is_relative_to(real_root):
                    out.append(target)
                else:
                    log.warning(
                        "rule_sources: %s resolves to %s, outside %s or not a rule file; skipped",
                        child,
                        target,
                        root,
                    )
                    if escaped is not None:
                        escaped.append(child)
    return out


def rule_file_admitted(roots: list[Path]) -> Callable[[Path], bool]:
    """Containment over a resolution's roots, judged on a resolved path (security review D23).

    A path is admitted when it is a registered file root itself, or a file named
    in ``rule_sources.filenames`` under a root. The sync judges the descriptor
    it reads with this, so a name swapped after the walk is not followed out.
    """
    filenames = set(get_config().rule_sources.filenames)
    real_roots = [r.resolve() for r in roots]
    return lambda real: any(
        real == r or (real.name in filenames and real.is_relative_to(r)) for r in real_roots
    )


# ---------------------------------------------------------------------------
# Sync — the deposit half
# ---------------------------------------------------------------------------


def refresh_policy_for(raw: str, deposited: str) -> FetchPolicy:
    """``LAZY`` only when the deposited body *is* the file's bytes.

    The local tier is a **byte-identity** mechanism: it stats the file,
    SHA-256s the file, and compares against the snapshot's ``content_hash``. So
    it may only own a source whose snapshot was written from those same bytes.

    A rule file carrying a projected region is deposited *filtered*
    (the store must never archive its own rendered output), so its snapshot hash
    can never equal the file's hash. Enrolling it anyway does three things,
    measured on a scratch store rather than reasoned about:

    1. The next sweep sees a change that did not happen and writes a RESPONSE
       snapshot. This is **once** per file, not perpetual — the new snapshot
       records the file's own hash, so later sweeps are quiet.
    2. That snapshot holds the **unfiltered** bytes. A pristine projected region
       — the store's own rendered output — is thereby archived in the corpus,
       which is precisely what the round-trip contract (belt 1)
       forbids, and the re-extraction then mints beliefs from the store's own
       beliefs.
    3. The generation cascade then demotes the correctly-filtered
       generation in favour of the unfiltered one.

    (2) is the reason this rule exists; (1) is merely how it is triggered. Such a
    file stays ``NEVER`` and is refreshed by the path that knows how to transform
    it — ``rules sync`` for rule documents, the SessionEnd harvest for memory
    files.
    """
    return FetchPolicy.LAZY if deposited == raw else FetchPolicy.NEVER


def identity_filter(text: str) -> str:
    """Default deposit-body transform: none.

    The real transform — stripping projected regions — is injected by
    the caller via ``sync_rule_sources(filter_text=…)`` rather than applied
    here. That inversion is deliberate: the region snapshots live in the CLI
    state directory, and a direct reach from this module would put
    ``corpus → render`` into the graph, closing a
    ``store → corpus → render → store`` subpackage cycle the acyclic
    contract (correctly) rejects. ``particles.api.cli.rules`` supplies it.
    """
    return text


def _tags_for(path: Path, project_key_for: Callable[[Path], str | None] | None) -> list[str]:
    key = project_key_for(path) if project_key_for is not None else None
    return [RULE_SOURCE_TAG, *([project_tag(key)] if key else [])]


async def sync_rule_sources(
    session: AsyncSession,
    paths: list[str] | None = None,
    *,
    cwd: Path | None = None,
    project_key_for: Callable[[Path], str | None] | None = None,
    deposited_by: str = "rule-sync",
    dry_run: bool = False,
    filter_text: Callable[[str], str] = identity_filter,
) -> RuleSyncReport:
    """Deposit the resolved rule-source set as ``MUTABLE`` corpus entries.

    ``fetch_policy`` is ``LAZY`` — enrolling the file in the refresh
    loop — for every source whose deposited body is byte-identical to the file;
    see :func:`refresh_policy_for` for the one case that is not.

    Idempotent: identity is the ``file://`` URI-R, so a re-run over unchanged
    content is a content-hash no-op that still reconciles ``fetch_policy``—
    which is what enrols a file that never changes.

    ``filter_text`` transforms the raw bytes into the deposit body (see
    :func:`identity_filter`); the CLI passes the sentinel strip.

    ``project_key_for`` maps a rule file to the project it belongs to, or
    ``None`` for a global one: a repository's ``AGENTS.md`` is that
    project's, the user-level ``~/.claude/CLAUDE.md`` is everyone's. The mapping
    is the caller's because it is a harness's path convention, which this layer
    does not know. Omitted, every file is deposited keyless, as before.

    One unreadable file does not stop the sweep; it lands in
    :attr:`RuleSyncReport.failed`. Does not commit — the caller owns the
    transaction.

    Every resolved root that exists is recorded, in the same transaction, as
    a root the local refresh may read rule files under,
    including a path passed for this one call: the operator named it. A dry
    run records nothing.
    """
    from particles.corpus.deposit import deposit_text_versioned

    resolution = resolve_rule_sources(paths, cwd=cwd)
    report = RuleSyncReport(resolution=resolution)
    admitted = rule_file_admitted(resolution.roots)
    max_bytes = get_config().local_files.max_bytes

    for path in resolution.files:
        try:
            read = read_local_file(
                path, follow_symlinks=True, max_bytes=max_bytes, allowed=admitted
            )
            raw = read.content.decode("utf-8", errors="replace")
            text = filter_text(raw)
        except OSError as exc:
            report.failed.append((path, str(exc)))
            continue
        if not text.strip():
            # A rule file that is empty once its projected regions are stripped
            # carries no authored input — depositing it would create an entry
            # whose every future extraction yields nothing.
            report.skipped_empty.append(path)
            continue
        if dry_run:
            report.deposited.append((path, ""))
            continue
        try:
            mtime = datetime.fromtimestamp(read.stat.st_mtime, tz=UTC)
            entry_id, _snapshot_id, unchanged = await deposit_text_versioned(
                session,
                text=text,
                uri_r=path.as_uri(),
                source_type=RULE_SOURCE_TYPE,
                mutability=Mutability.MUTABLE,
                tags=_tags_for(path, project_key_for),
                deposited_by=deposited_by,
                content_published_at=mtime,
                fetch_policy=refresh_policy_for(raw, text),
            )
        except OSError as exc:
            report.failed.append((path, str(exc)))
            continue
        (report.unchanged if unchanged else report.deposited).append((path, entry_id))

    if not dry_run:
        await record_rule_source_roots(
            session, [root for root in resolution.roots if root.exists()], by=deposited_by
        )
    return report
