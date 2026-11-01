# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Who may write a file the local refresh trusts.

The characterisation table below makes a hand-run baseline standing.
Each row lays out one mechanism on the real filesystem and records what
three doors answer: the store gate (``fetch.store_file_judgement``), the counts
(``fetch.local_refusal`` on a rule file under a recorded root) and the refresh
read itself (``fetch._do_local_refresh``). It was committed before the predicate
was consolidated, so a row that changes value says which decision changed it.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from particles.config import reset_config
from particles.core.schema import CorpusEntry, FetchPolicy, Mutability
from particles.corpus import fetch, operator_only, rule_roots
from particles.corpus.rule_roots import record_rule_source_roots
from particles.corpus.store import CorpusEntryRow

DARWIN = sys.platform == "darwin"
_ORIGINAL_EUID = rule_roots.effective_uid


@dataclass
class Layout:
    """What one row laid out: the store file, a recorded root and a rule file under it."""

    store: Path | None = None
    root: Path | None = None
    rule: Path | None = None
    undo: list[Callable[[], None]] = field(default_factory=list)
    #: Steps applied once the root is recorded, which refuses a directory it cannot judge.
    later: list[Callable[[], None]] = field(default_factory=list)

    def chmod(self, path: Path, mode: int) -> None:
        """Set ``path``'s mode now and restore it afterwards, so the cleanup can remove it."""
        before = os.lstat(path).st_mode & 0o7777
        os.chmod(path, mode)
        self.undo.append(lambda: os.chmod(path, before))


def _acl(path: Path, entry: str) -> None:
    subprocess.run(["/bin/chmod", "+a", entry, str(path)], check=True)


def _operator_name() -> str:
    import pwd

    return pwd.getpwuid(os.getuid()).pw_name


@contextmanager
def _as_uid(value: int | None) -> Iterator[None]:
    """Make every module that reads the refresher's effective uid see ``value``."""
    targets = [
        module
        for name, module in list(sys.modules.items())
        if name.startswith("particles.")
        and getattr(module, "effective_uid", None) is _ORIGINAL_EUID
    ]
    with mock.patch.multiple(rule_roots, effective_uid=lambda: value):
        patches = [mock.patch.object(m, "effective_uid", lambda: value) for m in targets]
        for p in patches:
            p.start()
        try:
            yield
        finally:
            for p in patches:
                p.stop()


def _base(tmp: Path) -> Layout:
    """A closed layout: an operator-owned store and a recorded root holding ``CLAUDE.md``."""
    layout = Layout()
    sdir = tmp / "sdir"
    sdir.mkdir(mode=0o755)
    os.chmod(sdir, 0o755)
    layout.store = sdir / "p.db"
    layout.store.write_bytes(b"store")
    os.chmod(layout.store, 0o644)
    layout.root = tmp / "root"
    layout.root.mkdir(mode=0o755)
    os.chmod(layout.root, 0o755)
    layout.rule = layout.root / "CLAUDE.md"
    layout.rule.write_text("# Rules\n\nRun the gates.\n")
    os.chmod(layout.rule, 0o644)
    return layout


def _file_mode_gw(tmp: Path) -> Layout:
    layout = _base(tmp)
    assert layout.store and layout.rule
    layout.chmod(layout.store, 0o664)
    layout.chmod(layout.rule, 0o664)
    return layout


def _acl_on_files(entry: str) -> Callable[[Path], Layout]:
    def build(tmp: Path) -> Layout:
        layout = _base(tmp)
        assert layout.store and layout.rule
        _acl(layout.store, entry)
        _acl(layout.rule, entry)
        return layout

    return build


def _inherited_acl(tmp: Path) -> Layout:
    layout = Layout()
    for name in ("sdir", "root"):
        (tmp / name).mkdir(mode=0o755)
        _acl(tmp / name, "everyone allow write,file_inherit,only_inherit")
    layout.store = tmp / "sdir" / "p.db"
    layout.store.write_bytes(b"store")
    os.chmod(layout.store, 0o644)
    layout.root = tmp / "root"
    layout.rule = layout.root / "CLAUDE.md"
    layout.rule.write_text("# Rules\n")
    os.chmod(layout.rule, 0o644)
    return layout


def _acl_on_dirs(entry: str) -> Callable[[Path], Layout]:
    def build(tmp: Path) -> Layout:
        layout = _base(tmp)
        assert layout.store and layout.root
        _acl(layout.store.parent, entry)
        _acl(layout.root, entry)
        return layout

    return build


def _deny_and_operator(tmp: Path) -> Layout:
    layout = _base(tmp)
    assert layout.store and layout.rule
    for path in (layout.store, layout.rule):
        _acl(path, "everyone deny write")
        _acl(path, f"user:{_operator_name()} allow write")
    return layout


def _unopenable_file(tmp: Path) -> Layout:
    layout = _base(tmp)
    assert layout.store and layout.rule
    layout.chmod(layout.store, 0o000)
    layout.chmod(layout.rule, 0o000)
    return layout


def _holding_dir(mode: int) -> Callable[[Path], Layout]:
    def build(tmp: Path) -> Layout:
        layout = _base(tmp)
        assert layout.store and layout.root
        layout.chmod(layout.store.parent, mode)
        layout.chmod(layout.root, mode)
        return layout

    return build


def _parent_dir(mode: int) -> Callable[[Path], Layout]:
    def build(tmp: Path) -> Layout:
        mid = tmp / "mid"
        mid.mkdir(mode=0o755)
        os.chmod(mid, 0o755)
        layout = _base(mid)
        layout.later.append(lambda: layout.chmod(mid, mode))
        return layout

    return build


def _sidecar(mode: int) -> Callable[[Path], Layout]:
    def build(tmp: Path) -> Layout:
        layout = _base(tmp)
        assert layout.store
        wal = layout.store.with_name(layout.store.name + "-wal")
        wal.write_bytes(b"")
        layout.chmod(wal, mode)
        layout.rule = layout.root = None
        return layout

    return build


def _store_missing_in_open_dir(tmp: Path) -> Layout:
    parent = tmp / "open"
    parent.mkdir(mode=0o755)
    layout = Layout(store=parent / "not-yet" / "p.db")
    layout.chmod(parent, 0o757)
    return layout


def _symlinked_component_in_open_dir(tmp: Path) -> Layout:
    real = tmp / "real"
    real.mkdir(mode=0o755)
    os.chmod(real, 0o755)
    (real / "p.db").write_bytes(b"store")
    os.chmod(real / "p.db", 0o644)
    (real / "CLAUDE.md").write_text("# Rules\n")
    os.chmod(real / "CLAUDE.md", 0o644)
    opened = tmp / "open"
    opened.mkdir(mode=0o755)
    (opened / "link").symlink_to(real)
    layout = Layout(store=opened / "link" / "p.db", root=tmp, rule=opened / "link" / "CLAUDE.md")
    layout.chmod(opened, 0o777)
    return layout


def _hard_link(tmp: Path) -> Layout:
    layout = _base(tmp)
    assert layout.store and layout.rule
    os.link(layout.store, tmp / "store-link")
    os.link(layout.rule, tmp / "rule-link")
    return layout


def _no_acl_library(build: Callable[[Path], Layout]) -> Callable[[Path], Layout]:
    def wrapped(tmp: Path) -> Layout:
        layout = build(tmp)
        patch = mock.patch.object(operator_only, "_acl_library", lambda: None)
        patch.start()
        layout.undo.append(patch.stop)
        return layout

    return wrapped


@dataclass(frozen=True)
class Row:
    """One mechanism and what the store gate, the counts and the read answer."""

    name: str
    build: Callable[[Path], Layout]
    store: str
    counts: str
    read: str
    uid: str = "operator"  # or "foreign" (the euid another user's) or "none" (no euid)
    darwin_only: bool = False


ADMIT, REFUSE, NA, TRUSTED = "admitted", "refused", "n/a", "trusted"
SWBO, CWD = "STORE_WRITABLE_BY_OTHERS", "CWD_STORE"
WBO = "WRITABLE_BY_OTHERS"

ROWS = [
    Row("every part closed (control)", _base, TRUSTED, ADMIT, ADMIT),
    Row("file mode g+w", _file_mode_gw, SWBO, WBO, REFUSE),
    Row(
        "file and root another user's (uid patched)",
        _base,
        CWD,
        "ROOT_NOT_OWNED",
        REFUSE,
        uid="foreign",
    ),
    Row(
        "file ACL allow write",
        _acl_on_files("everyone allow write"),
        SWBO,
        WBO,
        REFUSE,
        darwin_only=True,
    ),
    Row("file ACL allow,inherited", _inherited_acl, SWBO, WBO, REFUSE, darwin_only=True),
    Row(
        "holding directory ACL add_file,file_inherit,directory_inherit",
        _acl_on_dirs("everyone allow add_file,file_inherit,directory_inherit"),
        SWBO,
        WBO,
        REFUSE,
        darwin_only=True,
    ),
    Row(
        "holding directory only_inherit, file made before it",
        _acl_on_dirs("everyone allow write,file_inherit,only_inherit"),
        TRUSTED,
        ADMIT,
        ADMIT,
        darwin_only=True,
    ),
    Row(
        "ACL that only denies, plus one naming the operator",
        _deny_and_operator,
        TRUSTED,
        ADMIT,
        ADMIT,
        darwin_only=True,
    ),
    Row(
        "ACL granting root write",
        _acl_on_files("user:root allow write"),
        SWBO,
        WBO,
        REFUSE,
        darwin_only=True,
    ),
    Row(
        "file ACL delete,readattr,writeattr,writeextattr only",
        _acl_on_files("everyone allow delete,readattr,writeattr,writeextattr"),
        TRUSTED,
        ADMIT,
        ADMIT,
        darwin_only=True,
    ),
    Row(
        "ACL calls unavailable",
        _no_acl_library(_base),
        SWBO,
        WBO,
        REFUSE,
        darwin_only=True,
    ),
    Row("file exists, cannot be opened", _unopenable_file, CWD, "UNREADABLE", REFUSE),
    Row("holding directory o+w, no sticky bit", _holding_dir(0o757), SWBO, WBO, REFUSE),
    Row("holding directory o+wt (sticky)", _holding_dir(0o1777), SWBO, ADMIT, ADMIT),
    Row("a parent o+w, no sticky bit", _parent_dir(0o757), SWBO, WBO, REFUSE),
    Row("a parent o+wt (sticky)", _parent_dir(0o1777), TRUSTED, ADMIT, ADMIT),
    Row("a parent that cannot be opened (mode 311)", _parent_dir(0o311), SWBO, WBO, REFUSE),
    Row("-wal g+w", _sidecar(0o664), SWBO, NA, NA),
    Row("-wal cannot be opened", _sidecar(0o000), CWD, NA, NA),
    Row("store missing, nearest existing directory o+w", _store_missing_in_open_dir, SWBO, NA, NA),
    # the directory a symlinked component lives in is judged.
    Row(
        "symlinked component in a directory others may write",
        _symlinked_component_in_open_dir,
        SWBO,
        WBO,
        REFUSE,
    ),
    Row("hard link (closed inode)", _hard_link, TRUSTED, "HARD_LINKED", REFUSE),
    # with no effective uid nothing can be judged, so nothing is read.
    Row("no effective uid, file g+w", _file_mode_gw, CWD, "UNREADABLE", REFUSE, uid="none"),
    Row("no effective uid, closed", _base, CWD, "UNREADABLE", REFUSE, uid="none"),
]


async def _lazy_entry(session: Any, path: Path) -> CorpusEntryRow:
    """A ``LAZY`` rule-sync entry naming ``path`` as given, symlinks unresolved."""
    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="LOCAL_MARKDOWN",
        uri_r=Path(os.path.abspath(path)).as_uri(),
        fetch_policy=FetchPolicy.LAZY,
        mutability=Mutability.MUTABLE,
        deposited_by="rule-sync",
    )
    row = CorpusEntryRow.from_model(entry)
    session.add(row)
    await session.flush()
    return row


@pytest.mark.asyncio
@pytest.mark.parametrize("row", ROWS, ids=[r.name for r in ROWS])
async def test_characterisation_of_the_three_doors(
    row: Row, db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each mechanism gives the recorded answer at the store gate, the counts and the read."""
    if row.darwin_only and not DARWIN:
        pytest.skip("macOS extended ACLs")
    if os.geteuid() == 0:
        pytest.skip("root reads every file, so the mode rows cannot be laid out")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    tmp = tmp_path / "case"
    tmp.mkdir()
    layout = row.build(tmp)
    entry = None
    if layout.root is not None and layout.rule is not None:
        await record_rule_source_roots(db_session, [layout.root], by="rule-sync")
        entry = await _lazy_entry(db_session, layout.rule)
        await db_session.flush()
    uid = {"operator": os.geteuid(), "foreign": os.geteuid() + 1, "none": None}[row.uid]
    try:
        for step in layout.later:
            step()
        with _as_uid(uid):
            judgement = None if layout.store is None else fetch.store_file_judgement(layout.store)
            store = (
                NA
                if layout.store is None
                else (TRUSTED if judgement is None else judgement.refusal.name)
            )
            if layout.rule is None or layout.root is None or entry is None:
                counts = read = NA
            else:
                reason = fetch.local_refusal(
                    layout.rule, recorded_roots=[layout.root], deposited_by="rule-sync"
                )
                counts = ADMIT if reason is None else reason.name
                snap = await fetch._do_local_refresh(db_session, entry, None)
                read = ADMIT if snap is not None else REFUSE
    finally:
        for undo in reversed(layout.undo):
            undo()
    assert (store, counts, read) == (row.store, row.counts, row.read)


# ---------------------------------------------------------------------------
# The §6 table: every mechanism as gathered facts, judged on any host
# ---------------------------------------------------------------------------

OPERATOR, OTHER = 1000, 1001
_ACL_HEAD = "!#acl 1\n"


def _acl_text(*entries: str) -> str:
    return _ACL_HEAD + "".join(f"{e}\n" for e in entries)


def _everyone(kind: str, perms: str) -> str:
    return f"group:ABCDEFAB-CDEF-ABCD-EFAB-CDEF0000000C:everyone:12:{kind}:{perms}"


def _node(
    path: str,
    *,
    mode: int = 0o644,
    uid: int | None = OPERATOR,
    exists: bool = True,
    opened: bool = True,
    acl: operator_only.AclState = operator_only.AclState.NONE,
    text: str = "",
) -> operator_only.NodeFacts:
    return operator_only.NodeFacts(Path(path), exists, opened, uid, mode, acl, text)


def _dir(path: str, **kw: Any) -> operator_only.NodeFacts:
    kw.setdefault("mode", 0o40755)
    return _node(path, **kw)


def _facts(
    *,
    file: operator_only.NodeFacts | None = None,
    sidecars: tuple[operator_only.NodeFacts, ...] = (),
    holding: operator_only.NodeFacts | None = None,
    above: tuple[operator_only.NodeFacts, ...] | None = None,
) -> operator_only.WriteFacts:
    return operator_only.WriteFacts(
        file=file or _node("/w/d/f"),
        sidecars=sidecars,
        holding=holding or _dir("/w/d"),
        above=above if above is not None else (_dir("/w"),),
    )


_TEXT = operator_only.AclState.TEXT


@dataclass(frozen=True)
class FactRow:
    """One mechanism as facts, and what the store and a rule file are refused as."""

    name: str
    facts: operator_only.WriteFacts
    store: str
    rule: str
    operator_uid: int | None = OPERATOR


FACT_ROWS = [
    FactRow("every part closed (control)", _facts(), TRUSTED, TRUSTED),
    FactRow("file mode g+w", _facts(file=_node("/w/d/f", mode=0o664)), SWBO, WBO),
    FactRow("file mode o+w", _facts(file=_node("/w/d/f", mode=0o646)), SWBO, WBO),
    FactRow(
        "file owned by another user", _facts(file=_node("/w/d/f", uid=OTHER)), CWD, "FILE_NOT_OWNED"
    ),
    FactRow("file owned by root", _facts(file=_node("/w/d/f", uid=0)), CWD, "FILE_NOT_OWNED"),
    FactRow(
        "file ACL allow write",
        _facts(file=_node("/w/d/f", acl=_TEXT, text=_acl_text(_everyone("allow", "write")))),
        SWBO,
        WBO,
    ),
    FactRow(
        "file ACL allow,inherited",
        _facts(
            file=_node("/w/d/f", acl=_TEXT, text=_acl_text(_everyone("allow,inherited", "write")))
        ),
        SWBO,
        WBO,
    ),
    FactRow(
        "file ACL entry only_inherit (applies below, not here)",
        _facts(
            file=_node(
                "/w/d/f",
                acl=_TEXT,
                text=_acl_text(_everyone("allow,file_inherit,only_inherit", "write")),
            )
        ),
        TRUSTED,
        TRUSTED,
    ),
    FactRow(
        "ACL that only denies",
        _facts(file=_node("/w/d/f", acl=_TEXT, text=_acl_text(_everyone("deny", "write")))),
        TRUSTED,
        TRUSTED,
    ),
    FactRow(
        "ACL entry naming the operator",
        _facts(
            file=_node(
                "/w/d/f", acl=_TEXT, text=_acl_text(f"user:U:op:{OPERATOR}:allow:write,append")
            )
        ),
        TRUSTED,
        TRUSTED,
    ),
    FactRow(
        "ACL granting root write",
        _facts(file=_node("/w/d/f", acl=_TEXT, text=_acl_text("user:U:root:0:allow:write"))),
        SWBO,
        WBO,
    ),
    FactRow(
        "ACL delete, readattr, writeattr, writeextattr only",
        _facts(
            file=_node(
                "/w/d/f",
                acl=_TEXT,
                text=_acl_text(_everyone("allow", "delete,readattr,writeattr,writeextattr")),
            )
        ),
        TRUSTED,
        TRUSTED,
    ),
    FactRow(
        "ACL that cannot be read",
        _facts(file=_node("/w/d/f", acl=operator_only.AclState.UNREADABLE)),
        SWBO,
        WBO,
    ),
    FactRow(
        "ACL calls unavailable",
        _facts(file=_node("/w/d/f", acl=operator_only.AclState.UNAVAILABLE)),
        SWBO,
        WBO,
    ),
    FactRow(
        "file exists, cannot be opened",
        _facts(file=_node("/w/d/f", opened=False, mode=0o100000)),
        CWD,
        "UNREADABLE",
    ),
    FactRow("holding directory o+w", _facts(holding=_dir("/w/d", mode=0o40757)), SWBO, WBO),
    FactRow(
        "holding directory o+wt (sticky)", _facts(holding=_dir("/w/d", mode=0o41777)), SWBO, TRUSTED
    ),
    FactRow("holding directory another user's", _facts(holding=_dir("/w/d", uid=OTHER)), SWBO, WBO),
    FactRow(
        "holding directory root's, closed", _facts(holding=_dir("/w/d", uid=0)), TRUSTED, TRUSTED
    ),
    FactRow(
        "holding directory ACL add_file,file_inherit,directory_inherit",
        _facts(
            holding=_dir(
                "/w/d",
                acl=_TEXT,
                text=_acl_text(_everyone("allow,file_inherit,directory_inherit", "write")),
            )
        ),
        SWBO,
        WBO,
    ),
    FactRow(
        "holding directory ACL only_inherit",
        _facts(
            holding=_dir(
                "/w/d",
                acl=_TEXT,
                text=_acl_text(_everyone("allow,file_inherit,only_inherit", "write")),
            )
        ),
        TRUSTED,
        TRUSTED,
    ),
    FactRow(
        "holding directory cannot be opened",
        _facts(holding=_dir("/w/d", opened=False, mode=None, uid=None)),
        SWBO,
        WBO,
    ),
    FactRow("a parent o+w", _facts(above=(_dir("/w", mode=0o40757),)), SWBO, WBO),
    FactRow("a parent o+wt (sticky)", _facts(above=(_dir("/w", mode=0o41777),)), TRUSTED, TRUSTED),
    FactRow(
        "a parent that cannot be opened",
        _facts(above=(_dir("/w", opened=False, mode=None, uid=None),)),
        SWBO,
        WBO,
    ),
    FactRow(
        "a sidecar another user's",
        _facts(sidecars=(_node("/w/d/f-wal", uid=OTHER),)),
        CWD,
        NA,
    ),
    FactRow("a sidecar g+w", _facts(sidecars=(_node("/w/d/f-wal", mode=0o664),)), SWBO, NA),
    FactRow(
        "a sidecar that cannot be opened",
        _facts(sidecars=(_node("/w/d/f-wal", opened=False, mode=0o100000),)),
        CWD,
        NA,
    ),
    FactRow(
        "sidecars missing, holding directory closed",
        _facts(sidecars=(_node("/w/d/f-wal", exists=False, opened=False, uid=None, mode=None),)),
        TRUSTED,
        NA,
    ),
    FactRow(
        "file missing, nearest existing directory o+w",
        _facts(
            file=_node("/w/d/f", exists=False, opened=False, uid=None, mode=None),
            holding=_dir("/w/d", mode=0o40757),
        ),
        SWBO,
        NA,
    ),
    FactRow("no effective uid, closed", _facts(), CWD, "UNREADABLE", operator_uid=None),
    FactRow(
        "no effective uid, file g+w",
        _facts(file=_node("/w/d/f", mode=0o664)),
        CWD,
        "UNREADABLE",
        operator_uid=None,
    ),
    # A volume that ignores ownership reports every file as the caller's and
    # lets every user write as its owner, so nothing on it can be judged.
    FactRow(
        "file on a volume that ignores ownership",
        _facts(file=replace(_node("/w/d/f", mode=0o600), owners_ignored=True)),
        CWD,
        "UNREADABLE",
    ),
    FactRow(
        "holding directory on a volume that ignores ownership",
        _facts(holding=replace(_dir("/w/d", mode=0o40700), owners_ignored=True)),
        SWBO,
        WBO,
    ),
    # A symlink's owner may replace it even where the sticky bit stops
    # everyone else, so a link another user owns opens its directory.
    FactRow(
        "a symlink on the way another user owns",
        replace(_facts(), links=(operator_only.LinkFacts(Path("/w/s/f"), OTHER),)),
        SWBO,
        WBO,
    ),
    FactRow(
        "a symlink on the way the operator or root owns",
        replace(
            _facts(),
            links=(
                operator_only.LinkFacts(Path("/w/s/f"), OPERATOR),
                operator_only.LinkFacts(Path("/w/t"), 0),
            ),
        ),
        TRUSTED,
        TRUSTED,
    ),
    # The name as given is a symlink in a sticky directory: for a file named
    # ahead, anyone could have created it there, so the sticky bit counts for
    # nothing; for a rule file it closes the directory, as today.
    FactRow(
        "the file's name a symlink in a sticky directory",
        replace(_facts(), named=(_dir("/w/s", mode=0o41777),)),
        SWBO,
        TRUSTED,
    ),
    FactRow(
        "the file's name a symlink in a directory others may write",
        replace(_facts(), named=(_dir("/w/s", mode=0o40757),)),
        SWBO,
        WBO,
    ),
]


@pytest.mark.parametrize("row", FACT_ROWS, ids=[r.name for r in FACT_ROWS])
def test_the_judgement_over_gathered_facts(row: FactRow) -> None:
    """Each mechanism, as facts, gives the store and a rule file their causes."""
    store = operator_only.judge_write_facts(
        row.facts, operator_uid=row.operator_uid, named_ahead=True
    )
    assert (
        TRUSTED if store is None else fetch._store_exposure_judgement(store).refusal.name
    ) == row.store
    if row.rule == NA:
        return
    rule = operator_only.judge_write_facts(
        replace(row.facts, sidecars=()), operator_uid=row.operator_uid, named_ahead=False
    )
    assert (TRUSTED if rule is None else fetch.rule_file_refusal(rule).name) == row.rule


def test_every_caller_names_named_ahead_and_the_uid() -> None:
    """``named_ahead`` and ``operator_uid`` have no default, and each call site picks its rule."""
    import inspect

    sig = inspect.signature(operator_only.writable_only_by_operator)
    assert sig.parameters["named_ahead"].default is inspect.Parameter.empty
    assert sig.parameters["operator_uid"].default is inspect.Parameter.empty
    root = Path(fetch.__file__).parent
    calls = {
        "fetch.py": ("named_ahead=True", "named_ahead=False"),
        "local_read.py": ("named_ahead=False",),
    }
    for name, expected in calls.items():
        text = (root / name).read_text(encoding="utf-8")
        for snippet in expected:
            assert snippet in text, (name, snippet)


# ---------------------------------------------------------------------------
# The config file the refresh loaded
# ---------------------------------------------------------------------------


def _config(tmp: Path, *, file_mode: int = 0o600, dir_mode: int = 0o700) -> Path:
    directory = tmp / "cfg"
    directory.mkdir()
    path = directory / "config.yaml"
    path.write_text("", encoding="utf-8")
    os.chmod(path, file_mode)
    os.chmod(directory, dir_mode)
    return path


@pytest.mark.parametrize(
    ("file_mode", "dir_mode", "refused"),
    [
        (0o600, 0o700, False),
        (0o620, 0o700, True),
        (0o600, 0o757, True),
        # A config is named before it exists, so the sticky bit does not close
        # the directory that holds it.
        (0o600, 0o1777, True),
    ],
)
def test_the_refresh_judges_the_config_it_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_mode: int, dir_mode: int, refused: bool
) -> None:
    """A config others may write chooses no host file."""
    config = _config(tmp_path, file_mode=file_mode, dir_mode=dir_mode)
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    try:
        found = fetch.config_file_judgement()
    finally:
        os.chmod(config.parent, 0o700)
    if not refused:
        assert found is None
        return
    assert found is not None
    assert found[0] == config
    assert found[1].refusal is fetch.RefreshRefusal.CONFIG_WRITABLE_BY_OTHERS


def test_compiled_defaults_load_no_config_to_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PARTICLES_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert fetch.config_file_judgement() is None


@pytest.fixture
def writable_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config others may write, loaded before the store session opens.

    The refresh judges the file the settings were loaded from,
    so it is loaded here, ahead of ``db_session``: reloading the config
    discards the store's engine.
    """
    path = _config(tmp_path, file_mode=0o620)
    monkeypatch.setenv("PARTICLES_CONFIG", str(path))
    reset_config()
    return path


@pytest.mark.asyncio
async def test_a_writable_config_refuses_and_counts_every_local_entry(
    writable_config: Path, db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.operations.consolidation import ConsolidationReport, _pass_refresh
    from particles.operations.corpus_refresh import refresh_entries

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    layout = _base(tmp_path)
    assert layout.root and layout.rule
    await record_rule_source_roots(db_session, [layout.root], by="rule-sync")
    entry = await _lazy_entry(db_session, layout.rule)
    await db_session.commit()

    (result,) = [r async for r in refresh_entries(db_session, [entry.entry_id])]
    assert result.refusal is fetch.RefreshRefusal.CONFIG_WRITABLE_BY_OTHERS
    assert await fetch._do_local_refresh(db_session, entry, None) is None
    counts = await fetch.count_refused_local_entries(db_session)
    assert counts.config_writable_by_others == 1
    assert "config file" in counts.lines()[0]
    report = ConsolidationReport()
    assert await _pass_refresh(db_session, report) == 0
    assert report.refresh_skipped == "config_writable_by_others"
    assert report.refresh_checked == 0


def test_corpus_refresh_names_a_writable_config_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from typer.testing import CliRunner

    from particles.api.cli import app
    from particles.db import Base, get_engine, session_scope

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    store_dir = tmp_path / "store"
    store_dir.mkdir(mode=0o700)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{store_dir / 'p.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    reset_config()
    layout = _base(tmp_path)
    assert layout.rule

    async def _seed() -> None:
        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with session_scope() as session:
            await _lazy_entry(session, layout.rule)
            await session.commit()
        await get_engine().dispose()

    asyncio.run(_seed())
    config = _config(tmp_path, file_mode=0o620)
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    reset_config()
    result = CliRunner().invoke(app, ["corpus", "refresh"])

    assert result.exit_code == 1, result.output
    assert f"the config file is {config}" in result.stderr
    assert "chmod go-w" in result.stderr


# ---------------------------------------------------------------------------
# Security review 2026-10-28-2: ownership-ignoring volumes, symlinks in sticky
# directories, the loaded config, SQLite's sidecars and a host with no uid
# ---------------------------------------------------------------------------


def _closed_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def _closed_file(path: Path, text: str = "") -> Path:
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def _noowners_file() -> Path | None:
    """A regular file on a volume already mounted with ownership ignored, read only, or ``None``."""
    try:
        out = subprocess.run(["/sbin/mount"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        m = re.match(r"^\S+ on (.+) \(([^)]*)\)$", line)
        if not m or "noowners" not in m.group(2):
            continue
        for dirpath, _dirs, files in os.walk(m.group(1)):
            for name in files:
                candidate = Path(dirpath) / name
                if candidate.is_file() and not candidate.is_symlink():
                    return candidate
    return None


@pytest.mark.skipif(not DARWIN, reason="macOS volume flags")
def test_a_file_on_a_volume_that_ignores_ownership_is_not_judged_the_operators() -> None:
    """A file on a volume where every user acts as its owner is not judged the operator's own."""
    target = _noowners_file()
    if target is None:
        pytest.skip("no volume mounted with ownership ignored")
    exposure = operator_only.writable_only_by_operator(
        target, named_ahead=False, operator_uid=os.geteuid()
    )
    assert exposure is not None
    assert exposure.owners_ignored


@pytest.mark.skipif(not DARWIN, reason="macOS volume flags")
def test_the_volume_flags_are_read_from_the_descriptor(tmp_path: Path) -> None:
    """``fstatfs`` fills the layout the reader declares: the volume holding the path, by name."""
    fd = os.open(tmp_path, os.O_RDONLY)
    try:
        found = operator_only.volume_statfs(fd)
        ignored = operator_only.ownership_ignored(fd)
    finally:
        os.close(fd)
    assert found is not None
    assert found.f_fstypename
    # The mount point it names is the volume the path is on (by device, since
    # a firmlink can put the path outside the mount point's own name).
    assert os.stat(found.f_mntonname.decode()).st_dev == os.stat(tmp_path).st_dev
    assert not ignored  # the suite's temporary directory keeps ownership


@pytest.mark.skipif(not DARWIN, reason="macOS volume flags")
def test_unreadable_volume_flags_are_not_trusted(tmp_path: Path) -> None:
    """A volume whose flags cannot be read counts as one that ignores ownership."""
    store = _closed_file(_closed_dir(tmp_path / "closed") / "p.db")
    with mock.patch.object(operator_only, "volume_statfs", lambda fd: None):
        judgement = fetch.store_file_judgement(store)
    assert judgement is not None
    assert "ignores ownership" in judgement.cause


def test_a_store_on_a_volume_that_ignores_ownership_is_not_trusted(tmp_path: Path) -> None:
    """The store gate refuses a store whose volume ignores ownership, and says so."""
    store = _closed_file(_closed_dir(tmp_path / "closed") / "p.db")
    assert fetch.store_file_judgement(store) is None
    with mock.patch.object(operator_only, "ownership_ignored", lambda fd: True):
        judgement = fetch.store_file_judgement(store)
    assert judgement is not None
    assert judgement.refusal is fetch.RefreshRefusal.CWD_STORE
    assert "ignores ownership" in judgement.cause


# The § 6 rows below are laid out on the real filesystem, beside the fact rows
# in FACT_ROWS, so a gather regression fails here even while the facts pass
# (security review 2026-10-30-1, AB60). The rows that need another account's
# file (another user's sidecar, holding directory or link) stay fact-only: no
# test can make one unprivileged. Files outside tmp_path are only judged.


def test_a_root_owned_store_file_is_refused_on_the_real_filesystem() -> None:
    """A file root owns is not the operator's own."""
    hosts = Path("/etc/hosts").resolve()
    if os.stat(hosts).st_uid != 0:  # pragma: no cover - an unusual host
        pytest.skip("/etc/hosts is not root's here")
    verdict = fetch.store_file_judgement(hosts)
    assert verdict is not None and verdict.refusal is fetch.RefreshRefusal.CWD_STORE


def test_a_missing_store_in_a_root_owned_closed_directory_is_trusted_on_the_real_filesystem() -> (
    None
):
    """A root-owned holding directory nobody else may write is closed."""
    holding = Path("/usr/bin")
    st = os.stat(holding)
    if st.st_uid != 0 or st.st_mode & 0o022:  # pragma: no cover - an unusual host
        pytest.skip("/usr/bin is not a closed root-owned directory here")
    assert fetch.store_file_judgement(holding / "particles-test-absent.db") is None


def test_a_device_node_at_the_store_name_is_refused_on_the_real_filesystem() -> None:
    """A device node at the store's name is never trusted as the operator's store."""
    assert fetch.store_file_judgement(Path("/dev/null")) is not None


@pytest.mark.skipif(not DARWIN, reason="macOS ACLs")
@pytest.mark.parametrize(
    ("where", "entry"),
    [
        ("file", "everyone allow write"),
        ("wal", "everyone allow write"),
        ("holding", "everyone allow add_file"),
    ],
)
def test_an_acl_write_grant_on_the_store_its_sidecar_or_its_directory_is_refused(
    tmp_path: Path, where: str, entry: str
) -> None:
    """A real ACL entry letting others write the store, a sidecar or its directory is refused."""
    holding = _closed_dir(tmp_path / "closed")
    store = _closed_file(holding / "p.db")
    wal = _closed_file(holding / "p.db-wal")
    target = {"file": store, "wal": wal, "holding": holding}[where]
    assert fetch.store_file_judgement(store) is None
    subprocess.run(["/bin/chmod", "+a", entry, str(target)], check=True)
    try:
        assert fetch.store_file_judgement(store) is not None
    finally:
        subprocess.run(["/bin/chmod", "-N", str(target)], check=True)


def _symlink_in_sticky(tmp_path: Path, name: str) -> tuple[Path, Path]:
    real = _closed_file(_closed_dir(tmp_path / "closed") / name)
    sticky = tmp_path / "sticky"
    sticky.mkdir()
    os.chmod(sticky, 0o1777)
    return real, sticky / name


def test_a_store_named_in_a_sticky_directory_is_refused_as_a_symlink_too(tmp_path: Path) -> None:
    """A store named where others may create names is refused, as a file or as a link."""
    real, name = _symlink_in_sticky(tmp_path, "p.db")
    try:
        assert fetch.store_file_judgement(name) is not None
        name.symlink_to(real)
        assert fetch.store_file_judgement(real) is None
        assert fetch.store_file_judgement(name) is not None
    finally:
        os.chmod(name.parent, 0o700)


def test_a_config_named_in_a_sticky_directory_is_refused_as_a_symlink_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config named where others may create names is refused, as a file or as a link."""
    real, name = _symlink_in_sticky(tmp_path, "config.yaml")
    try:
        assert (
            operator_only.writable_only_by_operator(
                name, named_ahead=True, operator_uid=os.geteuid()
            )
            is not None
        )
        name.symlink_to(real)
        monkeypatch.setenv("PARTICLES_CONFIG", str(name))
        reset_config()
        assert fetch.config_file_judgement() is not None
    finally:
        os.chmod(name.parent, 0o700)


def test_a_rule_file_symlinked_in_a_sticky_directory_keeps_todays_rule(tmp_path: Path) -> None:
    """A file read once it exists keeps the sticky bit closing the directory its link lives in."""
    real, name = _symlink_in_sticky(tmp_path, "CLAUDE.md")
    try:
        name.symlink_to(real)
        assert (
            operator_only.writable_only_by_operator(
                name, named_ahead=False, operator_uid=os.geteuid()
            )
            is None
        )
    finally:
        os.chmod(name.parent, 0o700)


def test_a_symlink_another_user_owns_opens_the_directory_it_lives_in(tmp_path: Path) -> None:
    """A link's owner may replace it in a sticky directory, so a foreign link is not trusted."""
    real = _closed_file(_closed_dir(tmp_path / "closed") / "CLAUDE.md")
    sticky = tmp_path / "sticky"
    sticky.mkdir()
    (sticky / "d").symlink_to(real.parent, target_is_directory=True)
    os.chmod(sticky, 0o1777)
    name = sticky / "d" / "CLAUDE.md"
    try:
        assert (
            operator_only.writable_only_by_operator(
                name, named_ahead=False, operator_uid=os.geteuid()
            )
            is None
        )
        facts = operator_only.gather_write_facts(name)
        foreign = replace(
            facts, links=tuple(replace(link, uid=os.geteuid() + 1) for link in facts.links)
        )
        exposure = operator_only.judge_write_facts(
            foreign, operator_uid=os.geteuid(), named_ahead=False
        )
    finally:
        os.chmod(sticky, 0o700)
    assert [link.path for link in facts.links] == [sticky / "d"]
    assert exposure is not None
    assert exposure.kind is operator_only.ExposureKind.PLACE_OPEN
    assert exposure.at == sticky


def _one_row_store(path: Path, value: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE t (x TEXT)")
        conn.execute("INSERT INTO t VALUES (?)", (value,))
        conn.commit()
    finally:
        conn.close()


def test_a_passing_store_gate_judges_the_file_the_open_connection_reads(tmp_path: Path) -> None:
    """When the store gate trusts the store, the connection already open serves the trusted file."""
    closed = _closed_dir(tmp_path / "closed")
    operators = closed / "p.db"
    _one_row_store(operators, "operator")
    os.chmod(operators, 0o600)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    planted = elsewhere / "p.db"
    _one_row_store(planted, "planted")
    sticky = tmp_path / "sticky"
    sticky.mkdir()
    os.chmod(sticky, 0o1777)
    name = sticky / "p.db"
    try:
        # The configured name is created first as a link to another file, and
        # SQLite opens the store through it; the link's creator may then
        # replace their own entry in the sticky directory.
        name.symlink_to(planted)
        conn = sqlite3.connect(name)
        try:
            name.unlink()
            name.symlink_to(operators)
            verdict = fetch.store_file_judgement(name)
            rows = conn.execute("SELECT x FROM t").fetchall()
        finally:
            conn.close()
    finally:
        os.chmod(sticky, 0o700)
    assert rows == [("planted",)]
    assert verdict is not None, "the gate trusted a name whose open connection serves another file"


@pytest.mark.parametrize("via", ["file", "directory"])
def test_sqlite_writes_sidecars_beside_the_target_of_a_symlinked_store(
    tmp_path: Path, via: str
) -> None:
    """SQLite puts -wal and -shm beside the resolved store, where they are judged (§1 part 3)."""
    target_dir = _closed_dir(tmp_path / "target")
    target = target_dir / "p.db"
    links = _closed_dir(tmp_path / "links")
    if via == "file":
        name = links / "p.db"
        name.symlink_to(target)
    else:
        (links / "d").symlink_to(target_dir, target_is_directory=True)
        name = links / "d" / "p.db"
    conn = sqlite3.connect(name)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("CREATE TABLE t (x)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        beside_target = {p.name for p in target_dir.iterdir()}
        beside_link = {p.name for p in links.iterdir()}
    finally:
        conn.close()
    assert {"p.db-wal", "p.db-shm"} <= beside_target
    assert not any(n.endswith(("-wal", "-shm", "-journal")) for n in beside_link)


async def _entry_by(session: Any, path: Path, deposited_by: str) -> CorpusEntryRow:
    entry = CorpusEntry(
        entry_id=str(uuid.uuid4()),
        source_type="LOCAL_MARKDOWN",
        uri_r=Path(os.path.abspath(path)).as_uri(),
        fetch_policy=FetchPolicy.LAZY,
        mutability=Mutability.MUTABLE,
        deposited_by=deposited_by,
    )
    row = CorpusEntryRow.from_model(entry)
    session.add(row)
    await session.flush()
    return row


def _config_choosing(directory: Path, root: Path) -> Path:
    config = directory / "config.yaml"
    config.write_text(f"local_refresh:\n  allowed_roots:\n    - {root}\n", encoding="utf-8")
    os.chmod(config, 0o600)
    return config


@pytest.fixture
def loaded_open_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[dict[str, Path]]:
    """A config in a directory others may write, loaded before the store session opens."""
    monkeypatch.setenv("HOME", str(_closed_dir(tmp_path / "home")))
    root = _closed_dir(tmp_path / "chosen-root")
    target = _closed_file(root / "notes.txt", "a host file the config's root admits\n")
    shared = tmp_path / "shared"
    shared.mkdir()
    config = _config_choosing(shared, root)
    os.chmod(shared, 0o777)
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(_closed_dir(tmp_path / "blobs")))
    reset_config()
    yield {"config": config, "target": target, "root": root}
    os.chmod(shared, 0o700)


@pytest.mark.asyncio
async def test_the_config_gate_judges_the_file_the_settings_came_from(
    loaded_open_config: dict[str, Path], db_session: Any
) -> None:
    """Once a refused config is removed, the gate still refuses while its settings are in force."""
    from particles.config import get_config

    config, target = loaded_open_config["config"], loaded_open_config["target"]
    entry = await _entry_by(db_session, target, "operator")
    assert fetch.config_file_judgement() is not None
    assert await fetch._do_local_refresh(db_session, entry, None) is None

    config.unlink()

    assert get_config().local_refresh.allowed_roots == [str(loaded_open_config["root"])]
    found = fetch.config_file_judgement()
    assert found is not None
    assert found[0] == config
    assert found[1].refusal is fetch.RefreshRefusal.CONFIG_WRITABLE_BY_OTHERS
    assert await fetch._do_local_refresh(db_session, entry, None) is None


@pytest.fixture
def loaded_closed_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A config only the operator may write, loaded before the store session opens."""
    monkeypatch.setenv("HOME", str(_closed_dir(tmp_path / "home")))
    root = _closed_dir(tmp_path / "chosen-root")
    target = _closed_file(root / "notes.txt", "a host file the config's root admits\n")
    config = _config_choosing(_closed_dir(tmp_path / "cfg"), root)
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(_closed_dir(tmp_path / "blobs")))
    reset_config()
    return {"config": config, "target": target}


@pytest.mark.asyncio
async def test_a_config_name_that_no_longer_holds_the_loaded_file_is_refused(
    loaded_closed_config: dict[str, Path], db_session: Any
) -> None:
    """The gate passes the closed config it loaded, and refuses once the name holds another file."""
    config, target = loaded_closed_config["config"], loaded_closed_config["target"]
    entry = await _entry_by(db_session, target, "operator")
    assert fetch.config_file_judgement() is None
    assert await fetch._do_local_refresh(db_session, entry, None) is not None

    replacement = _closed_file(config.with_name("new.yaml"))
    os.replace(replacement, config)

    found = fetch.config_file_judgement()
    assert found is not None
    assert "no longer the file this process loaded" in found[1].cause


def test_the_refresh_refuses_a_config_the_doctor_reports_as_a_clones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config inside a git work tree, which the doctor warns about, chooses no host file."""
    from particles.api.cli import hook as hook_mod

    monkeypatch.setenv("HOME", str(_closed_dir(tmp_path / "home")))
    clone = _closed_dir(tmp_path / "clone")
    (clone / ".git").mkdir()
    config = _closed_file(clone / "config.yaml")
    monkeypatch.setenv("PARTICLES_CONFIG", str(config))
    reset_config()
    assert hook_mod._config_pin_problem(str(config)) is not None
    found = fetch.config_file_judgement()
    assert found is not None
    assert found[1].refusal is fetch.RefreshRefusal.CONFIG_WRITABLE_BY_OTHERS
    assert "git work tree" in found[1].cause


@pytest.fixture
def compiled_defaults_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Compiled defaults under a scratch home, loaded before the store session opens."""
    home = _closed_dir(tmp_path / "home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PARTICLES_CONFIG", raising=False)
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(_closed_dir(tmp_path / "blobs")))
    reset_config()
    return home


@pytest.mark.asyncio
async def test_no_effective_uid_reads_no_user_claude_rule_file(
    compiled_defaults_home: Path, db_session: Any
) -> None:
    """With no effective uid and a store with no SQLite file, no host file is read (§4.2)."""
    rule = _closed_file(_closed_dir(compiled_defaults_home / ".claude") / "CLAUDE.md", "# rules\n")
    entry = await _lazy_entry(db_session, rule)
    with _as_uid(None):
        assert fetch.config_file_judgement() is None  # compiled defaults
        assert fetch.cwd_relative_store(db_session) is None  # no SQLite file
        assert await fetch._do_local_refresh(db_session, entry, None) is None
    assert await fetch._do_local_refresh(db_session, entry, None) is not None


# A component swapped between the open and the by-name walk still leaves a
# judged open directory, for the config gate and for a rule read (security
# review 2026-10-30-1, Z64 P4). The swap runs inside the walk's own entry, so
# the interleaving is exact.

_REAL_LOOKUP = operator_only._lookup_directories


def _swap_then_walk(link: Path, new_target: Path) -> Callable[[Path], Any]:
    def walk(path: Path) -> Any:
        link.unlink()
        link.symlink_to(new_target, target_is_directory=True)
        return _REAL_LOOKUP(path)

    return walk


def test_the_config_gate_refuses_when_a_component_is_swapped_after_the_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A component swapped between the inode check and the walk leaves the gate refusing."""
    from particles.config import get_config

    monkeypatch.setenv("HOME", str(_closed_dir(tmp_path / "home")))
    first = _closed_dir(tmp_path / "first")
    _closed_file(first / "config.yaml", "{}\n")
    second = _closed_dir(tmp_path / "second")
    _closed_file(second / "config.yaml", "{}\n")
    shared = tmp_path / "shared"
    shared.mkdir()
    link = shared / "cfg"
    link.symlink_to(first, target_is_directory=True)
    os.chmod(shared, 0o777)
    try:
        monkeypatch.setenv("PARTICLES_CONFIG", str(link / "config.yaml"))
        reset_config()
        get_config()
        with mock.patch.object(operator_only, "_lookup_directories", _swap_then_walk(link, second)):
            verdict = fetch.config_file_judgement()
    finally:
        os.chmod(shared, 0o700)
    assert verdict is not None


def test_a_rule_read_refuses_when_a_component_is_swapped_after_the_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rule file whose path is swapped between its open and the walk is still refused."""
    from particles.corpus.local_read import LocalReadRefused, read_local_file

    monkeypatch.setenv("HOME", str(_closed_dir(tmp_path / "home")))
    first = _closed_dir(tmp_path / "first")
    _closed_file(first / "CLAUDE.md", "x\n")
    second = _closed_dir(tmp_path / "second")
    _closed_file(second / "CLAUDE.md", "x\n")
    shared = tmp_path / "shared"
    shared.mkdir()
    link = shared / "d"
    link.symlink_to(first, target_is_directory=True)
    os.chmod(shared, 0o777)
    try:
        with (
            mock.patch.object(operator_only, "_lookup_directories", _swap_then_walk(link, second)),
            pytest.raises(LocalReadRefused),
        ):
            read_local_file(link / "CLAUDE.md", owner=os.geteuid())
    finally:
        os.chmod(shared, 0o700)
