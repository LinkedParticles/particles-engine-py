# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The engine's per-client keys: format, digest, the key file and its lifecycle.

A key is ``ptk_<role>_<id>_<secret>``: ``<role>`` is ``read``, ``agent`` or
``op``, ``<id>`` is 12 lowercase hex characters, and ``<secret>`` is 32 random
bytes, base64url without padding. The prefix lets a secret scanner match a
leaked key and lets a client read its own key's role; the engine never trusts
it and reads the role from the file.

The file (``api.keys_file``) holds one record per key: id, principal, role,
project, label, creation and revocation times, and the SHA-256 of the secret.
A random 256-bit secret has no dictionary to guess from, so a slow password
hash would add cost to every request and no protection. The file is written
only here, from the ``particles engine keys`` verbs on the engine's host, by a
write to a temporary file renamed into place; no HTTP route writes it. Every
CLI change therefore gives the file a new inode, which is what the reload keys
on. An in-place edit or a ``chmod`` keeps the inode but moves the change time,
which the reload also keys on, and the cache is re-read at least every
:data:`_RELOAD_SECONDS` regardless.

A file that is group- or world-writable, owned by another user, or unparseable
is unusable: :func:`live_keys` raises :class:`KeyFileUnusable` rather than
serve the last good copy, which could hide a revocation. An absent file means
no keys, unless this process has read keys from it, or it is a link to nothing.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import stat
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from particles.config import get_config
from particles.core.observer_scope import (
    canonical_project_key,
    hides_a_character,
    lookalike_reason,
    names_a_project,
)
from particles.corpus.harvest_marks import HARVEST_DEPOSITORS
from particles.corpus.operator_only import (
    AclState,
    NodeFacts,
    acl_line_grants_write,
    gather_write_facts,
    ownership_ignored,
)

log = logging.getLogger(__name__)


class Role(StrEnum):
    """The closed role set. A role is fixed when its key is minted."""

    READER = "reader"
    AGENT = "agent"
    OPERATOR = "operator"


#: The role's spelling in a key's prefix.
ROLE_PREFIX: dict[Role, str] = {Role.READER: "read", Role.AGENT: "agent", Role.OPERATOR: "op"}
_PREFIX_ROLE = {prefix: role for role, prefix in ROLE_PREFIX.items()}

#: What every per-client key starts with.
KEY_PREFIX = "ptk_"

_TOKEN = re.compile(r"^ptk_(read|agent|op)_([0-9a-f]{12})_([A-Za-z0-9_-]{43})$")

#: The longest a cached copy of the file is trusted without a re-read.
_RELOAD_SECONDS = 10.0

#: The longest ``keys create`` / ``keys revoke`` waits for the key file's lock.
#: Each holder keeps it for one read and one rename, so a longer wait means a
#: process that will not let go (security review 2026-10-28-2, AA42).
_LOCK_WAIT_SECONDS = 2.0

#: The key file's format version, so a later field can be read back safely.
_FORMAT_VERSION = 1


class KeyFileUnusable(Exception):
    """The key file exists but cannot be trusted (unsafe permissions, wrong owner, unparseable)."""


class KeyRuleError(ValueError):
    """A ``keys create`` / ``keys revoke`` request the rules refuse."""


@dataclass(frozen=True)
class KeyRecord:
    """One key as the file stores it: never the secret, only its digest."""

    key_id: str
    principal: str
    role: Role
    project: str | None
    label: str | None
    created_at: str
    revoked_at: str | None
    digest: str
    #: Whether ``keys create`` was told ``--reserved-principal`` for a reserved
    #: name. A record under a reserved name without it never authenticates.
    reserved_ok: bool = False
    #: A harvester key's client directories: the Claude Code
    #: ``projects`` directories its memory files lie under, and the
    #: ``<claude_code.state_dir>/projects`` directories its archive files lie
    #: under. ``None`` on a record that names none, which names no ``file://``
    #: URI; never empty.
    memory_roots: tuple[str, ...] | None = None
    archive_roots: tuple[str, ...] | None = None

    @property
    def live(self) -> bool:
        return self.revoked_at is None


def keys_file_path() -> Path:
    """Where the key file lives (``api.keys_file``, ``~`` expanded).

    Raises :class:`KeyFileUnusable` for a relative setting. A relative path
    would be read against whatever directory the engine or the CLI runs in, so
    a key list a cloned repository ships would become the engine's live keys
    (security review 2026-10-26-1, W14). Refused, it counts as an unusable file:
    per-client keys answer 503, the engine is never read as keyless, and
    ``keys create`` writes nothing.
    """
    path = Path(get_config().api.keys_file).expanduser()
    if not path.is_absolute():
        raise KeyFileUnusable(
            f"api.keys_file ({str(path)!r}) is a relative path, which would be read "
            "against the working directory; set it to an absolute path"
        )
    return path


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def parse_token(token: str) -> tuple[Role, str, str] | None:
    """``(role, key_id, secret)`` for a well-formed per-client key, else ``None``."""
    match = _TOKEN.match(token)
    if match is None:
        return None
    return _PREFIX_ROLE[match.group(1)], match.group(2), match.group(3)


def role_from_prefix(token: str) -> Role | None:
    """The role a key's prefix claims, for a client reading its own key (§7). Advisory only."""
    parsed = parse_token(token)
    return parsed[0] if parsed else None


def client_token_warning(
    token: str | None, legacy_key: str | None, *, agent_side: bool
) -> str | None:
    """Why a client's ``PARTICLES_ENGINE_TOKEN`` carries more authority than it should, or ``None``.

    Reads the token's prefix only, which the engine never trusts but a client
    may read for its own key. A token with no per-client prefix
    is the legacy shared key (or not a key at all), which is the operator's.
    An agent-side client (the hooks, the MCP server) holding an operator key
    hands the operator's authority to any process that can read its
    environment, an agent with a shell among them.
    """
    if token is None:
        return None
    if legacy_key is not None and token == legacy_key:
        return (
            "PARTICLES_ENGINE_TOKEN equals this environment's PARTICLES_API_KEY, the "
            "operator's key; give this client its own (particles engine keys create)"
        )
    role = role_from_prefix(token)
    if role is None:
        return (
            "PARTICLES_ENGINE_TOKEN is not a per-client key, so it is the legacy shared "
            "key, which carries the operator's authority; give this client its own "
            "(particles engine keys create)"
        )
    if agent_side and role is Role.OPERATOR:
        return (
            "PARTICLES_ENGINE_TOKEN is an operator key; any process that can read this "
            "environment, an agent with a shell among them, takes every operator action. "
            "Mint an agent key (particles engine keys create --role agent)"
        )
    return None


# --- reading -------------------------------------------------------------------


@dataclass
class _Cache:
    path: Path | None = None
    stamp: tuple[int, ...] | None = None
    read_at: float = 0.0
    records: dict[str, KeyRecord] | None = None


_cache = _Cache()

#: Key files this process has read a live key from. Kept across
#: :func:`reset_key_cache`, so a file that later disappears, or whose last live
#: key is revoked, never turns the running engine keyless.
_seen_with_keys: set[Path] = set()


def reset_key_cache() -> None:
    """Forget the cached file. Tests call it; the engine never needs to."""
    global _cache
    _cache = _Cache()


def forget_key_files() -> None:
    """Forget which key files held keys, so a removed file reads as no keys. Tests only."""
    _seen_with_keys.clear()


def _group_shared_with_others(gid: int) -> bool:
    """Whether a local account other than the engine's (and root) may hold group ``gid``.

    A group-writable key directory is only as private as its group: whoever is
    in the group may rename an older copy of the key file over the current one
    and so undo a revocation (security review 2026-10-28-2, AA49). The group
    counts as shared when the group database names another member or another
    account has it as its primary group. On macOS every group counts as
    shared, since membership there is also computed by nesting (every local
    account holds ``localaccounts`` with no member listed), which the group
    database does not show; a gid neither database lists counts as shared
    too, since its members are accounts this host cannot see, such as a host
    group on a directory bind-mounted into a container (security review
    2026-10-30-1, AB44, AB53).
    """
    import grp
    import pwd

    if sys.platform == "darwin":
        return True
    euid = os.geteuid()
    try:
        me: str | None = pwd.getpwuid(euid).pw_name
    except KeyError:
        me = None
    try:
        accounts = pwd.getpwall()
    except OSError:
        return True
    try:
        members = set(grp.getgrgid(gid).gr_mem)
    except KeyError:
        if not any(a.pw_gid == gid for a in accounts):
            return True
        members = set()
    if members - {me, "root"}:
        return True
    return any(a.pw_gid == gid and a.pw_uid not in {euid, 0} for a in accounts)


def _acl_lets_others_write(node: NodeFacts) -> bool:
    if node.acl is AclState.NONE:
        return False
    if node.acl is not AclState.TEXT or node.uid is None:
        return True
    return any(acl_line_grants_write(line, node.uid) for line in node.acl_text.splitlines()[1:])


def _directory_refusal(node: NodeFacts, *, holding: bool) -> str | None:
    """Why another user may add, remove or rename entries in ``node``'s directory, or None.

    The directory must belong to the engine's uid or root and be writable by no
    one else. A sticky directory closes removal and renaming to each entry's
    owner: above the key file that is enough, but the directory holding it
    must also be the engine's own, since anyone may create the key file's or
    the lock file's name first in a sticky directory others write into
    (security review 2026-10-28-2, AA42). One group-writable shape is allowed:
    a setgid directory of one of the engine's own groups that no other account
    shares, which is what a Kubernetes ``fsGroup`` volume is (the Helm chart's
    ``/data``; AA49). The operator names that group in ``api.keys_dir_group``,
    since nothing inside a container tells a volume only its pods write from a
    host directory a host group shares (security review 2026-10-30-1, AB53).
    A directory on a volume that ignores ownership is refused.
    """
    where = node.path
    if not node.opened or node.mode is None:
        return f"the key directory {where} cannot be read"
    if node.owners_ignored:
        # Every user acts as each file's owner there, so its owner and mode
        # say nothing (security review 2026-10-30-1, AB14, AB50).
        return (
            f"the key directory {where} is on a volume that ignores ownership, where "
            "every local user may write"
        )
    if node.uid not in {os.geteuid(), 0}:
        return (
            f"the key directory {where} is owned by uid {node.uid}, not by this "
            f"engine's uid {os.geteuid()} or root"
        )
    mode = node.mode
    if _acl_lets_others_write(node):
        return f"an ACL entry on the key directory {where} lets another user write into it"
    sticky = bool(mode & stat.S_ISVTX) and (not holding or node.uid == os.geteuid())
    if sticky or not mode & (stat.S_IWGRP | stat.S_IWOTH):
        return None
    if mode & stat.S_IWOTH:
        return f"the key directory {where} is writable by every user; `chmod o-w {where}` fixes it"
    own_groups = {os.getegid(), *os.getgroups()}
    private_gid = get_config().api.keys_dir_group
    if mode & stat.S_ISGID and node.uid is not None and private_gid is not None:
        gid = os.stat(where).st_gid
        if gid == private_gid and gid in own_groups and not _group_shared_with_others(gid):
            return None
    return (
        f"the key directory {where} is writable by a group other accounts share; "
        f"`chmod g-w {where}` fixes it"
    )


def _check_dir_safe(path: Path) -> None:
    """Refuse a key file in a place another user may change.

    Whoever can write the directory holding the key file, or any directory its
    name resolves through, can unlink, rename or replace the file, or swap a
    directory for a link to another one the operator owns, which the file's own
    owner and mode cannot prevent (security review 2026-10-28-1, O50;
    2026-10-28-2, AA8). Every such directory is gathered as the local refresh
    gathers them, up to the first broad root, and judged by
    :func:`_directory_refusal`, and a symlink on the way that a user other
    than the engine's and root owns is refused.
    """
    facts = gather_write_facts(path)
    if not facts.holding.exists:
        return
    refusal = _directory_refusal(facts.holding, holding=True)
    for link in facts.links:
        if refusal is not None:
            break
        if link.uid not in {os.geteuid(), 0}:
            # Its owner may repoint it whatever the directory holding it says,
            # as the refresh's own judge refuses it (security
            # review 2026-10-30-1, AB43, AB51).
            refusal = (
                f"the key file's path runs through the link {link.path}, which uid "
                f"{link.uid} owns, not this engine's uid {os.geteuid()} or root"
            )
    for node in facts.above:
        if refusal is not None:
            break
        refusal = _directory_refusal(node, holding=False)
    if refusal is not None:
        raise KeyFileUnusable(refusal)


def _check_safe(path: Path, st: os.stat_result) -> None:
    if st.st_uid != os.geteuid():
        raise KeyFileUnusable(
            f"{path} is owned by uid {st.st_uid}, not by this engine's uid {os.geteuid()}; "
            f"`chown {os.geteuid()} {path}` (or mint keys as the engine's user) fixes it"
        )
    if st.st_mode & 0o077:
        # Only the mode ``keys create`` writes: a file other users may read or
        # write is one the engine did not write for itself (security review
        # 2026-10-28-2, AA8).
        raise KeyFileUnusable(
            f"{path} is readable or writable by users other than its owner; "
            f"`chmod 600 {path}` fixes it"
        )


def _parse(path: Path, raw: str) -> dict[str, KeyRecord]:
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("version") != _FORMAT_VERSION:
            raise ValueError("unknown or missing format version")
        out: dict[str, KeyRecord] = {}
        for item in data["keys"]:
            if not isinstance(item, dict):
                # A record that is not an object raised AttributeError past
                # this handler, a 500 for every per-client key (security
                # review 2026-10-27-1, G72).
                raise ValueError(f"a key record is a {type(item).__name__}, not an object")
            project = item.get("project")
            reserved_ok = item.get("reserved_ok", False)
            if not isinstance(reserved_ok, bool):
                raise ValueError(f"key {item.get('key_id')!r} has a non-boolean reserved_ok")
            if project is not None and (
                not isinstance(project, str) or not names_a_project(project)
            ):
                # A blank binding is no binding: read as unbound, the key would
                # write as the global project (security review 2026-10-26-1, W71).
                raise ValueError(f"key {item.get('key_id')!r} names a blank project")
            if project is not None:
                if hides_a_character(project):
                    # A binding that renders as another project is refused as
                    # check_mint refuses it (security review 2026-10-28-1, O46).
                    raise ValueError(
                        f"key {item.get('key_id')!r} names a project holding an invisible "
                        "character, or one that reads as another"
                    )
                project = canonical_project_key(project)
            roots = {name: _parse_roots(item, name) for name in _ROOT_FIELDS}
            record = KeyRecord(
                key_id=str(item["key_id"]),
                principal=str(item["principal"]),
                role=Role(item["role"]),
                project=project,
                label=item.get("label"),
                created_at=str(item["created_at"]),
                revoked_at=item.get("revoked_at"),
                digest=str(item["digest"]),
                reserved_ok=reserved_ok,
                memory_roots=roots["memory_roots"],
                archive_roots=roots["archive_roots"],
            )
            out[record.key_id] = record
        return out
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        raise KeyFileUnusable(f"{path} cannot be parsed: {exc}") from exc


def _known_absent(path: Path) -> bool:
    """Whether ``path`` names nothing: no entry, or a missing directory on the way."""
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False
    return False


def _load(path: Path) -> dict[str, KeyRecord]:
    """The file's records, cached on its stat, re-read every :data:`_RELOAD_SECONDS`.

    A file this process has read keys from and that is now absent, or a link
    to nothing, is unusable rather than no keys: read as no keys, it turned an
    engine that required a key for reads keyless (security review
    2026-10-27-1, G37). Deliberately removing every key takes a restart.

    The file is opened without following a link, and its owner, mode and bytes
    are judged on that one descriptor, in a directory no other user may write
    into, through directories no other user may change (:func:`_check_dir_safe`):
    a link planted at the path made the engine read whatever engine-owned file
    it named (security review 2026-10-28-1, O50; 2026-10-28-2, AA8).
    """
    try:
        _check_dir_safe(path)
    except KeyFileUnusable:
        # A file that is not there holds no keys wherever it would be; one that
        # is there is read only from a place no other user may change. A name
        # the engine cannot reach (a directory it cannot search) may hold
        # keys, so it is unusable rather than absent (security review
        # 2026-10-30-1, AB3).
        if path in _seen_with_keys or not _known_absent(path):
            raise
        _cache.path, _cache.stamp, _cache.records = path, None, {}
        return {}
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        if path in _seen_with_keys:
            raise KeyFileUnusable(
                f"{path} held keys and is gone; restore it, or restart the engine to "
                "run with no per-client keys"
            ) from None
        _cache.path, _cache.stamp, _cache.records = path, None, {}
        return {}
    except OSError as exc:
        if path.is_symlink():
            raise KeyFileUnusable(
                f"{path} is a link; the key file is read only where it is, never through one"
            ) from None
        raise KeyFileUnusable(f"{path} cannot be read: {exc}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            # A directory, a FIFO or a device (security review 2026-10-26-1, W76).
            raise KeyFileUnusable(f"{path} is not a regular file")
        if ownership_ignored(fd):
            # Its owner and mode say nothing there (security review
            # 2026-10-30-1, AB14, AB50).
            raise KeyFileUnusable(
                f"{path} is on a volume that ignores ownership, where every local user may write"
            )
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, "rb") as handle:
        # The mode, owner and ctime are in the stamp, and the file is checked on
        # every call: a chmod keeps the inode, size and mtime, and the cache
        # served a file made group-writable for the reload window (security
        # review 2026-10-27-1, G76).
        _check_safe(path, st)
        stamp = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_mode, st.st_uid)
        now = time.monotonic()
        if (
            _cache.path == path
            and _cache.stamp == stamp
            and _cache.records is not None
            and now - _cache.read_at < _RELOAD_SECONDS
        ):
            return _cache.records
        try:
            raw = handle.read().decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # An unreadable file or bytes that are not UTF-8 are as unusable as
            # an unparseable file (security review 2026-10-26-1, W76).
            raise KeyFileUnusable(f"{path} cannot be read: {exc}") from exc
    records = _parse(path, raw)
    if any(record.live for record in records.values()):
        # Only a live key counts: a file whose keys were all revoked while the
        # engine ran must not turn it keyless (security review 2026-10-28-1, O48).
        _seen_with_keys.add(path)
    _cache.path, _cache.stamp, _cache.read_at, _cache.records = path, stamp, now, records
    return records


def all_keys() -> dict[str, KeyRecord]:
    """Every record in the key file, revoked ones included. Raises :class:`KeyFileUnusable`."""
    return _load(keys_file_path())


def live_keys() -> list[KeyRecord]:
    """The unrevoked keys. Raises :class:`KeyFileUnusable`."""
    return [record for record in all_keys().values() if record.live]


def has_live_keys() -> bool:
    """Whether the file holds any unrevoked key. An unusable file counts as holding keys.

    So a file that exists but cannot be used never makes the engine look
    keyless (§6): it neither relaxes the startup bind gate nor
    reopens the keyless read-only mode. Nor does a file whose last live key was
    revoked while this process ran: the engine keeps refusing an unkeyed
    request, and becoming keyless takes a restart (security review
    2026-10-28-1, O48).
    """
    try:
        return bool(live_keys()) or keys_file_path() in _seen_with_keys
    except KeyFileUnusable:
        return True


def has_usable_live_keys() -> bool:
    """Whether the file holds an unrevoked key the engine can check. An unusable file holds none.

    The per-request bounds read this, so a damaged file never lifts them
    (security review 2026-10-26, W9); :func:`has_live_keys` still counts it as
    keys for the startup gate and the keyless write refusal.
    """
    try:
        return bool(live_keys())
    except KeyFileUnusable:
        return False


def lookup(token: str) -> KeyRecord | None:
    """The live record ``token`` authenticates, or ``None``. Raises :class:`KeyFileUnusable`.

    The key id selects the row. It is not secret, so the lookup's timing says
    only whether an id exists. The secret's digest is then compared in constant
    time, which is hygiene here rather than a defence: turning a leaked digest
    prefix into the secret would need a SHA-256 preimage.
    """
    parsed = parse_token(token)
    if parsed is None:
        return None
    role, key_id, secret = parsed
    record = all_keys().get(key_id)
    if record is None or not record.live:
        if record is not None:
            log.warning("engine key %s (%s) is revoked; refusing it", key_id, record.principal)
        return None
    if not hmac.compare_digest(record.digest.encode(), _digest(secret).encode()):
        return None
    if record.role is not role:
        # A key whose prefix disagrees with its record was not minted here.
        return None
    if unacknowledged_reserved(record):
        log.warning(
            "engine key %s holds the reserved name %r with no --reserved-principal on "
            "record; refusing it (revoke it and mint it again)",
            key_id,
            record.principal,
        )
        return None
    refusal = record_rule_refusal(record)
    if refusal is not None:
        log.warning("engine key %s %s; refusing it", key_id, refusal)
        return None
    return record


def record_rule_refusal(record: KeyRecord) -> str | None:
    """The :func:`check_mint` rule a record breaks, for a record written by hand.

    ``keys create`` never writes such a record, but the file is plain JSON, and
    a record ``check_mint`` refuses (an agent under the operator's name, say)
    authenticated as a principal that readers trusting a name take for the
    operator (security review 2026-10-28-1, O79). The reserved-name rule is
    :func:`unacknowledged_reserved`.
    """
    principal = record.principal.strip()
    if not principal or principal != record.principal:
        return "has a blank or padded principal"
    operator_identity = get_config().curation.operator_identity
    if record.role is Role.OPERATOR and principal != operator_identity:
        # check_mint gives every operator key the one operator principal; a
        # record under another name would record operator acts under it
        # (security review 2026-10-28-2, AA65).
        return (
            f"holds the operator role under {principal!r}, not the operator principal "
            f"{operator_identity!r} (curation.operator_identity)"
        )
    if record.role is not Role.OPERATOR and principal in {operator_identity, "operator"}:
        return f"holds the operator's name {principal!r} under the {record.role.value} role"
    if principal == "api-user":
        return "holds the anonymous depositor label 'api-user'"
    if record.project is not None and record.role is not Role.AGENT:
        return f"binds a project under the {record.role.value} role"
    for name in _ROOT_FIELDS:
        roots = getattr(record, name)
        if roots is None:
            continue
        if not is_harvester(record.role, principal):
            return f"records {name} under a key that is not a harvester's"
        if not roots:
            return f"records an empty {name}"
        for root in roots:
            try:
                if normalised_root(root) != root:
                    return f"records {name} entry {root!r}, which is not in normalised form"
            except ValueError as exc:
                return f"records {name} entry {root!r}: {exc}"
    return None


def unacknowledged_reserved(record: KeyRecord) -> bool:
    """Whether ``record`` holds a reserved name that ``keys create`` was never told to mint.

    The reserved-name rule ran at mint only, so a record minted before a name
    was reserved, or written by hand, authenticated as that writer with no
    acknowledgement (security review 2026-10-27-1, G44).
    """
    return (
        record.role is not Role.OPERATOR
        and not record.reserved_ok
        and record.principal in reserved_principals()
    )


# --- minting rules -------------------------------------------------------------


def store_writer_labels() -> frozenset[str]:
    """The labels the store's own writers stamp as ``asserted_by`` or ``deposited_by``.

    Every registered extractor's id, the extraction pipeline's, the interchange
    import's and restore's, consolidation's passes, the trust cascade's, the
    update sweep's, the URL inbox's and the web clipper's. An agent key under one of these names
    would own, under the agent routes' own-beliefs rule, every belief that
    writer made (security review 2026-10-26-1, W5).

    ``mcp.write.asserter_identity`` is deliberately absent: the HTTP MCP
    server's agent key shares that principal with the in-process server, so it
    keeps owning the beliefs either one asserted (the owner's
    ruling).
    """
    # Deferred imports: Engine code this module needs only when a key is minted,
    # never at import or per request.
    from particles.core.conflict_review import CASCADE_ACTOR
    from particles.core.contradiction_disclosure import DISCLOSURE_ACTOR
    from particles.core.reanchor import REANCHOR_ACTOR
    from particles.corpus.deposit import CAPTURE_DEPOSITORS
    from particles.extraction.registry import get_extractors
    from particles.ingest.conflict_plan import PIPELINE_ACTOR
    from particles.interchange.store import IMPORT_ASSERTER, RESTORE_ASSERTER
    from particles.operations.abstraction import ABSTRACTION_ACTOR
    from particles.operations.consolidation import CONSOLIDATION_ACTOR
    from particles.operations.inbox import INBOX_DEPOSITOR
    from particles.operations.reconcile import UPDATE_SWEEP_ACTOR

    return frozenset(
        {plugin.EXTRACTOR_ID for plugin in get_extractors()}
        | {
            PIPELINE_ACTOR,
            IMPORT_ASSERTER,
            RESTORE_ASSERTER,
            CONSOLIDATION_ACTOR,
            DISCLOSURE_ACTOR,
            ABSTRACTION_ACTOR,
            REANCHOR_ACTOR,
            CASCADE_ACTOR,
            UPDATE_SWEEP_ACTOR,
            INBOX_DEPOSITOR,
        }
        | CAPTURE_DEPOSITORS
    )


def reserved_principals() -> frozenset[str]:
    """Labels some reader trusts by name, so no key holds one by accident (rule 5)."""
    # Deferred import: the corpus modules are Engine code this module only needs
    # when a key is minted or resolved, never at import.
    from particles.corpus.deposit import CAPTURE_DEPOSITORS
    from particles.corpus.fetch import LOCAL_LAZY_DEPOSITORS, MEMORY_FILE_DEPOSITORS
    from particles.corpus.harvest_marks import RESERVED_DEPOSITORS

    cfg = get_config()
    return frozenset(
        {
            cfg.curation.operator_identity,
            "operator",
            "api-user",
            cfg.mcp.memory_compat.asserter_identity,
        }
        | RESERVED_DEPOSITORS
        | LOCAL_LAZY_DEPOSITORS
        | MEMORY_FILE_DEPOSITORS
        # The capture rule treats a web capture's depositor as the owner of
        # every capture entry (security review 2026-10-26-1, W67).
        | CAPTURE_DEPOSITORS
        | store_writer_labels()
    )


def check_mint(role: Role, principal: str, project: str | None, *, reserved_ok: bool) -> None:
    """Refuse a key the rules forbid (rule 5, §4). Raises :class:`KeyRuleError`."""
    operator_identity = get_config().curation.operator_identity
    principal = principal.strip()
    if not principal:
        raise KeyRuleError("--principal must name the caller, e.g. mcp:claude-code")
    if role is Role.OPERATOR:
        if principal != operator_identity:
            raise KeyRuleError(
                f"every operator key carries the one operator principal, {operator_identity!r} "
                "(curation.operator_identity); the key id tells devices apart"
            )
    elif principal in {operator_identity, "operator"}:
        raise KeyRuleError(
            f"{principal!r} is the operator's name; only an operator key may hold it"
        )
    if principal == "api-user":
        raise KeyRuleError(
            "'api-user' is the anonymous depositor label and is never a key's principal"
        )
    if role is not Role.OPERATOR and principal in reserved_principals() and not reserved_ok:
        raise KeyRuleError(
            f"{principal!r} is a label some reader trusts by name; pass --reserved-principal "
            "to mint it on purpose"
        )
    if project is not None:
        if role is not Role.AGENT:
            raise KeyRuleError(
                "--project binds an agent key's writes; reader and operator keys take none"
            )
        if not names_a_project(project):
            raise KeyRuleError(
                "--project must name a project; a key with no visible character binds nothing"
            )
        reason = lookalike_reason(project)
        if reason is not None:
            # Such a key renders exactly as another project: a binding no
            # reader can tell from the real one (security review 2026-10-28-1, O46).
            # The refusal names the rule that fired (security review 2026-10-30-1,
            # AB32).
            raise KeyRuleError(f"--project {reason}; type the project's name plainly")


# --- harvester roots -------------------------------------------------

#: The key-record fields that hold a harvester key's client directories.
_ROOT_FIELDS = ("memory_roots", "archive_roots")


def is_harvester(role: Role, principal: str) -> bool:
    """Whether a key with ``role`` under ``principal`` is a harvester's (rule 4)."""
    return role is Role.AGENT and principal in HARVEST_DEPOSITORS


def normalised_root(raw: str) -> str:
    """``raw`` as a recorded root: absolute, no trailing slash, no empty, ``.`` or ``..`` segment.

    Raises :class:`ValueError` for a path that is not absolute, holds a control
    character, or names the filesystem root, so the stored form is the one the
    route compares segment for segment (rule 3).
    """
    if not raw.startswith("/"):
        raise ValueError("a root must be an absolute path")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in raw):
        raise ValueError("a root holds a control character")
    segments = raw.split("/")[1:]
    if segments and segments[-1] == "":
        segments.pop()
    if not segments:
        raise ValueError("the filesystem root is never a harvest root")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError("a root holds an empty, '.' or '..' segment")
    return "/" + "/".join(segments)


def _parse_roots(item: dict[str, object], name: str) -> tuple[str, ...] | None:
    value = item.get(name)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(root, str) for root in value):
        raise ValueError(f"key {item.get('key_id')!r} has a {name} that is not a list of paths")
    return tuple(value)


def _mint_roots(
    role: Role, principal: str, name: str, roots: Sequence[str] | None
) -> tuple[str, ...] | None:
    """The normalised, de-duplicated roots ``keys create`` records under ``name``."""
    if roots is None:
        return None
    flag = "--" + name.removesuffix("s").replace("_", "-")
    if not is_harvester(role, principal):
        raise KeyRuleError(
            f"{flag} names a harvester key's client directories; only an agent key under "
            f"{' or '.join(sorted(HARVEST_DEPOSITORS))} takes it"
        )
    if not roots:
        raise KeyRuleError(f"{flag} needs at least one directory")
    out: list[str] = []
    for raw in roots:
        try:
            root = normalised_root(raw)
        except ValueError as exc:
            raise KeyRuleError(f"{flag} {raw!r}: {exc}") from exc
        if root not in out:
            out.append(root)
    return tuple(out)


# --- writing -------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _serialise(records: dict[str, KeyRecord]) -> str:
    keys = []
    for record in sorted(records.values(), key=lambda r: r.created_at):
        item = asdict(record)
        item["role"] = record.role.value
        for name in _ROOT_FIELDS:
            # Written only when recorded, so a record with no roots reads the
            # same to an older engine as it always did.
            if item[name] is None:
                del item[name]
            else:
                item[name] = list(item[name])
        keys.append(item)
    return json.dumps({"version": _FORMAT_VERSION, "keys": keys}, indent=2) + "\n"


def _write(path: Path, records: dict[str, KeyRecord]) -> None:
    """Replace the file whole: a temporary file in the same directory, renamed into place.

    Refuses to write as a user other than the file's owner, or as root when no
    file exists yet, so a ``keys create`` run as root in a container never
    leaves a file the engine's own user cannot use.
    """
    try:
        st = path.stat()
    except FileNotFoundError:
        st = None
    if st is not None and st.st_uid != os.geteuid():
        raise KeyRuleError(
            f"{path} is owned by uid {st.st_uid}; run this as that user (the engine's), "
            f"not as uid {os.geteuid()}"
        )
    if st is None and os.geteuid() == 0:
        raise KeyRuleError(
            "refusing to create the key file as root; run this as the engine's user "
            "(`docker compose exec` and `kubectl exec` already do)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".engine-keys.", suffix=".tmp")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_serialise(records))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    reset_key_cache()


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on the key file's sibling lock file.

    ``keys create`` and ``keys revoke`` each read the whole file and write it
    back whole, so two run at once lost one of the changes: a revocation could
    be undone by a concurrent mint (security review 2026-10-26-1, W54). Each
    holds this lock from its read through the rename. The lock file is never
    the key file, so the rename does not drop the lock.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # The place is judged before the lock file is made, the lock file must be
    # the engine's own and private, and the wait for it is bounded: a lock file
    # another user made first and holds made `keys revoke` wait forever
    # (security review 2026-10-28-2, AA42).
    _check_dir_safe(path)
    lock = path.with_name(f".{path.name}.lock")
    # O_NOFOLLOW: a link planted at the lock path never makes a mint create
    # or open its target (security review 2026-10-27-1, G77).
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
            raise KeyFileUnusable(
                f"the key file's lock file {lock} is not a private file of this user's; "
                "remove it and run the command again"
            )
        deadline = time.monotonic() + _LOCK_WAIT_SECONDS
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise KeyRuleError(
                        f"another process holds the key file's lock {lock}; run the "
                        "command again once it finishes"
                    ) from None
                time.sleep(0.05)
        yield
    finally:
        os.close(fd)


def mint(
    role: Role,
    principal: str,
    *,
    project: str | None = None,
    label: str | None = None,
    reserved_ok: bool = False,
    memory_roots: Sequence[str] | None = None,
    archive_roots: Sequence[str] | None = None,
) -> tuple[str, KeyRecord]:
    """Create a key and record it. Returns ``(token, record)``; the token is shown once.

    ``memory_roots`` and ``archive_roots`` are a harvester key's client
    directories; a harvester key minted with neither names no
    ``file://`` URI. ``keys create`` fills them with the minting host's own
    roots when no flag names any.
    """
    principal = principal.strip()
    project = canonical_project_key(project) if project is not None else None
    check_mint(role, principal, project, reserved_ok=reserved_ok)
    memory = _mint_roots(role, principal, "memory_roots", memory_roots)
    archive = _mint_roots(role, principal, "archive_roots", archive_roots)
    path = keys_file_path()
    with _file_lock(path):
        reset_key_cache()
        records = dict(_load(path))
        key_id = secrets.token_hex(6)
        while key_id in records:
            key_id = secrets.token_hex(6)
        secret = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
        record = KeyRecord(
            key_id=key_id,
            principal=principal,
            role=role,
            project=project,
            label=label,
            created_at=_now(),
            revoked_at=None,
            digest=_digest(secret),
            reserved_ok=role is not Role.OPERATOR and principal in reserved_principals(),
            memory_roots=memory,
            archive_roots=archive,
        )
        records[key_id] = record
        _write(path, records)
    return f"{KEY_PREFIX}{ROLE_PREFIX[role]}_{key_id}_{secret}", record


def revoke(key_id: str) -> KeyRecord:
    """Stamp ``revoked_at`` on a key. The row stays for the audit trail; its id is never reused."""
    path = keys_file_path()
    with _file_lock(path):
        reset_key_cache()
        records = dict(_load(path))
        record = records.get(key_id)
        if record is None:
            raise KeyRuleError(f"no key {key_id!r} in {path}")
        if not record.live:
            return record
        revoked = replace(record, revoked_at=_now())
        records[key_id] = revoked
        _write(path, records)
    return revoked
