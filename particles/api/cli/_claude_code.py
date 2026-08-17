"""Shared helpers for the Claude Code integration.

Used by ``particles init claude-code`` (``cli/init.py``) and the
``particles hook …`` verbs (``cli/hook.py``). Everything here is deliberately
**pure / store-free** except the hook-log writers (local file I/O): the
settings-merge, transcript-distillation, redaction, and config.yaml-surgeon
functions take data in and hand data back, so the ADR's determinism and
surgicality guarantees are unit-testable without a store.

The leading underscore marks the module as internal to the ``cli`` package
(see ``cli/AGENTS.md`` — same convention as ``_logging.py`` / ``_remote.py``).
"""

from __future__ import annotations

import copy
import json
import logging
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from particles.config import get_config

log = logging.getLogger(__name__)

#: Sentinel substring identifying Particles-owned hook entries in Claude Code
#: settings. ``init`` replaces/removes exactly the entries whose
#: command contains it — the surgicality contract.
HOOK_SENTINEL = "particles hook"

#: Truncate-rotation cap for the hook log: when the JSONL file
#: exceeds this size, it is truncated and restarted (simple, no side-car state).
_HOOK_LOG_MAX_BYTES = 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# State directory + hook log (§1 / §6)
# ---------------------------------------------------------------------------


def state_dir() -> Path:
    """The integration's per-machine state directory (``claude_code.state_dir``)."""
    return Path(get_config().claude_code.state_dir).expanduser()


def hook_log_path() -> Path:
    """Resolved hook-log path: ``claude_code.hook_log_path`` or ``<state_dir>/hooks.jsonl``."""
    configured = get_config().claude_code.hook_log_path
    if configured:
        return Path(configured).expanduser()
    return state_dir() / "hooks.jsonl"


def append_hook_log(record: dict[str, Any]) -> None:
    """Append one JSONL line to the hook log; never raises.

    The log's most important entries are written when the store is unreachable,
    so the writer itself must not become a failure surface — any I/O error is
    swallowed (logged at debug). Truncate-rotation: an over-cap file is
    restarted rather than rolled. Transcript *content* is never logged — only
    counts, identifiers, and error strings belong in ``record``.
    """
    try:
        path = hook_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > _HOOK_LOG_MAX_BYTES:
            path.unlink()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 — the log must never break a hook
        log.debug("hook log write failed", exc_info=True)


def read_hook_log_tail(n: int) -> list[str]:
    """Return the last ``n`` raw JSONL lines of the hook log ([] when absent)."""
    path = hook_log_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[-n:] if n > 0 else []


# ---------------------------------------------------------------------------
# Settings merge — marker-owned, idempotent, surgical
# ---------------------------------------------------------------------------


def build_hook_commands(
    command_base: str, store: str, env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The two hook command strings for a given base + store handle.

    Claude Code runs the hook from the *session's* working directory, which is
    almost never the store's directory (it is routinely a git worktree). Any
    ``env`` pins — resolved to absolute paths at install time — are rendered as
    a leading ``env KEY=VALUE …`` prefix so the store the hook resolves is
    CWD-independent (see :func:`render_env_prefix`). An empty /
    absent ``env`` reproduces the original bare command exactly.
    """
    prefix = render_env_prefix(env)
    return {
        "SessionStart": f"{prefix}{command_base} hook session-start --store {store}",
        "SessionEnd": f"{prefix}{command_base} hook session-end --store {store}",
    }


def render_env_prefix(env: Mapping[str, str] | None) -> str:
    """Render an ``env KEY=VALUE … `` shell prefix (empty string when no vars).

    Values are shell-quoted so paths with spaces survive; keys are assumed to
    be well-formed env-var names (the caller controls them). The trailing space
    makes the prefix concatenate cleanly in front of the command base.
    """
    if not env:
        return ""
    parts = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    return f"env {parts} " if parts else ""


def absolutize_sqlite_dsn(dsn: str, *, base_dir: Path | None = None) -> str:
    """Resolve a CWD-relative SQLite DSN to an absolute one.

    ``sqlite+aiosqlite:///./particles.db`` (three slashes, relative path) is
    resolved against ``base_dir`` (default: the current working directory,
    matching how SQLAlchemy resolves it at runtime) and returned as an absolute
    four-slash DSN. An already-absolute SQLite DSN and any non-SQLite DSN
    (Postgres, …) are returned unchanged — a network DSN is already
    location-independent.
    """
    scheme, sep, path = dsn.partition(":///")
    if not sep or not scheme.startswith("sqlite"):
        return dsn
    file_path = Path(path)
    if file_path.is_absolute():
        return dsn
    base = base_dir if base_dir is not None else Path.cwd()
    abs_path = (base / file_path).resolve()
    return f"{scheme}:///{abs_path}"


def _is_particles_hook(hook: Any) -> bool:
    return isinstance(hook, dict) and HOOK_SENTINEL in str(hook.get("command", ""))


def particles_hook_commands(settings: dict[str, Any]) -> list[str]:
    """Every Particles-owned hook command string currently in ``settings``."""
    found: list[str] = []
    hooks_cfg = settings.get("hooks")
    if not isinstance(hooks_cfg, dict):
        return found
    for groups in hooks_cfg.values():
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                continue
            for hook in group["hooks"]:
                if _is_particles_hook(hook):
                    found.append(str(hook["command"]))
    return found


def strip_particles_hook_entries(settings: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``settings`` with every Particles-owned hook removed.

    Surgical by construction: only hooks whose command string contains
    :data:`HOOK_SENTINEL` are dropped. Matcher groups / event arrays / the
    ``hooks`` key are removed only when *our* removal emptied them — a
    structure that was already empty (or is foreign) is preserved.
    """
    result = copy.deepcopy(settings)
    hooks_cfg = result.get("hooks")
    if not isinstance(hooks_cfg, dict):
        return result

    for event in list(hooks_cfg):
        groups = hooks_cfg[event]
        if not isinstance(groups, list):
            continue
        new_groups: list[Any] = []
        removed_any = False
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                new_groups.append(group)
                continue
            kept = [h for h in group["hooks"] if not _is_particles_hook(h)]
            if len(kept) == len(group["hooks"]):
                new_groups.append(group)
                continue
            removed_any = True
            if kept:
                group = dict(group)
                group["hooks"] = kept
                new_groups.append(group)
            # else: the group held only our hooks — drop it entirely.
        if removed_any and not new_groups:
            del hooks_cfg[event]
        else:
            hooks_cfg[event] = new_groups

    if not hooks_cfg and isinstance(settings.get("hooks"), dict) and settings["hooks"]:
        # Our removal emptied the hooks mapping; a pre-existing empty mapping
        # (settings["hooks"] == {}) is left alone by the condition above.
        del result["hooks"]
    return result


def merge_particles_hook_entries(
    settings: dict[str, Any], commands: dict[str, str]
) -> dict[str, Any]:
    """Merge the Particles-owned hook entries into ``settings`` (replace semantics).

    Strips any existing Particles-owned entries first (re-run = repair/upgrade),
    then appends one matcher-group per event. Everything else is preserved.
    """
    result = strip_particles_hook_entries(settings)
    hooks_cfg = result.setdefault("hooks", {})
    if not isinstance(hooks_cfg, dict):
        raise ValueError(
            "Claude Code settings 'hooks' key is not an object — refusing to rewrite it."
        )
    for event, command in commands.items():
        groups = hooks_cfg.setdefault(event, [])
        if not isinstance(groups, list):
            raise ValueError(
                f"Claude Code settings 'hooks.{event}' is not an array — refusing to rewrite it."
            )
        groups.append({"hooks": [{"type": "command", "command": command}]})
    return result


def render_settings_json(settings: dict[str, Any]) -> str:
    """Stable 2-space serialization for settings files."""
    return json.dumps(settings, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Transcript distillation — deterministic, LLM-free
# ---------------------------------------------------------------------------

#: Preference order for the one-line tool summary: the first of these keys
#: present in a tool_use input is shown (``[tool: Bash — git status]``).
_TOOL_SUMMARY_KEYS = (
    "command",
    "file_path",
    "path",
    "url",
    "pattern",
    "query",
    "description",
    "prompt",
    "skill",
    "title",
)

_MAX_TOOL_SUMMARY_CHARS = 120


def _tool_line(block: dict[str, Any]) -> str:
    name = str(block.get("name", "unknown"))
    tool_input = block.get("input")
    detail = ""
    if isinstance(tool_input, dict):
        for key in _TOOL_SUMMARY_KEYS:
            value = tool_input.get(key)
            if isinstance(value, str) and value.strip():
                detail = " ".join(value.split())
                if len(detail) > _MAX_TOOL_SUMMARY_CHARS:
                    detail = detail[: _MAX_TOOL_SUMMARY_CHARS - 1] + "…"
                break
    return f"[tool: {name} — {detail}]" if detail else f"[tool: {name}]"


def _turn_parts(entry_type: str, content: Any) -> list[str]:
    """Extract the distilled parts of one transcript turn.

    Speaker text blocks are kept verbatim; assistant ``tool_use`` blocks become
    one-line summaries; ``tool_result`` and ``thinking`` blocks are dropped
    (tool results are where payloads and secrets concentrate).
    """
    if isinstance(content, str):
        return [content] if content.strip() else []
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
        elif block_type == "tool_use" and entry_type == "assistant":
            parts.append(_tool_line(block))
    return parts


def distill_transcript(jsonl_text: str, session_id: str | None = None) -> str:
    """Distill a Claude Code transcript JSONL into speaker-turn Markdown.

    Deterministic and LLM-free: user and assistant text turns
    verbatim under ``## User`` / ``## Assistant`` headings, tool calls elided
    to one-line summaries, tool results and thinking blocks dropped, malformed
    lines skipped. The output format is **stable input** for a future
    conversation-aware extractor (§ Composition point) — change it
    only with a distiller version story.
    """
    out: list[str] = []
    if session_id:
        out.append(f"# Claude Code session {session_id}")
    for raw_line in jsonl_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            entry = json.loads(raw_line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue
        entry_type = entry.get("type")
        if entry_type not in ("user", "assistant"):
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        parts = _turn_parts(str(entry_type), message.get("content"))
        if not parts:
            continue
        speaker = "User" if entry_type == "user" else "Assistant"
        out.append(f"## {speaker}")
        out.append("\n\n".join(parts))
    return ("\n\n".join(out) + "\n") if out else ""


# ---------------------------------------------------------------------------
# Redaction — best-effort defence in depth, not a guarantee
# ---------------------------------------------------------------------------

_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    # PEM blocks first (they may contain substrings the later patterns match).
    (
        re.compile(r"-----BEGIN [A-Z0-9 ]+-----.*?-----END [A-Z0-9 ]+-----", re.DOTALL),
        "[REDACTED PEM BLOCK]",
    ),
    # Bearer headers before generic keys so the header keeps its label.
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer [REDACTED]"),
    # Anthropic / OpenAI-style secret keys.
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "[REDACTED API KEY]"),
    # AWS access key IDs.
    (re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}\b"), "[REDACTED AWS KEY]"),
]


def redact_secrets(text: str) -> str:
    """Mask common credential shapes.

    Deterministic pattern pass over the distilled transcript before it reaches
    the corpus: ``sk-…`` keys, AWS access key IDs, ``Bearer`` headers, PEM
    blocks. Best-effort defence in depth — stated as such in the docs.
    """
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# MEMORY.md projection paths + manifest (§3/§6/§7)
# ---------------------------------------------------------------------------

#: The one projected region name in MEMORY.md. Also the manifest
#: ``name:`` and the render-snapshot filename stem, per the
#: ``snapshot_path_for`` convention relocated into the state directory.
MEMORY_REGION = "memory-index"

#: Filename of the append-only fold-and-archive target in the state directory.
MEMORY_ARCHIVE_NAME = "MEMORY.archive.md"

#: Every fold-and-archive pointer line in MEMORY.md starts with this. It is
#: machine-generated (like the sentinels), so the pre-deposit filter drops it
#: — which is what keeps the §2 fixed point exact once fold-and-archive has
#: converged the file to region + pointer.
ARCHIVE_POINTER_PREFIX = "*Older agent memories were consolidated"


def memory_manifest_path() -> Path:
    """The projection manifest path: ``agent_memory.projection.manifest`` or
    ``<state_dir>/memory.yaml``."""
    configured = get_config().agent_memory.projection.manifest
    if configured:
        return Path(configured).expanduser()
    return state_dir() / "memory.yaml"


def memory_snapshot_path() -> Path:
    """The render snapshot: ``<state_dir>/memory-index.snapshot.md``.

    Because the bullets renderer is deterministic, snapshot ≡ last-spliced
    region body — the pristine test at harvest and the drift check before a
    re-splice both compare against this file.
    """
    return state_dir() / f"{MEMORY_REGION}.snapshot.md"


def memory_archive_path() -> Path:
    """The fold-and-archive target: ``<state_dir>/MEMORY.archive.md``."""
    return state_dir() / MEMORY_ARCHIVE_NAME


def memory_backup_path() -> Path:
    """The one-deep pre-splice backup: ``<state_dir>/MEMORY.md.pre-render``."""
    return state_dir() / "MEMORY.md.pre-render"


def projection_enabled() -> bool:
    """Whether the MEMORY.md projection is active: enabled in config *and* the
    manifest exists (``particles init claude-code`` writes it)."""
    return get_config().agent_memory.projection.enabled and memory_manifest_path().is_file()


def default_memory_manifest_text() -> str:
    """The zero-config default ``memory.yaml`` — one ranked list.

    A standard manifest: the operator may edit it, add sections, pin
    with ``select.allow/deny``, and register it in
    ``curation.projection_manifests``. The numeric defaults mirror
    ``agent_memory.projection`` at generation time; afterwards the manifest is
    the single tuning surface (it wins where both speak).
    """
    proj = get_config().agent_memory.projection
    return (
        "# MEMORY.md projection manifest — written by\n"
        "# `particles init claude-code`; yours to edit. Add sections, pin claims\n"
        "# with select.allow/deny, or tighten the floor. Docs:\n"
        "# docs/user-guide/claude-code.md § The MEMORY.md projection.\n"
        f"name: {MEMORY_REGION}\n"
        "sections:\n"
        '  - title: "Memory index"\n'
        "    query: null            # no semantic refinement — rank purely by eff. conf.\n"
        "    top_k: 60\n"
        f"    min_confidence: {proj.min_confidence:.2f}   # noise floor (the R1.8 lesson)\n"
        "    render: bullets        # deterministic ranked bullets — never LLM prose\n"
        f"max_lines: {proj.max_lines}             "
        "# document budget — headroom under the 200-line load cap\n"
        "max_bytes: 16384\n"
    )


def load_projection_snapshots() -> dict[str, str]:
    """Region-name → last-rendered-body map from the state directory's snapshots.

    Reads every ``*.snapshot.md`` beside the manifest state (the ``<name>.snapshot.md`` convention, relocated); the stem
    before ``.snapshot`` is the region name. Unreadable files are skipped —
    the strip then treats their regions as dirtied, which routes the content
    through the harvest ladder rather than risk dropping an edit.
    """
    snapshots: dict[str, str] = {}
    directory = state_dir()
    if not directory.is_dir():
        return snapshots
    for path in sorted(directory.glob("*.snapshot.md")):
        try:
            snapshots[path.name[: -len(".snapshot.md")]] = path.read_text(encoding="utf-8")
        except OSError:
            log.debug("could not read snapshot %s", path, exc_info=True)
    return snapshots


# ---------------------------------------------------------------------------
# Memory-file pre-deposit filter — the sentinel strip
# ---------------------------------------------------------------------------


def filter_memory_file_for_deposit(
    text: str, snapshot_bodies: Mapping[str, str] | None = None
) -> str:
    """Pre-deposit filter for harvested memory files.

    Strips every **pristine** ``PROJECTED`` sentinel region — one whose body
    still matches its render snapshot — before deposit: the corpus must never
    contain the store's own rendered output (belt 1 of the round-trip
    contract; the fixed-point test rests on it). A **dirtied** region (body ≠
    snapshot, or no snapshot known) is human/agent signal: its body is kept
    and deposited as ordinarily-authored input for the §6.6 ladder; only the sentinel comment lines are dropped.

    The fold-and-archive pointer line is dropped too — it is
    machine-generated, like the sentinels, and the archive file it points at
    is harvested in full anyway.

    ``snapshot_bodies`` (region name → body) defaults to the state
    directory's ``*.snapshot.md`` files. The parse regexes are shared with
    the renderer (``particles.render.markdown``), so strip and splice can
    never disagree about where a region begins.
    """
    from particles.render.markdown import strip_projected_regions_for_deposit

    if snapshot_bodies is None:
        snapshot_bodies = load_projection_snapshots()
    stripped = strip_projected_regions_for_deposit(text, snapshot_bodies)
    if ARCHIVE_POINTER_PREFIX in stripped:
        kept = [
            line for line in stripped.splitlines() if not line.startswith(ARCHIVE_POINTER_PREFIX)
        ]
        stripped = "\n".join(kept) + ("\n" if stripped.endswith("\n") else "")
    return stripped


# ---------------------------------------------------------------------------
# Digest byte budget
# ---------------------------------------------------------------------------


def truncate_on_line_boundary(text: str, max_bytes: int) -> str:
    """Enforce the ``claude_code.digest_max_bytes`` budget (0 = no cap).

    Truncates on a line boundary and appends a disclosed footer; the footer is
    budgeted for, so the result (footer included) never exceeds ``max_bytes``
    unless the budget is too small to hold the footer alone.
    """
    if max_bytes <= 0 or len(text.encode("utf-8")) <= max_bytes:
        return text
    footer = f"\n\n*[digest truncated at {max_bytes} bytes — see `particles query` for the rest]*\n"
    budget = max_bytes - len(footer.encode("utf-8"))
    kept: list[str] = []
    used = 0
    for line in text.splitlines(keepends=True):
        line_bytes = len(line.encode("utf-8"))
        if used + line_bytes > budget:
            break
        kept.append(line)
        used += line_bytes
    return "".join(kept).rstrip("\n") + footer


# ---------------------------------------------------------------------------
# config.yaml surgeon (fresh-install store auto-create)
# ---------------------------------------------------------------------------
#
# The store-enable edit must parse-preserve-append: parse, preserve everything
# else, append the two keys, never clobber. PyYAML cannot round-trip comments
# or formatting, so the edit is *textual* (block-scoped line surgery) and then
# **verified**: the edited text is re-parsed and compared against the expected
# tree (the old tree plus exactly the intended modifications). Any structure
# the surgeon does not confidently understand raises ConfigEditError instead
# of guessing — an unparseable or unusual config is an error with
# instructions, never a rewrite.

_INIT_MARKER_COMMENT = "# Added by `particles init claude-code`"


class ConfigEditError(ValueError):
    """The config.yaml edit could not be performed safely (never-clobber rule)."""


def _parse_yaml_mapping(yaml_text: str) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(yaml_text) if yaml_text.strip() else None
    except yaml.YAMLError as exc:
        raise ConfigEditError(f"config.yaml is not parseable YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigEditError("config.yaml top level is not a mapping")
    return loaded


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _top_level_key_line(lines: list[str], key: str) -> int | None:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(#.*)?$")
    for i, line in enumerate(lines):
        if pattern.match(line):
            return i
    return None


def _block_end(lines: list[str], key_idx: int) -> int:
    """Index one past the last line belonging to the block opened at ``key_idx``.

    A comment line indented at or left of the key terminates the block (it may
    describe whatever follows, so it must never be swept up by a block
    deletion); a deeper-indented comment belongs to the block. Trailing blank
    lines are excluded either way.
    """
    key_indent = _indent_of(lines[key_idx])
    end = key_idx + 1
    for i in range(key_idx + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        if _indent_of(line) <= key_indent:
            break
        end = i + 1
    return end


def _child_key_line(lines: list[str], parent_idx: int, key: str) -> int | None:
    """Find ``key:`` as a block-mapping child of the key opened at ``parent_idx``."""
    pattern = re.compile(rf"^(\s+){re.escape(key)}:\s*(#.*)?$")
    for i in range(parent_idx + 1, _block_end(lines, parent_idx)):
        if pattern.match(lines[i]):
            return i
    return None


def _child_indent(lines: list[str], key_idx: int, default: int) -> int:
    """Indent of the first effective child of the block at ``key_idx`` (or default)."""
    for i in range(key_idx + 1, _block_end(lines, key_idx)):
        line = lines[i]
        if line.strip() and not line.lstrip().startswith("#"):
            return _indent_of(line)
    return default


def _verify(new_text: str, expected: dict[str, Any]) -> str:
    reparsed = _parse_yaml_mapping(new_text)
    if reparsed != expected:
        raise ConfigEditError(
            "the edited config.yaml did not verify against the expected result — "
            "the file's layout is unusual; make the change by hand instead"
        )
    return new_text


def _insert_after(lines: list[str], idx: int, new_lines: list[str]) -> list[str]:
    return lines[: idx + 1] + new_lines + lines[idx + 1 :]


def _expected_after_enable(tree: dict[str, Any], handle: str, dsn: str) -> dict[str, Any]:
    expected = copy.deepcopy(tree)
    expected.setdefault("storage", {}).setdefault("stores", {})[handle] = dsn
    write = expected.setdefault("mcp", {}).setdefault("write", {})
    enabled = write.setdefault("enabled_stores", [])
    if handle not in enabled:
        enabled.append(handle)
    return expected


def enable_memory_store_text(yaml_text: str, handle: str, dsn: str) -> str:
    """Append ``storage.stores.<handle>`` + ``mcp.write.enabled_stores`` to config.yaml text.

    Parse-preserve-append: existing content — including comments
    and formatting — is untouched; only the two keys are added, and the result
    is verified by re-parse against the expected tree. Raises
    :class:`ConfigEditError` when the file is unparseable or shaped in a way
    the surgeon cannot edit safely (flow-style sections, aliases, …); the
    caller turns that into an actionable message, never a rewrite.
    """
    tree = _parse_yaml_mapping(yaml_text)

    storage = tree.get("storage")
    stores = storage.get("stores") if isinstance(storage, dict) else None
    if isinstance(stores, dict) and handle in stores and stores[handle] != dsn:
        raise ConfigEditError(
            f"store {handle!r} is already configured with a different DSN "
            f"({stores[handle]!r}); refusing to overwrite it"
        )

    expected = _expected_after_enable(tree, handle, dsn)
    if tree == expected:
        return yaml_text  # both keys already present — idempotent no-op

    lines = yaml_text.splitlines()

    # -- storage.stores.<handle> ------------------------------------------
    if not (isinstance(stores, dict) and stores.get(handle) == dsn):
        if "storage" not in tree:
            lines += [
                "",
                _INIT_MARKER_COMMENT,
                "storage:",
                "  stores:",
                f"    {handle}: {dsn}",
            ]
        else:
            storage_idx = _top_level_key_line(lines, "storage")
            if storage_idx is None:
                raise ConfigEditError("could not locate the 'storage:' block")
            stores_idx = _child_key_line(lines, storage_idx, "stores")
            if stores_idx is None:
                if isinstance(stores, dict) and stores:
                    raise ConfigEditError("could not locate the 'storage.stores:' block")
                child = _child_indent(lines, storage_idx, 2)
                lines = _insert_after(
                    lines,
                    storage_idx,
                    [
                        f"{' ' * child}stores:  {_INIT_MARKER_COMMENT}",
                        f"{' ' * (child + 2)}{handle}: {dsn}",
                    ],
                )
            else:
                item = _child_indent(lines, stores_idx, _indent_of(lines[stores_idx]) + 2)
                lines = _insert_after(
                    lines,
                    stores_idx,
                    [f"{' ' * item}{handle}: {dsn}  {_INIT_MARKER_COMMENT}"],
                )

    # -- mcp.write.enabled_stores ------------------------------------------
    mcp = tree.get("mcp")
    write = mcp.get("write") if isinstance(mcp, dict) else None
    enabled = write.get("enabled_stores") if isinstance(write, dict) else None
    if not (isinstance(enabled, list) and handle in enabled):
        if "mcp" not in tree:
            lines += [
                "",
                _INIT_MARKER_COMMENT,
                "mcp:",
                "  write:",
                "    enabled_stores:",
                f"      - {handle}",
            ]
        else:
            mcp_idx = _top_level_key_line(lines, "mcp")
            if mcp_idx is None:
                raise ConfigEditError("could not locate the 'mcp:' block")
            write_idx = _child_key_line(lines, mcp_idx, "write")
            if write_idx is None:
                if isinstance(write, dict) and write:
                    raise ConfigEditError("could not locate the 'mcp.write:' block")
                child = _child_indent(lines, mcp_idx, 2)
                lines = _insert_after(
                    lines,
                    mcp_idx,
                    [
                        f"{' ' * child}write:  {_INIT_MARKER_COMMENT}",
                        f"{' ' * (child + 2)}enabled_stores:",
                        f"{' ' * (child + 4)}- {handle}",
                    ],
                )
            else:
                lines = _add_enabled_store_item(lines, write_idx, handle)

    return _verify("\n".join(lines) + "\n", expected)


def _enabled_stores_line(lines: list[str], write_idx: int) -> int | None:
    """Locate the ``enabled_stores`` key line inside the ``write:`` block."""
    pattern = re.compile(r"^\s+enabled_stores:")
    for i in range(write_idx + 1, _block_end(lines, write_idx)):
        if pattern.match(lines[i]):
            return i
    return None


def _add_enabled_store_item(lines: list[str], write_idx: int, handle: str) -> list[str]:
    """Add ``handle`` to ``mcp.write.enabled_stores`` under the ``write:`` block."""
    key_idx = _enabled_stores_line(lines, write_idx)
    if key_idx is None:
        # No enabled_stores key inside write: — add one.
        child = _child_indent(lines, write_idx, _indent_of(lines[write_idx]) + 2)
        return _insert_after(
            lines,
            write_idx,
            [
                f"{' ' * child}enabled_stores:  {_INIT_MARKER_COMMENT}",
                f"{' ' * (child + 2)}- {handle}",
            ],
        )
    m = re.match(r"^(\s+)enabled_stores:\s*(.*?)\s*(#.*)?$", lines[key_idx])
    assert m is not None  # _enabled_stores_line matched the prefix
    indent, value, comment = m.group(1), m.group(2), m.group(3) or ""
    if value:  # flow style: enabled_stores: [] or [a, b]
        try:
            flow = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise ConfigEditError("could not parse the 'enabled_stores' value") from exc
        if not isinstance(flow, list):
            raise ConfigEditError("'enabled_stores' is not a list")
        flow.append(handle)
        rendered = "[" + ", ".join(str(x) for x in flow) + "]"
        suffix = f"  {comment}" if comment else f"  {_INIT_MARKER_COMMENT}"
        lines[key_idx] = f"{indent}enabled_stores: {rendered}{suffix}"
        return lines
    # Block style: append an item at the end of the block (matching the
    # expected tree's list-append), at the existing items' indent if any.
    item = _child_indent(lines, key_idx, len(indent) + 2)
    return _insert_after(
        lines, _block_end(lines, key_idx) - 1, [f"{' ' * item}- {handle}  {_INIT_MARKER_COMMENT}"]
    )


def fresh_config_text(handle: str, dsn: str) -> str:
    """A brand-new config.yaml enabling the auto-created memory store."""
    return (
        "# Created by `particles init claude-code`.\n"
        f"# The `{handle}` store holds agent memory harvested from Claude Code sessions.\n"
        "storage:\n"
        "  stores:\n"
        f"    {handle}: {dsn}\n"
        "\n"
        "mcp:\n"
        "  write:\n"
        "    enabled_stores:\n"
        f"      - {handle}\n"
    )


def _expected_after_disable(tree: dict[str, Any], handle: str) -> dict[str, Any]:
    expected = copy.deepcopy(tree)
    storage = expected.get("storage")
    if isinstance(storage, dict) and isinstance(storage.get("stores"), dict):
        storage["stores"].pop(handle, None)
        if not storage["stores"]:
            del storage["stores"]
        if not storage:
            del expected["storage"]
    mcp = expected.get("mcp")
    write = mcp.get("write") if isinstance(mcp, dict) else None
    if isinstance(mcp, dict) and isinstance(write, dict):
        enabled = write.get("enabled_stores")
        if isinstance(enabled, list) and handle in enabled:
            remaining = [x for x in enabled if x != handle]
            if remaining:
                write["enabled_stores"] = remaining
            else:
                del write["enabled_stores"]
            if not write:
                del mcp["write"]
            if not mcp:
                del expected["mcp"]
    return expected


def _block_is_effectively_empty(lines: list[str], key_idx: int) -> bool:
    """True when the block opened at ``key_idx`` holds only blanks / comments."""
    for i in range(key_idx + 1, _block_end(lines, key_idx)):
        line = lines[i]
        if line.strip() and not line.lstrip().startswith("#"):
            return False
    return True


def _delete_block_if_empty(lines: list[str], key_idx: int) -> bool:
    """Delete the key line (plus any orphaned marker comments in its block) if empty."""
    if not _block_is_effectively_empty(lines, key_idx):
        return False
    del lines[key_idx : _block_end(lines, key_idx)]
    return True


def disable_memory_store_text(yaml_text: str, handle: str) -> str:
    """Reverse :func:`enable_memory_store_text` — remove the two keys, keep the rest.

    Removes ``storage.stores.<handle>`` and the ``<handle>`` item of
    ``mcp.write.enabled_stores``, pruning containers *that the removal
    emptied* and any marker comment lines the installer added. The surgery is
    block-scoped (never a whole-file pattern sweep) and verified by re-parse
    against the expected tree; raises :class:`ConfigEditError` when it cannot
    be performed safely — the caller prints instructions, never rewrites.
    """
    tree = _parse_yaml_mapping(yaml_text)
    expected = _expected_after_disable(tree, handle)
    if expected == tree:
        return yaml_text  # nothing to remove

    lines = yaml_text.splitlines()

    # -- storage.stores.<handle> ------------------------------------------
    storage = tree.get("storage")
    stores = storage.get("stores") if isinstance(storage, dict) else None
    if isinstance(stores, dict) and handle in stores:
        storage_idx = _top_level_key_line(lines, "storage")
        if storage_idx is None:
            raise ConfigEditError("could not locate the 'storage:' block")
        stores_idx = _child_key_line(lines, storage_idx, "stores")
        if stores_idx is None:
            raise ConfigEditError("could not locate the 'storage.stores:' block")
        entry_re = re.compile(rf"^\s+{re.escape(handle)}:\s+\S")
        entry_idx = next(
            (
                i
                for i in range(stores_idx + 1, _block_end(lines, stores_idx))
                if entry_re.match(lines[i])
            ),
            None,
        )
        if entry_idx is None:
            raise ConfigEditError(f"could not locate the {handle!r} store entry")
        del lines[entry_idx]
        if _delete_block_if_empty(lines, stores_idx):
            storage_idx = _top_level_key_line(lines, "storage")
            if storage_idx is not None:
                _delete_block_if_empty(lines, storage_idx)

    # -- mcp.write.enabled_stores ------------------------------------------
    mcp = tree.get("mcp")
    write = mcp.get("write") if isinstance(mcp, dict) else None
    enabled = write.get("enabled_stores") if isinstance(write, dict) else None
    if isinstance(enabled, list) and handle in enabled:
        mcp_idx = _top_level_key_line(lines, "mcp")
        if mcp_idx is None:
            raise ConfigEditError("could not locate the 'mcp:' block")
        write_idx = _child_key_line(lines, mcp_idx, "write")
        if write_idx is None:
            raise ConfigEditError("could not locate the 'mcp.write:' block")
        key_idx = _enabled_stores_line(lines, write_idx)
        if key_idx is None:
            raise ConfigEditError("could not locate 'mcp.write.enabled_stores'")
        m = re.match(r"^(\s+)enabled_stores:\s*(.*?)\s*(#.*)?$", lines[key_idx])
        assert m is not None
        indent, value = m.group(1), m.group(2)
        if value:  # flow style
            try:
                flow = yaml.safe_load(value)
            except yaml.YAMLError as exc:
                raise ConfigEditError("could not parse the 'enabled_stores' value") from exc
            if not isinstance(flow, list):
                raise ConfigEditError("'enabled_stores' is not a list")
            remaining = [x for x in flow if x != handle]
            if remaining:
                rendered = "[" + ", ".join(str(x) for x in remaining) + "]"
                lines[key_idx] = f"{indent}enabled_stores: {rendered}"
            else:
                del lines[key_idx]
        else:  # block style: delete the "- handle" item within this block
            item_re = re.compile(rf"^\s+-\s+{re.escape(handle)}\s*(#.*)?$")
            item_idx = next(
                (
                    i
                    for i in range(key_idx + 1, _block_end(lines, key_idx))
                    if item_re.match(lines[i])
                ),
                None,
            )
            if item_idx is None:
                raise ConfigEditError(f"could not locate the {handle!r} enabled_stores item")
            del lines[item_idx]
            _delete_block_if_empty(lines, key_idx)
        # Prune write:/mcp: if the removal emptied them (re-locate: indices moved).
        mcp_idx = _top_level_key_line(lines, "mcp")
        if mcp_idx is not None:
            write_idx = _child_key_line(lines, mcp_idx, "write")
            if write_idx is not None and _delete_block_if_empty(lines, write_idx):
                mcp_idx = _top_level_key_line(lines, "mcp")
            if mcp_idx is not None:
                _delete_block_if_empty(lines, mcp_idx)

    # Drop full-line marker comments the installer added, and inline marker
    # suffixes on lines that survived (e.g. a `stores:` opener the operator
    # later populated with their own entries).
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == _INIT_MARKER_COMMENT or stripped.startswith(
            "# Created by `particles init claude-code`"
        ):
            continue
        # The fresh-file header's second line describes the removed store.
        if stripped.startswith(f"# The `{handle}` store holds agent memory"):
            continue
        cleaned.append(line.replace(f"  {_INIT_MARKER_COMMENT}", ""))

    # Collapse any doubled blank lines the pruning left behind.
    pruned: list[str] = []
    for line in cleaned:
        if not line.strip() and pruned and not pruned[-1].strip():
            continue
        pruned.append(line)
    while pruned and not pruned[-1].strip():
        pruned.pop()

    return _verify(("\n".join(pruned) + "\n") if pruned else "", expected)
