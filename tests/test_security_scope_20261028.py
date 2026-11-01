# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the observer-scope and agent-route findings of security review 2026-10-28-1.

E21: an entry keyed under a project key the current rule refuses fails closed.
G65, W53, W72, E9: a project key a reader cannot tell from another is refused
wherever a key is accepted, mint included.
B21: the published MCP manifest names the store an unpinned server serves.

A scratch store and key file under ``tmp_path``; no model is called.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from particles.core.observer_scope import (
    EntryScope,
    canonical_project_key,
    classify_entry,
    hides_a_character,
    names_a_project,
    scope_of_entries,
    visible,
)
from tests._upstream import upstream_only

_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# E21, G65: project keys
# --------------------------------------------------------------------------- #

#: Characters an earlier rule accepted and the current one refuses: a combining
#: mark, a control, a private-use and an unassigned code point, and two blank
#: glyphs.
_NOW_REFUSED = ["\u0301", "\u0001", "\ue000", "\U000e0fff", "\u3164", "\U0001d159"]


@pytest.mark.parametrize("key", _NOW_REFUSED)
def test_an_entry_keyed_under_a_refused_key_never_becomes_global(key: str) -> None:
    """E21: an entry whose only project tag names no project stays out of other projects."""
    tags = [f"project:{key}"]
    harness_tags = ("claude-code",)
    assert not names_a_project(key)
    assert classify_entry(tags, harness_tags) is EntryScope.UNATTRIBUTED
    assert not visible(scope_of_entries([tags], harness_tags), "alpha")


def test_an_entry_with_a_valid_key_beside_a_refused_one_stays_keyed() -> None:
    tags = ["project:alpha", "project:alpha​"]
    assert classify_entry(tags, ()) is EntryScope.KEYED


def test_a_keyless_hand_deposit_is_still_global() -> None:
    assert classify_entry(["web"], ("claude-code",)) is EntryScope.GLOBAL


def _accepted_as(key: str) -> str | None:
    """The project an accept site records for ``key``: ``None`` when it names none or is refused."""
    if not names_a_project(key) or hides_a_character(key):
        return None
    return canonical_project_key(key)


@pytest.mark.parametrize(
    "lookalike",
    [
        "alpha\u200b",  # a zero-width space after a visible key
        "alpha\ufe0f",  # a variation selector
        "alpha\u3164",  # a Hangul filler after a visible key
        "alph\u0430",  # a Cyrillic a for the Latin one
        "\uff41lpha",  # a fullwidth a, NFKC-equal to 'alpha'
        "alpha\U0001d159",  # a blank-rendering symbol after a visible key
    ],
)
def test_a_project_key_that_reads_as_another_is_one_project_or_refused(lookalike: str) -> None:
    """Two keys a reader cannot tell apart never name two projects."""
    assert _accepted_as("alpha") == "alpha"
    assert _accepted_as(lookalike) in (None, "alpha")


def test_a_decomposed_accent_is_the_composed_project() -> None:
    assert _accepted_as("cafe\u0301") == "caf\u00e9"


def test_a_key_that_renders_blank_names_no_project() -> None:
    assert not names_a_project("\U0001d159")  # MUSICAL SYMBOL NULL NOTEHEAD


@pytest.mark.parametrize(
    "key",
    [
        "alpha",
        "-Users-u-src-particles-engine-py",
        "my project",
        "café",
        "日本語のプロジェクト",
        "proj_1.2",
    ],
)
def test_a_plain_key_in_one_script_still_names_a_project(key: str) -> None:
    assert _accepted_as(key) == key


@pytest.mark.parametrize("lookalike", ["alph\u0430", "\uff41lpha", "alpha\u3164"])
def test_mint_refuses_a_binding_that_reads_as_another(lookalike: str) -> None:
    from particles.api import _keys
    from particles.api._keys import KeyRuleError, Role

    with pytest.raises(KeyRuleError):
        _keys.check_mint(Role.AGENT, "agent:laptop", lookalike, reserved_ok=False)


def test_mint_refuses_a_binding_padded_with_invisible_characters() -> None:
    from particles.api import _keys
    from particles.api._keys import KeyRuleError, Role

    with pytest.raises(KeyRuleError):
        _keys.check_mint(Role.AGENT, "agent:laptop", "alpha​", reserved_ok=False)


# --------------------------------------------------------------------------- #
# B21: the store an unpinned stdio server serves
# --------------------------------------------------------------------------- #


@upstream_only
def test_the_mcp_manifest_names_the_store_an_unpinned_server_serves() -> None:
    manifest = json.loads(
        (_ROOT / "publish" / "overlays" / "particles-engine-py" / "server.json").read_text(
            encoding="utf-8"
        )
    )
    (package,) = manifest["packages"]
    (database_url,) = [v for v in package["environmentVariables"] if v["name"] == "DATABASE_URL"]
    text = database_url["description"]
    assert "the store `~/.particles/config.yaml` names" in text
    assert "`~/.particles/particles.db` when that file names none" in text
