# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""One project, one key, at every accept and compare site (security review 2026-10-28-2).

AA24: a key mixing scripts (a Devanagari digit after Latin letters, Old Italic
beside Old Persian) or rendering as an ASCII key (a Unicode hyphen or minus
sign) is never minted. AA25: a tag in another normalisation form is the
project its NFC spelling names, and an operator deposit stores it in NFC.
AA26, AA48: a project key the mint accepts is accepted as a deposit tag.
AA47, AA61: /session-exposures applies the project-key rule. A scratch
store and key file under tmp_path; configuration only through the loader;
no model call, no outside connection.
"""

from __future__ import annotations

import asyncio
import json
import unicodedata
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

LEGACY = "project-keys-legacy-operator-key"


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str = "") -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("network:\n  egress: none\n" + body, encoding="utf-8")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))


def _create_tables() -> None:
    async def _go() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await get_engine().dispose()

    asyncio.run(_go())


def _client() -> TestClient:
    from particles.api.app import app

    return TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    from particles.api import _keys
    from particles.config import reset_config

    keydir = tmp_path / "keys"
    keydir.mkdir(mode=0o700)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 's6.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.setenv("PARTICLES_API_KEY", LEGACY)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    _write_config(tmp_path, monkeypatch, "mcp:\n  write:\n    enabled_stores: [default]\n")
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    _create_tables()
    yield keydir
    _keys.reset_key_cache()
    _keys.forget_key_files()
    reset_config()


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _stored_tags() -> list[list[str]]:
    async def _go() -> list[list[str]]:
        from particles.corpus.store import list_entries
        from particles.db import get_engine, session_scope

        async with session_scope() as s:
            out = [list(e.tags or []) for e in await list_entries(s)]
        await get_engine().dispose()
        return out

    return asyncio.run(_go())


# --------------------------------------------------------------------------- #
# Attack 2: the whole-tag check refuses a key the mint accepts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", ["проект", "日本語"], ids=["cyrillic", "japanese"])
def test_a_bound_key_may_name_its_own_project_in_a_deposit_tag(store: Path, key: str) -> None:
    """A project key the mint accepts is accepted as that key's own ``project:`` deposit tag."""
    from particles.api import _keys
    from particles.api._keys import Role

    token, record = _keys.mint(Role.AGENT, "agent:ci-bot", project=key)
    assert record.project == key
    resp = _client().post(
        "/corpus/deposit/text",
        json={"text": "a note for the project", "tags": [f"project:{key}"]},
        headers=_bearer(token),
    )
    assert resp.status_code < 300, resp.text


# --------------------------------------------------------------------------- #
# Attack 3: lookalikes that pass every check
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "lookalike",
    [
        "acme०",  # DEVANAGARI DIGIT ZERO for the ASCII 0 after Latin letters
        "acme‐web",  # HYPHEN (U+2010) for the ASCII hyphen-minus
        "acme−web",  # MINUS SIGN for the ASCII hyphen-minus
        "\U00010300\U000103a0",  # OLD ITALIC beside OLD PERSIAN: two scripts, one name word
    ],
    ids=["devanagari-digit", "unicode-hyphen", "minus-sign", "old-italic-old-persian"],
)
def test_a_mixed_script_or_lookalike_project_key_is_never_minted(
    store: Path, lookalike: str
) -> None:
    """A key mixing scripts, or one that renders as an ASCII key, is refused at the mint."""
    from particles.api import _keys
    from particles.api._keys import KeyRuleError, Role

    with pytest.raises(KeyRuleError):
        _keys.mint(Role.AGENT, "agent:ci-bot", project=lookalike)


# --------------------------------------------------------------------------- #
# Attack 1: one project, one key, wherever keys are compared
# --------------------------------------------------------------------------- #


def test_an_entry_tagged_in_decomposed_form_is_in_view_for_the_composed_project() -> None:
    """A belief observed under ``café`` decomposed is in view for the NFC ``café`` observer."""
    from particles.core.observer_scope import scope_of_entries, visible

    decomposed = unicodedata.normalize("NFD", "café")
    scope = scope_of_entries([[f"project:{decomposed}"]], ())
    assert visible(scope, "café")


def test_an_operator_deposit_project_tag_is_stored_in_nfc(store: Path) -> None:
    """An operator key's deposit stores a ``project:`` tag in the NFC form bindings compare."""
    decomposed = unicodedata.normalize("NFD", "café")
    resp = _client().post(
        "/corpus/deposit/text",
        json={"text": "operator note", "tags": [f"project:{decomposed}"]},
        headers=_bearer(LEGACY),
    )
    assert resp.status_code < 300, resp.text
    project_tags = [t for tags in _stored_tags() for t in tags if t.startswith("project:")]
    assert project_tags == ["project:café"], project_tags


def test_an_unbound_keys_session_exposure_never_stores_a_hidden_character_project(
    store: Path,
) -> None:
    """``/session-exposures`` applies the project-key rule to an unbound key's ``project_key``."""
    from particles.api import _keys
    from particles.api._keys import Role

    token, _ = _keys.mint(Role.AGENT, "agent:ci-bot")
    resp = _client().post(
        "/session-exposures",
        json={
            "session_id": "s-1",
            "recorded_at": "2026-10-28T00:00:00Z",
            "source": "startup",
            "action": "full",
            "project_key": "acme​",
            "observer_scope_applied": True,
            "beliefs": [],
        },
        headers=_bearer(token),
    )
    if resp.status_code >= 400:
        return

    async def _keys_stored() -> list[str]:
        from sqlalchemy import select

        from particles.db import get_engine, session_scope
        from particles.store.session_exposure_store import SessionExposureRow

        async with session_scope() as s:
            rows = (await s.execute(select(SessionExposureRow.project_key))).scalars().all()
        await get_engine().dispose()
        return list(rows)

    assert asyncio.run(_keys_stored()) in ([], ["acme"]), json.dumps(asyncio.run(_keys_stored()))


def _harvester() -> str:
    from particles.api import _keys
    from particles.api._keys import Role

    return _keys.mint(Role.AGENT, "claude-code-hook", reserved_ok=True)[0]


# ---------------------------------------------------------------------------
# O33: project keys on the exposure route and in deposit tags
# ---------------------------------------------------------------------------


def _exposure(project_key: str) -> dict[str, object]:
    return {
        "session_id": "s9-session",
        "recorded_at": datetime.now(UTC).isoformat(),
        "source": "startup",
        "action": "full",
        "project_key": project_key,
        "observer_scope_applied": True,
        "beliefs": [],
    }


def _exposure_keys() -> list[str]:
    async def _go() -> list[str]:
        from sqlalchemy import select

        from particles.db import get_engine, session_scope
        from particles.store.session_exposure_store import SessionExposureRow

        async with session_scope() as s:
            out = list((await s.execute(select(SessionExposureRow.project_key))).scalars())
        await get_engine().dispose()
        return out

    return asyncio.run(_go())


def test_a_harvester_exposure_records_its_project_in_nfc(store: Path) -> None:
    """A trusted recorder's exposure stores the project key in NFC, as every binding does."""
    decomposed = unicodedata.normalize("NFD", "café")
    resp = _client().post(
        "/session-exposures", json=_exposure(decomposed), headers=_bearer(_harvester())
    )
    assert resp.status_code == 200, resp.text
    assert _exposure_keys() == ["café"], (
        f"the exposure stored {[ascii(k) for k in _exposure_keys()]}, not the NFC key"
    )


def test_a_harvester_exposure_naming_a_lookalike_project_is_refused(store: Path) -> None:
    """A trusted recorder's exposure never records a project key holding an invisible character."""
    resp = _client().post(
        "/session-exposures", json=_exposure("acme​"), headers=_bearer(_harvester())
    )
    assert resp.status_code in {403, 422}, (
        f"the exposure route stored {[ascii(k) for k in _exposure_keys()]} ({resp.status_code})"
    )


def test_an_unbound_agent_may_tag_a_project_a_key_may_be_bound_to(store: Path) -> None:
    """A project key the mint accepts is also accepted as an unbound agent deposit's tag."""
    from particles.api import _keys
    from particles.api._keys import Role

    name = "проект"
    _bound, record = _keys.mint(Role.AGENT, "agent:ru-bound", project=name)
    assert record.project == name
    unbound, _ = _keys.mint(Role.AGENT, "agent:ru")
    resp = _client().post(
        "/corpus/deposit/text",
        json={"text": "a note for the project", "tags": [f"project:{name}"]},
        headers=_bearer(unbound),
    )
    assert resp.status_code == 200, (
        f"the tag project:{name} was refused although a key may be bound to {name!r}: "
        f"{resp.status_code} {resp.text}"
    )


@pytest.mark.parametrize(
    ("key", "rule"),
    [
        ("acme​", "invisible"),
        ("acme‐web", "punctuation"),
        ("acmе", "scripts"),
        ("ａcme", "compatibility"),
    ],
)
def test_a_refused_project_tag_is_worded_by_the_rule_that_fired(
    store: Path, key: str, rule: str
) -> None:
    """A deposit refusal names the rule a lookalike key broke, not an invisible character."""
    from particles.api import _keys
    from particles.api._keys import Role

    token, _ = _keys.mint(Role.AGENT, "agent:ci-bot")
    resp = _client().post(
        "/corpus/deposit/text",
        json={"text": "x", "tags": [f"project:{key}"]},
        headers=_bearer(token),
    )
    assert resp.status_code == 403, resp.text
    assert rule in resp.text, resp.text


def test_an_observer_in_decomposed_form_reads_the_composed_project() -> None:
    """The observer key compares canonical, as the entry keys do."""
    from particles.core.observer_scope import scope_of_entries, visible

    scope = scope_of_entries([["project:café"]], ())
    assert visible(scope, unicodedata.normalize("NFD", "café"))
    assert not visible(scope, "cafe")
