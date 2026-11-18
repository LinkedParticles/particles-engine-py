# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""The curation core's fixtures cover every card kind the engine emits.

The core's golden capture and contract are built from
``clients/curation-core/fixtures/cards.json``, one example card per kind. A
kind the engine gains without a fixture would reach the web UI and the
Claude Code pane untested, so this fails until the fixture is added (and the
golden and contract regenerated, per that package's README).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from particles.operations.curation.cards import CardKind

FIXTURES = Path(__file__).resolve().parents[1] / "clients" / "curation-core" / "fixtures"


def _fixture_kinds() -> set[str]:
    path = FIXTURES / "cards.json"
    if not path.exists():
        pytest.skip("this tree carries no curation core")
    cards = json.loads(path.read_text(encoding="utf-8"))["cards"]
    return {str(card["kind"]) for card in cards}


def test_every_card_kind_has_a_fixture() -> None:
    missing = {k.value for k in CardKind} - _fixture_kinds()
    assert not missing, f"card kinds with no curation-core fixture: {sorted(missing)}"


def test_every_fixture_names_a_card_kind() -> None:
    unknown = _fixture_kinds() - {k.value for k in CardKind}
    assert not unknown, f"curation-core fixtures for kinds the engine lacks: {sorted(unknown)}"
