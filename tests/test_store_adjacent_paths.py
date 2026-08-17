"""Tests for store-adjacent path resolution."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from particles import config as config_mod
from particles import db
from particles.config import resolve_store_adjacent_path, sqlite_file_path


def _stub_config(database_url: str) -> SimpleNamespace:
    return SimpleNamespace(storage=SimpleNamespace(database_url=database_url))


@pytest.fixture(autouse=True)
def _no_config_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to the compiled-defaults case: no ``config.yaml`` discovered."""
    monkeypatch.setattr(config_mod, "_find_config_file", lambda: None)


class TestSqliteFilePath:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("sqlite+aiosqlite:///./particles.db", "./particles.db"),
            ("sqlite+aiosqlite:////srv/data/particles.db", "/srv/data/particles.db"),
            ("sqlite+aiosqlite:///:memory:", None),
            ("postgresql+asyncpg://u:p@h/db", None),
            ("sqlite://", None),
        ],
    )
    def test_parsing(self, url: str, expected: str | None) -> None:
        assert sqlite_file_path(url) == expected


class TestResolveStoreAdjacentPath:
    def test_absolute_value_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            config_mod, "get_config", lambda: _stub_config("sqlite+aiosqlite:////srv/p.db")
        )
        assert resolve_store_adjacent_path("/var/blobs") == Path("/var/blobs")

    def test_relative_anchors_to_absolute_store_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The 2026-07-18 incident: an absolute DATABASE_URL plus the relative
        # `./corpus_blobs` default sent blobs to whichever directory the process
        # happened to run in, decoupling a store's rows from its content. They
        # must land beside the store instead.
        monkeypatch.setattr(
            config_mod,
            "get_config",
            lambda: _stub_config("sqlite+aiosqlite:////srv/data/particles.db"),
        )
        assert resolve_store_adjacent_path("./corpus_blobs") == Path("/srv/data/corpus_blobs")

    def test_relative_dsn_keeps_cwd_behaviour(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Both cwd-relative ⇒ store and blobs already travel together; unchanged.
        monkeypatch.setattr(
            config_mod, "get_config", lambda: _stub_config("sqlite+aiosqlite:///./particles.db")
        )
        assert resolve_store_adjacent_path("./corpus_blobs") == Path("./corpus_blobs")

    def test_non_file_dsn_anchors_to_config_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("storage: {}\n")
        monkeypatch.setattr(config_mod, "_find_config_file", lambda: cfg)
        monkeypatch.setattr(
            config_mod, "get_config", lambda: _stub_config("postgresql+asyncpg://u:p@h/db")
        )
        assert resolve_store_adjacent_path("blobs") == tmp_path.resolve() / "blobs"

    def test_no_anchor_falls_back_to_cwd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            config_mod, "get_config", lambda: _stub_config("postgresql+asyncpg://u:p@h/db")
        )
        assert resolve_store_adjacent_path("blobs") == Path("blobs")

    def test_user_home_is_expanded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            config_mod, "get_config", lambda: _stub_config("postgresql+asyncpg://u:p@h/db")
        )
        assert resolve_store_adjacent_path("~/blobs") == Path.home() / "blobs"


class TestSharedDsnParser:
    """the write lock and the resolver share one DSN→path rule."""

    def test_db_reexports_the_client_layer_parser(self) -> None:
        # Identity, not equivalence: two copies of the rule are what let the
        # write lock and blob_dir disagree about where the store lives.
        assert db._sqlite_file_path is sqlite_file_path

    def test_lockfile_and_blob_dir_land_beside_the_same_store(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/particles.db")
        config_mod.reset_config()

        lockfile = Path(db._write_lock_path(db.DEFAULT_STORE))
        blob_dir = resolve_store_adjacent_path("./corpus_blobs")

        assert lockfile == tmp_path / "particles.db.writelock"
        assert blob_dir == tmp_path / "corpus_blobs"
