# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the harness findings of security review 2026-10-25-2 (S5).

R18: a project-scope install merges only the settings file it checked.
R20: redaction masks an empty-user URL password, a JSON-escaped assignment and
a value with an early separator.
R21: the PEM mask stays linear on unterminated ``BEGIN`` lines.
R22: the SessionEnd utility miner receives the masked transcript.
Critic item: the projection's contained write stays contained on a platform
with no kernel-path recheck.

Every test runs under a temporary ``HOME``, opens no real store and makes no
model call.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from particles.corpus.redaction import redact_secrets

# --- R20 ---------------------------------------------------------------------


def test_q28_fixed_shapes_are_still_masked() -> None:
    """Control: the four shapes the fourth audit named stay masked."""
    cases = {
        "PGPASSWORD=correcthorse9": "correcthorse9",
        'password: "a long, spaced secret"': "a long, spaced secret",
        "https://alice:hunter2pass@example.com/x": "hunter2pass",
        "Authorization: Basic dXNlcjpwYXNzd29yZA==": "dXNlcjpwYXNzd29yZA==",
    }
    for text, secret in cases.items():
        assert secret not in redact_secrets(text), text


def test_url_password_with_an_empty_username_is_masked() -> None:
    """A URL whose userinfo has an empty user (the usual Redis form) has its password masked."""
    secret = "Sup3rS3cretPw"
    out = redact_secrets(f"REDIS_URL=redis://:{secret}@cache:6379/0")
    assert secret not in out
    assert out == "REDIS_URL=redis://:[REDACTED]@cache:6379/0"


def test_json_escaped_secret_assignment_is_masked() -> None:
    """A secret-named assignment inside JSON-escaped quotes has its value masked."""
    secret = "hunter2-escaped-value"
    text = '{\\"password\\": \\"' + secret + '\\"}'
    out = redact_secrets(text)
    assert secret not in out
    assert out == '{\\"password\\": \\"[REDACTED]\\"}'


def test_unquoted_value_with_a_separator_early_is_masked() -> None:
    """A secret-named value holding a ';' or ',' in its first six characters is masked whole."""
    assert redact_secrets("password=ab;cdefgh1234") == "password=[REDACTED]"
    assert redact_secrets("API_KEY=x,yzzy98765 next") == "API_KEY=[REDACTED] next"


@pytest.mark.parametrize(
    "text",
    [
        "see https://example.invalid:8443/path@v2",
        "ssh://git@github.com/org/repo.git",
        'password: "short"',
        "password=abc;",
        '{\\"password\\": \\"abc\\"}',
    ],
)
def test_r20_lookalikes_are_kept(text: str) -> None:
    assert redact_secrets(text) == text


# --- R21 ---------------------------------------------------------------------


def test_pem_mask_stays_linear_on_unterminated_headers() -> None:
    """Masking ~300 KB of PEM BEGIN lines that never END finishes well under the hook deadline."""
    text = "-----BEGIN A-----\n" * 16_000  # 288 KB
    start = time.monotonic()
    assert redact_secrets(text) == text
    assert time.monotonic() - start < 3.0


def test_pem_mask_masks_each_terminated_block() -> None:
    """A terminated block is masked, an unterminated BEGIN after it is kept."""
    block = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----"
    text = f"a {block} b {block} c -----BEGIN X----- tail"
    assert redact_secrets(text) == (
        "a [REDACTED PEM BLOCK] b [REDACTED PEM BLOCK] c -----BEGIN X----- tail"
    )


# --- R22 ---------------------------------------------------------------------

_TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"


def _transcript(tmp_path: Path) -> Path:
    record = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": "Bash",
                    "input": {"command": f"GITHUB_TOKEN={_TOKEN} gh api repos/x/y"},
                }
            ]
        },
    }
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def test_the_deposited_transcript_masks_the_token(tmp_path: Path) -> None:
    """Control: the harvest deposit path masks the token."""
    from particles.api.cli.hook import _distilled_session

    text = _distilled_session(_transcript(tmp_path).read_text(encoding="utf-8"), "s")
    assert "[tool: Bash" in text
    assert _TOKEN not in text


def test_the_utility_miner_receives_a_masked_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The transcript the SessionEnd hook hands the utility miner (whose judge sends action
    lines to a model) carries no credential the deposit would have masked."""
    import particles.operations.utility_mining as mining
    from particles.api.cli.hook import _mine_utility

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    seen: dict[str, Any] = {}

    async def recorder(store: str, transcript: str, session_id: str, **_: Any) -> Any:
        seen["transcript"] = transcript

        class _R:
            literal = behavioural = candidates = literal_nominated = behavioural_calls = 0

        return _R()

    monkeypatch.setattr(mining, "mine_session_from_transcript", recorder)
    asyncio.run(_mine_utility("default", _transcript(tmp_path), "s", "proj"))
    assert "[tool: Bash" in seen["transcript"]
    assert _TOKEN not in seen["transcript"]


# --- R18 ---------------------------------------------------------------------

_MARKER = "user-settings-only-value-7f3a"


def _project_install_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A temporary HOME holding a user settings file, and a project to install into."""
    from particles.config import reset_config

    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    user_settings = home / ".claude" / "settings.json"
    user_settings.write_text(json.dumps({"env": {"SOME_TOKEN": _MARKER}}) + "\n", encoding="utf-8")
    project = tmp_path / "project"
    (project / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PARTICLES_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'store.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.chdir(project)
    reset_config()
    return user_settings, project / ".claude" / "settings.local.json"


_PROJECT_INSTALL = ["init", "claude-code", "--project", "--store", "default", "--no-audit"]


def test_project_install_never_merges_a_file_other_than_the_one_it_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `--project` install whose settings file is swapped for a link after the check writes
    nothing from the link's target into the project settings."""
    import particles.api.cli.init as init_mod
    from particles.api.cli import app

    user_settings, local = _project_install_env(tmp_path, monkeypatch)
    local.write_text("{}\n", encoding="utf-8")

    original = init_mod._refuse_symlinked_project_settings

    def check_then_swap(path: Path) -> None:
        original(path)
        os.unlink(path)
        os.symlink(user_settings, path)

    original_load = init_mod._load_settings

    def read_then_swap_back(path: Path) -> dict[str, object]:
        loaded = original_load(path)
        if path.is_symlink():
            os.unlink(path)
            path.write_text("{}\n", encoding="utf-8")
        return loaded

    monkeypatch.setattr(init_mod, "_refuse_symlinked_project_settings", check_then_swap)
    monkeypatch.setattr(init_mod, "_load_settings", read_then_swap_back)
    CliRunner().invoke(app, [*_PROJECT_INSTALL, "--no-skills"])
    assert not local.is_symlink()
    written = local.read_text(encoding="utf-8") if local.exists() else ""
    assert _MARKER not in written
    assert "SessionStart" in written
    assert not init_mod._CHECKED_PROJECT_SETTINGS


def test_project_settings_load_without_a_prior_check_refuses_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A project-scope load that no check preceded opens the name without following a link."""
    import typer

    import particles.api.cli.init as init_mod

    user_settings, local = _project_install_env(tmp_path, monkeypatch)
    local.symlink_to(user_settings)
    with pytest.raises(typer.Exit):
        init_mod._load_settings(local)


def test_project_install_refuses_a_settings_directory_of_another_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A project `.claude` directory another account owns is refused before anything is read."""
    from particles.api.cli import app

    _, local = _project_install_env(tmp_path, monkeypatch)
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
    result = CliRunner().invoke(app, [*_PROJECT_INSTALL, "--no-skills"])
    assert result.exit_code == 1
    assert "belongs to another account" in result.output
    assert not local.exists()


# --- Critic item: the projection's walk without a kernel path ----------------


@pytest.mark.parametrize("kernel_path", [True, False])
def test_projection_root_swapped_after_the_resolve_is_not_written_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kernel_path: bool
) -> None:
    """A projection root replaced by a link after it was resolved is refused, also on a
    platform with neither ``F_GETPATH`` nor ``/proc`` (no kernel-path recheck)."""
    from particles.corpus.local_read import LocalReadRefused
    from particles.operations.projection import paths

    if not kernel_path:
        monkeypatch.setattr(paths, "_descriptor_path", lambda fd: None)
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    outside = tmp_path / "outside"
    (outside / "docs").mkdir(parents=True)
    real_realpath = os.path.realpath
    swapped: list[bool] = []

    def swap_after_the_last_resolve(p: Any, *args: Any, **kwargs: Any) -> str:
        resolved = real_realpath(p, *args, **kwargs)
        if not swapped and str(p) == str(root / "docs"):
            swapped.append(True)
            os.rename(root, tmp_path / "repo.moved")
            os.symlink(outside, root)
        return resolved

    monkeypatch.setattr(os.path, "realpath", swap_after_the_last_resolve)
    with pytest.raises(LocalReadRefused):
        paths.write_text_no_follow(root / "docs" / "out.md", "x", root=root)
    assert swapped
    assert not (outside / "docs" / "out.md").exists()


def test_projection_walk_refuses_a_swapped_parent_without_a_kernel_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Q29 proof holds with the kernel-path recheck unavailable."""
    from particles.corpus.local_read import LocalReadRefused
    from particles.operations.projection import paths

    monkeypatch.setattr(paths, "_descriptor_path", lambda fd: None)
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    real_mkdir = os.mkdir

    def swap_then_mkdir(name: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        if name == "docs" and dir_fd is not None and (root / "docs").is_dir():
            os.rename(root / "docs", root / "docs.moved")
            os.symlink(outside, root / "docs")
        real_mkdir(name, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", swap_then_mkdir)
    with pytest.raises(LocalReadRefused):
        paths.write_text_no_follow(root / "docs" / "out.md", "x", root=root)
    assert not (outside / "out.md").exists()
