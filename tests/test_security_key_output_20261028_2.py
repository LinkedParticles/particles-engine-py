# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""``keys create`` never shows a key to a harness that owns the terminal (2026-10-28-2 review).

AA45: a harness that runs its shell on a pseudo-terminal looks like a person's
terminal, so the key goes to a new private file unless ``--print-token`` is
given. The verb runs in a child process with a temporary ``HOME`` and key
file, ``CLAUDECODE`` cleared and no ``--print-token``; the test is the
harness, reading the child's output from the pty's master side.
"""

from __future__ import annotations

import os
import pty
import re
import select
import subprocess
import sys
import time
from pathlib import Path

_TOKEN = re.compile(rb"ptk_agent_[0-9a-f]{12}_[A-Za-z0-9_-]{43}")


def test_a_key_minted_under_a_capturing_pty_harness_is_not_printed_to_it(tmp_path: Path) -> None:
    """A harness capturing a pty (no CLAUDECODE, no --print-token) never receives a key."""
    home = tmp_path / "home"
    home.mkdir()
    (tmp_path / "keys").mkdir(mode=0o700)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("network:\n  egress: none\n", encoding="utf-8")
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"CLAUDECODE", "ANTHROPIC_API_KEY", "PARTICLES_API_KEY"}
        and not k.startswith("PARTICLES_LLM_API_KEY_")
    }
    env.update(
        HOME=str(home),
        PARTICLES_CONFIG=str(cfg),
        PARTICLES_API_KEYS_FILE=str(tmp_path / "keys" / "engine-keys.json"),
        DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 's9.db'}",
    )
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from particles.api.cli import app; app()",
            "engine",
            "keys",
            "create",
            "--role",
            "agent",
            "--principal",
            "agent:other-harness",
        ],
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=slave,
        env=env,
    )
    os.close(slave)
    captured = b""
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master], [], [], 0.2)
        if ready:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            captured += chunk
        elif proc.poll() is not None:
            break
    proc.wait(timeout=30)
    os.close(master)
    assert b"Created key" in captured, captured.decode(errors="replace")
    assert proc.returncode == 0, captured.decode(errors="replace")
    assert not bool(_TOKEN.search(captured)), (
        "keys create printed a usable key to the harness that owns the pseudo-terminal"
    )
    written = list((tmp_path / "keys").glob("*.key"))
    assert len(written) == 1 and written[0].stat().st_mode & 0o777 == 0o600
    assert _TOKEN.fullmatch(written[0].read_bytes().strip())
