# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""The Obsidian plugin refuses its bearer over cleartext to a remote host (P11, K4).

The behaviour is tested under Node by the plugin's own ``npm test``
(``clients/obsidian-plugin/test/transport.test.mts``); no CI job runs it, so
this static read keeps the Python gate from losing the call that applies it.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests._upstream import upstream_only

_PLUGIN = Path(__file__).resolve().parents[1] / "clients" / "obsidian-plugin" / "src"


@upstream_only
def test_every_request_checks_the_url_before_attaching_the_bearer() -> None:
    api = (_PLUGIN / "api.ts").read_text(encoding="utf-8")
    request = api[api.index("private async request<T>(") :]
    before_send = request[: request.index("this.fetchFn(")]
    assert "this.requireEncryptedOrLoopback();" in before_send
    guard = api[api.index("private requireEncryptedOrLoopback(): void {") :]
    guard = guard[: guard.index("\n  }\n")]
    assert "isCleartextRemote(" in guard and "allowPlaintextToken" in guard


@upstream_only
def test_the_cleartext_check_mirrors_the_python_rule() -> None:
    transport = (_PLUGIN / "transport.ts").read_text(encoding="utf-8")
    assert 'parsed.protocol === "https:"' in transport
    assert re.search(r"127\\\.\\d", transport), "127.0.0.0/8 is loopback"
    assert '"[::1]"' in transport and '"localhost"' in transport
