# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Child-side shim that serves one adebench adapter over JSON lines.

Run by the door arm as ``python -I door_shim.py --adapter module:Class --root
<adebench clone>`` under the clone's own interpreter, with the clone as the
working directory. It imports nothing from Particles (``-I`` keeps this file's
directory off ``sys.path``), loads the adapter exactly as adebench's own
``--adapter`` flag does, and answers one request per line:

    {"op": "methods"}                       the contract methods the adapter has
    {"op": "health"}                        adapter.health()
    {"op": "doors"}                         adapter.doors()
    {"op": "door_text", "query", "door"}    adapter.door_text(query, door)
    {"op": "import_memory", "text", "written_at"}
    {"op": "import_session", "turns", "written_at"}
    {"op": "settle"}

Each reply is ``{"ok": true, "result": ...}`` or ``{"ok": false, "error": ...}``.
The real stdout carries the protocol and nothing else: anything an adapter or
its memory prints goes to stderr.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from typing import Any

#: A raw result larger than this many characters of JSON is kept as its size
#: and its first characters only (the scorer never reads it).
RAW_LIMIT = 200_000

_METHODS = (
    "health",
    "doors",
    "door_text",
    "import_memory",
    "import_session",
    "settle",
)


def _jsonable(value: Any) -> Any:
    text = json.dumps(value, default=str)
    if len(text) <= RAW_LIMIT:
        return json.loads(text)
    return {"truncated": True, "chars": len(text), "head": text[:RAW_LIMIT]}


def _dispatch(adapter: Any, req: dict[str, Any]) -> Any:
    op = req.get("op")
    if op == "methods":
        return [m for m in _METHODS if callable(getattr(adapter, m, None))]
    if op == "health":
        return bool(adapter.health())
    if op == "doors":
        return list(adapter.doors())
    if op == "door_text":
        text, raw = adapter.door_text(req["query"], req["door"])
        return {"text": text, "raw": _jsonable(raw)}
    if op == "import_memory":
        return _jsonable(adapter.import_memory(req["text"], req["written_at"]))
    if op == "import_session":
        return _jsonable(adapter.import_session(req["turns"], req["written_at"]))
    if op == "settle":
        settle = getattr(adapter, "settle", None)
        return None if settle is None else _jsonable(settle())
    raise ValueError(f"unknown op {op!r}")


def main() -> int:
    """Serve requests until stdin closes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr
    sys.path.insert(0, args.root)
    module, _, cls_name = args.adapter.partition(":")
    try:
        adapter = getattr(importlib.import_module(module), cls_name)()
    except Exception as exc:  # noqa: BLE001 — reported to the parent, which refuses
        proto.write(json.dumps({"ok": False, "error": f"cannot load {args.adapter}: {exc}"}) + "\n")
        return 2
    proto.write(json.dumps({"ok": True, "result": "ready"}) + "\n")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            reply = {"ok": True, "result": _dispatch(adapter, json.loads(line))}
        except Exception as exc:  # noqa: BLE001 — every failure goes back as data
            traceback.print_exc(file=sys.stderr)
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        proto.write(json.dumps(reply) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
