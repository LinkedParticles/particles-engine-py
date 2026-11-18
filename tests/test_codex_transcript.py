# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Positive provenance, stable append rendering and fail-closed parsing."""

import copy
import hashlib
import json
from typing import Any

import pytest

from particles.extraction.codex_transcript import UnsupportedTranscript, parse_transcript

SID = "10000000-0000-4000-8000-000000000001"


def metadata() -> dict[str, Any]:
    return {
        "type": "session_meta",
        "payload": {"id": SID, "cwd": "/tmp/repository", "timestamp": "2026-11-01T23:00:00Z"},
    }


def message(
    text: str, mid: str = "one", origin: str = "user", stamp: str = "2026-11-01T23:10:00Z"
) -> dict[str, Any]:
    return {
        "type": "response_item",
        "timestamp": stamp,
        "payload": {
            "type": "message",
            "id": mid,
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
            "internal_chat_message_metadata_passthrough": {
                "content_item_kinds": [
                    "user.text" if origin == "user" else "hooks.additional_context"
                ],
                "content_item_metadata": [{"provenance": {"type": origin}}],
            },
        },
    }


def blob(*records: dict[str, Any]) -> str:
    return "".join(json.dumps(r) + "\n" for r in records)


def test_provenance_and_exact_echo_are_distinct() -> None:
    transcript = parse_transcript(
        blob(
            metadata(), message("digest", origin="harness"), message("I quote the digest", "human")
        )
    )
    text, _ = transcript.render(lambda text: text)
    assert text == "## User\n\nI quote the digest\n\n"
    with pytest.raises(UnsupportedTranscript, match="ambiguous_echo"):
        parse_transcript(
            blob(metadata(), message("digest")),
            emitted_hashes=frozenset({hashlib.sha256(b"digest").hexdigest()}),
        )


def test_day_marks_redaction_and_append_prefix() -> None:
    first = blob(metadata(), message("SECRET"))

    def redact(text: str) -> str:
        return text.replace("SECRET", "[REDACTED]")

    before, identities = parse_transcript(first).render(redact)
    later = message("## User · 2026-01-01\nnew", "two", stamp="2026-11-02T01:00:00Z")
    rollback = message("clock went back", "three", stamp="2026-10-01T01:00:00Z")
    after, _ = parse_transcript(first + blob(later, rollback)).render(redact)
    assert after.startswith(before)
    assert "## User · 2026-11-02" in after
    assert "\\## User · 2026-01-01" in after
    assert "## User · 2026-10-01" not in after
    assert identities[0]["body_hash"] == hashlib.sha256(b"[REDACTED]").hexdigest()
    assert "SECRET" not in after


@pytest.mark.parametrize(
    "record",
    [
        {"type": "novel", "payload": {}},
        {"type": "event_msg", "payload": {"type": "novel"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": []}},
        {"type": "compacted", "payload": {"replacement_history": []}},
    ],
)
def test_unknown_complete_records_quarantine(record: dict[str, Any]) -> None:
    with pytest.raises(UnsupportedTranscript):
        parse_transcript(blob(metadata(), message("good"), record))


def test_incomplete_tail_and_duplicate_mirrors() -> None:
    first = message("good")
    duplicate = copy.deepcopy(first)
    duplicate["timestamp"] = "2026-11-02T00:00:00Z"
    text, _ = parse_transcript(blob(metadata(), first, duplicate) + '{"incomplete"').render(str)
    assert text.count("good") == 1
    with pytest.raises(UnsupportedTranscript, match="conflicting_message_id"):
        parse_transcript(blob(metadata(), first, message("changed")))
    with pytest.raises(UnsupportedTranscript, match="malformed_complete_record"):
        parse_transcript(blob(metadata()) + '{"incomplete"\n')


def test_paginated_ancestry_and_compaction_summary_not_testimony() -> None:
    meta = metadata()
    meta["payload"]["history_base"] = {
        "thread_id": "20000000-0000-4000-8000-000000000002",
        "end_ordinal_exclusive": 3,
        "end_byte_offset": 100,
    }
    transcript = parse_transcript(
        blob(
            meta,
            message("child only"),
            {"type": "compacted", "payload": {"message": "GENERATED", "replacement_history": []}},
        )
    )
    assert transcript.ancestry and transcript.ancestry.end_byte_offset == 100
    assert "GENERATED" not in transcript.render(str)[0]
    meta["payload"].pop("history_base")
    meta["payload"]["forked_from_id"] = SID
    with pytest.raises(UnsupportedTranscript, match="unsupported_legacy_fork"):
        parse_transcript(blob(meta))
