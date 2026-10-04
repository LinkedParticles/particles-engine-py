# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""The append-only delta read's text arithmetic (``extraction/append_delta.py``).

Store-free: the context window, the delta chunk plan, the decoded-prefix
check, the raw-byte mapping of a stop point, and the §4 derivation of a
pre-record base's offset from chunk hashes. The pipeline behaviour built on
them is in ``test_append_base.py``.
"""

from __future__ import annotations

import hashlib

from particles.config import get_config
from particles.extraction.append_delta import (
    FALLBACK_DECODED_PREFIX,
    context_before,
    decoded_prefix_length,
    derive_base_offset,
    plan_delta_chunks,
    raw_offset_for,
)
from particles.extraction.general import (
    APPEND_CONTEXT_RULE,
    _build_llm_request,
    _normalise_for_hashing,
    _split_into_paragraph_chunks,
    paragraph_spans,
)
from particles.extraction.incremental import ChunkUnit, _hash_chunk


def _para(tag: str, n: int) -> str:
    return f"{tag} " + " ".join(f"{tag.lower()}{i}" for i in range(n))


class TestContextBefore:
    def test_cut_back_to_a_paragraph_boundary(self) -> None:
        text = "first paragraph is long enough\n\nsecond paragraph\n\nthird paragraph"
        # A window that starts inside the first paragraph drops it whole.
        ctx = context_before(text, len(text), 40)
        assert ctx == "second paragraph\n\nthird paragraph"

    def test_whole_text_when_it_fits(self) -> None:
        text = "one\n\ntwo"
        assert context_before(text, len(text), 4000) == "one\n\ntwo"

    def test_zero_chars_means_no_context(self) -> None:
        assert context_before("anything", 8, 0) is None

    def test_nothing_before_the_start(self) -> None:
        assert context_before("anything", 0, 4000) is None


class TestPlanDeltaChunks:
    def test_the_delta_only_with_the_prefix_end_as_context(self) -> None:
        old = "old one\n\nold two\n\n"
        text = old + "new three\n\nnew four"
        [chunk] = plan_delta_chunks(text, len(old), chunk_chars=1000, context_chars=4000)
        assert chunk.text == "new three\n\nnew four"
        assert "old" not in chunk.text
        assert chunk.context == "old one\n\nold two"
        assert text[chunk.start : chunk.end] == chunk.text

    def test_a_long_delta_splits_at_paragraphs_and_chains_context(self) -> None:
        old = _para("Old", 50) + "\n\n"
        paras = [_para(tag, 150) for tag in ("Aa", "Bb", "Cc")]
        text = old + "\n\n".join(paras)
        chunks = plan_delta_chunks(text, len(old), chunk_chars=1000, context_chars=300)

        assert [c.text for c in chunks] == paras
        # Each chunk ends on a paragraph boundary of the text.
        assert all(text[c.end : c.end + 2] in ("\n\n", "") for c in chunks)
        # The first chunk's context is the end of the old text; each later
        # chunk's is the end of the chunk before it, never older text.
        assert chunks[0].context is not None and chunks[0].context.startswith("Old")
        for prev, chunk in zip(chunks, chunks[1:], strict=False):
            assert chunk.context is not None
            assert prev.text.endswith(chunk.context)
            assert "Old" not in chunk.context

    def test_an_empty_delta_plans_nothing(self) -> None:
        text = "all old\n\n"
        assert plan_delta_chunks(text, len(text), chunk_chars=1000, context_chars=4000) == []


class TestParagraphSpans:
    def test_spans_reproduce_the_chunker_byte_for_byte(self) -> None:
        text = "  lead\n\n" + "\n\n".join(_para(t, 40) for t in "ABCDEFG") + "\n\n  \n\ntail  \n"
        for size in (80, 300, 1000):
            spans = paragraph_spans(text, size)
            assert [text[s:e] for s, e in spans] == _split_into_paragraph_chunks(text, size)


class TestPrefixAndOffsets:
    def test_decoded_prefix_that_prefixes(self) -> None:
        prefix = b"one\n\ntwo\n\n"
        length, reason = decoded_prefix_length(
            prefix, "one\n\ntwo\n\nthree", is_markdown=False, mark_tools=False
        )
        assert (length, reason) == (len("one\n\ntwo\n\n"), None)

    def test_decoded_prefix_that_no_longer_prefixes_falls_back(self) -> None:
        # The same raw bytes, decoded by a decoder that now reads the text
        # differently: here the text the extractor holds was tool-marked and
        # the prefix, decoded without the marker, no longer lines up.
        prefix = b"hello\n\ntool: ls\n\n"
        text = "hello\n\ntool output (unverified, not the speaker's words): ls\n\nmore"
        length, reason = decoded_prefix_length(prefix, text, is_markdown=False, mark_tools=False)
        assert length is None
        assert reason == FALLBACK_DECODED_PREFIX

    def test_the_offset_is_in_raw_bytes(self) -> None:
        # Multi-byte characters make raw bytes and decoded characters differ.
        content = "café one\n\nnaïve two\n\nthird\n".encode()
        decoded = content.decode()
        stop = decoded.index("naïve")
        offset = raw_offset_for(content, stop, is_markdown=False, mark_tools=False)
        assert offset == content.index("naïve".encode())
        assert offset != stop
        assert content[:offset].decode() == decoded[:stop]

    def test_reaching_the_end_is_the_content_length(self) -> None:
        content = b"one\n\ntwo"
        assert raw_offset_for(content, 10_000, is_markdown=False, mark_tools=False) == len(content)

    def test_a_stop_inside_a_paragraph_rounds_down(self) -> None:
        content = b"one\n\ntwo two two\n\nthree"
        offset = raw_offset_for(
            content, content.index(b"two") + 3, is_markdown=False, mark_tools=False
        )
        assert offset == content.index(b"two")


class TestDeriveBaseOffset:
    def _content(self) -> tuple[bytes, list[str]]:
        size = get_config().extraction.html_chunk_size
        paras = [_para(tag, size // 12) for tag in ("Aa", "Bb", "Cc", "Dd")]
        content = "\n\n".join(paras).encode()
        text = _normalise_for_hashing(content.decode())
        chunks = _split_into_paragraph_chunks(text, size)
        assert len(chunks) >= 3, "setup: the base must be chunked"
        return content, chunks

    def test_a_chunk_that_yielded_nothing_does_not_pull_the_offset_back(self) -> None:
        # The first chunk was read and produced no claim; the second did. The
        # read reached the second chunk, so it is not read again.
        content, chunks = self._content()
        carried = {_hash_chunk(chunks[1])}
        offset = derive_base_offset(content, carried, is_markdown=False, mark_tools=False)
        assert offset is not None
        assert content[offset:].decode().startswith(chunks[2])

    def test_derived_from_the_last_carried_chunk(self) -> None:
        content, chunks = self._content()
        carried = {_hash_chunk(c) for c in chunks[:2]}
        offset = derive_base_offset(content, carried, is_markdown=False, mark_tools=False)
        assert offset is not None
        assert content[offset:].decode().startswith(chunks[2])

    def test_fully_extracted_when_a_hash_is_not_a_current_chunk(self) -> None:
        content, chunks = self._content()
        carried = {_hash_chunk(chunks[0]), hashlib.sha256(b"an older chunking").hexdigest()}
        assert derive_base_offset(content, carried, is_markdown=False, mark_tools=False) is None

    def test_fully_extracted_when_read_in_one_call(self) -> None:
        content = b"a short transcript\n\nread in one call\n"
        assert (
            derive_base_offset(content, {"anything"}, is_markdown=False, mark_tools=False) is None
        )

    def test_fully_extracted_when_every_chunk_is_carried(self) -> None:
        content, chunks = self._content()
        carried = {_hash_chunk(c) for c in chunks}
        assert derive_base_offset(content, carried, is_markdown=False, mark_tools=False) is None


class TestDeltaPrompt:
    def test_context_rides_its_own_fence_and_the_rule_is_in_the_system_turn(self) -> None:
        planned = _build_llm_request("the new text", context="the earlier text")
        prompt = planned.request.prompt
        assert prompt.index("<context nonce=") < prompt.index("<source nonce=")
        assert "the earlier text" in prompt and "the new text" in prompt
        system = (planned.request.cache_prefix or "") + planned.request.system
        assert APPEND_CONTEXT_RULE.strip() in system
        # The instruction is not inlined in the data.
        assert "Extract NO claim" not in prompt

    def test_no_context_leaves_the_request_as_before(self) -> None:
        planned = _build_llm_request("the new text")
        system = (planned.request.cache_prefix or "") + planned.request.system
        assert "<context" not in planned.request.prompt
        assert APPEND_CONTEXT_RULE.strip() not in system

    def test_the_chunk_hash_covers_the_context(self) -> None:
        plain = ChunkUnit(chunk_id="c", chunk_text="text")
        with_ctx = ChunkUnit(chunk_id="c", chunk_text="text", context="before")
        other_ctx = ChunkUnit(chunk_id="c", chunk_text="text", context="elsewhere")
        assert plain.prompt_hash == _hash_chunk("text")
        assert len({plain.prompt_hash, with_ctx.prompt_hash, other_ctx.prompt_hash}) == 3
