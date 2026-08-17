"""The §16.1 context-fingerprint digest.

Steps 2–3 of the normative procedure — sort the ACTIVE UUIDs lexicographically,
SHA-256 the concatenation with no delimiter. The spec says the algorithm "MUST
be followed exactly to ensure cross-agent fingerprint compatibility", so every
property below is a contract, not an implementation detail: an L2 vector
family pins the same behaviour for external implementations.
"""

from __future__ import annotations

import hashlib

from particles.core.fingerprint import context_fingerprint

_A = "0f2b6c1e-7a4d-4c9b-8f31-2a5e6d7c8b90"
_B = "7d3e9a12-4b6c-4f80-9a2d-1c5b3e7f0a44"
_C = "b91f4c33-2d7e-4a15-8c60-9e4d2b1a7f83"


def test_empty_baseline_is_sha256_of_empty_string() -> None:
    """A fresh store's canonical fingerprint."""
    assert context_fingerprint([]) == hashlib.sha256(b"").hexdigest()


def test_digest_is_sha256_of_sorted_concatenation() -> None:
    """No delimiter, lexicographic order — the exact §16.1 wording."""
    expected = hashlib.sha256(f"{_A}{_B}{_C}".encode()).hexdigest()
    assert context_fingerprint([_A, _B, _C]) == expected


def test_input_order_does_not_change_the_digest() -> None:
    """The sort is inside the function, so a caller cannot get it wrong."""
    assert context_fingerprint([_C, _A, _B]) == context_fingerprint([_A, _B, _C])


def test_distinct_baselines_differ() -> None:
    assert context_fingerprint([_A, _B]) != context_fingerprint([_A, _B, _C])


def test_digest_is_64_char_lowercase_hex() -> None:
    digest = context_fingerprint([_A, _B, _C])
    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(ch in "0123456789abcdef" for ch in digest)


def test_accepts_any_iterable() -> None:
    """The store passes a generator over the ACTIVE-id query result."""
    assert context_fingerprint(i for i in (_C, _A, _B)) == context_fingerprint([_A, _B, _C])
