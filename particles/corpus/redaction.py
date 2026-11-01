# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Mask common credential shapes in harvested text.

Best-effort defence in depth, not a guarantee. The Claude Code harness applies
it to distilled transcripts, memory files and rule files before deposit, and
the local refresh tier applies it when it re-reads a rule or memory file the
harness enrolled, so a secret written into one after enrolment is not archived
raw (security review P48). It lives here, below both, because the refresh tier
may not import the harness.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Callable

from particles.secrets import ENGINE_KEY_SHAPE, mask_engine_keys

_Replacement = str | Callable[[re.Match[str]], str]


def _basic_credentials(match: re.Match[str]) -> str:
    """Mask an HTTP ``Basic`` value only when it decodes to ``user:password`` (Q28).

    "Basic" is an ordinary English word, so the decode is what tells a
    credential from prose ("Basic configuration" is not valid base64).
    """
    token = match["token"]
    try:
        decoded = base64.b64decode(token, validate=True)
    except (binascii.Error, ValueError):
        return match[0]
    return f"{match['head']}[REDACTED]" if b":" in decoded else match[0]


_PEM_BEGIN = re.compile(r"-----BEGIN [A-Z0-9 ]+-----")
_PEM_END = re.compile(r"-----END [A-Z0-9 ]+-----")


def _mask_pem_blocks(text: str) -> str:
    """Replace each ``BEGIN … END`` PEM block, scanning the text once (R21).

    A lazy ``BEGIN.*?END`` pattern rescans to the end of the text from every
    unterminated ``BEGIN``, which is quadratic. Here a ``BEGIN`` with no ``END``
    after it ends the scan, since no later ``BEGIN`` can have one either.
    """
    parts: list[str] = []
    pos = 0
    while (begin := _PEM_BEGIN.search(text, pos)) is not None:
        end = _PEM_END.search(text, begin.end())
        if end is None:
            break
        parts.append(text[pos : begin.start()])
        parts.append("[REDACTED PEM BLOCK]")
        pos = end.end()
    if not parts:
        return text
    parts.append(text[pos:])
    return "".join(parts)


_REDACTIONS: list[tuple[re.Pattern[str], _Replacement]] = [
    # Bearer headers before generic keys so the header keeps its label.
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{6,}"), "Bearer [REDACTED]"),
    # HTTP Basic credentials: base64 of ``user:password`` (Q28).
    (
        re.compile(r"(?P<head>\b[Bb]asic\s+)(?P<token>[A-Za-z0-9+/]{6,}={0,2})(?![A-Za-z0-9+/=])"),
        _basic_credentials,
    ),
    # Per-client engine keys, ``ptk_<role>_<id>_<secret>`` (security
    # review 2026-10-26-1, W1).
    (ENGINE_KEY_SHAPE, "[REDACTED ENGINE KEY]"),
    # Anthropic / OpenAI-style secret keys.
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "[REDACTED API KEY]"),
    # AWS access key IDs.
    (re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}\b"), "[REDACTED AWS KEY]"),
    # GitHub tokens: classic (ghp_, gho_, ghu_, ghs_, ghr_) and fine-grained.
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"), "[REDACTED GITHUB TOKEN]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"), "[REDACTED GITHUB TOKEN]"),
    # Slack tokens (xoxb-, xoxp-, xoxa-, xoxr-, xoxs-, xoxe-, xoxc-) and app tokens.
    (re.compile(r"\bxox[abceprs]-[A-Za-z0-9-]{10,}"), "[REDACTED SLACK TOKEN]"),
    (re.compile(r"\bxapp-\d+-[A-Za-z0-9-]{10,}"), "[REDACTED SLACK TOKEN]"),
    # Google API keys and OAuth access tokens.
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}"), "[REDACTED GOOGLE KEY]"),
    (re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}"), "[REDACTED GOOGLE TOKEN]"),
    # Stripe secret and restricted keys, GitLab, Hugging Face and npm tokens
    # (security review 2026-10-24-2, P25).
    (re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}"), "[REDACTED API KEY]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}"), "[REDACTED GITLAB TOKEN]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "[REDACTED HUGGING FACE TOKEN]"),
    (re.compile(r"\bnpm_[A-Za-z0-9]{36,}"), "[REDACTED NPM TOKEN]"),
    # JSON Web Tokens: three base64url segments, the first two JSON objects.
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
        "[REDACTED JWT]",
    ),
    # URL userinfo (scheme://user:password@host): the password only (Q28). The
    # user may be empty, as in ``redis://:password@host`` (R20), and the
    # password may hold a raw ``/``, ``?`` or ``#``: it runs to the last ``@``
    # before a host, so no tail of it is left unmasked (2026-10-26-1, R20). A
    # port followed by a path (``host:8443/path@v2``) is not a password.
    (
        re.compile(
            r"(?i)\b(?P<head>[a-z][a-z0-9+.-]{0,31}://[^\s:/?#@\"']*:)"
            r"(?!\d{1,5}[/?#])[^\s@\"']+@"
            r"(?=[A-Za-z0-9\[])"
        ),
        r"\g<head>[REDACTED]@",
    ),
    # Credential headers: ``Authorization: <scheme> <credentials>`` with any
    # scheme (``Token``, ``Digest`` …) and a ``Cookie:`` header's value
    # (2026-10-26-1, R20).
    (
        re.compile(
            r"(?i)\b(?P<head>(?:proxy-)?authorization\s*:\s*[A-Za-z]+\s+)[^\s\"'\[][^\s\"']*"
        ),
        r"\g<head>[REDACTED]",
    ),
    (
        re.compile(r"(?i)\b(?P<head>(?:set-)?cookie\s*:\s*)(?=[^\s=;\"']+=)[^\n\"']+"),
        r"\g<head>[REDACTED]",
    ),
    # Secret-named assignments (password=…, api_key: …, "token": "…"), value only.
    # The name may carry an identifier prefix, separated (DB_PASSWORD,
    # OPENAI_API_KEY) or not (PGPASSWORD, MYSQLPASSWORD), since ``_`` is a word
    # character and a leading ``\b`` never matched one (security review
    # 2026-10-24-2, P25, and 2026-10-25-1, Q28). A quoted value runs to its
    # closing quote, spaces and commas included (Q28). A quote may be
    # backslash-escaped, as in JSON echoed inside a string, and an unquoted
    # value runs to whitespace, so a ``,`` or ``;`` early in it leaves no
    # tail unmasked (R20).
    (
        re.compile(
            r"(?i)(?P<key>(?<![A-Za-z0-9])[A-Za-z0-9_-]{0,64}?"
            r"(?:password|passwd|pwd|secret|client[_-]?secret|"
            r"(?:secret|private|access)[_-]?(?:access[_-]?)?key|api[_-]?key|"
            r"access[_-]?token|auth[_-]?token|refresh[_-]?token|token)"
            r"(?![A-Za-z0-9_])(?:\\?[\"'])?\s*[:=]\s*)"
            r"(?:(?P<esc>\\?)(?P<quote>[\"'])"
            r"(?P<quoted>(?:(?!(?P=esc)(?P=quote))[^\n]){6,}?)(?P=esc)(?P=quote)"
            r"|(?P<value>[^\s\"']{6,}))"
        ),
        lambda m: (
            f"{m['key']}{m['esc']}{m['quote']}[REDACTED]{m['esc']}{m['quote']}"
            if m["quote"]
            else f"{m['key']}[REDACTED]"
        ),
    ),
]


def redact_secrets(text: str) -> str:
    """Mask common credential shapes.

    Deterministic pattern pass over harvested text before it reaches the
    corpus (distilled transcripts, memory files, rule files): ``sk-…`` keys,
    AWS access key IDs, GitHub, Slack, Google, Stripe, GitLab, Hugging Face
    and npm tokens, JWTs, ``Bearer`` and ``Basic`` credentials, PEM blocks,
    the password in URL userinfo (``scheme://user:password@``), and the value
    of a secret-named ``key=value`` or ``key: value`` assignment, prefixed
    names (``DB_PASSWORD=``, ``PGPASSWORD=``, ``*_API_KEY=``, ``*_TOKEN=``) and
    quoted values with spaces or commas included. Best-effort defence in
    depth — stated as such in the docs.
    """
    # PEM blocks first (they may contain substrings the later patterns match).
    text = _mask_pem_blocks(text)
    for pattern, replacement in _REDACTIONS:
        if pattern is ENGINE_KEY_SHAPE and isinstance(replacement, str):
            # Also a key split by an invisible character or an escaped
            # separator (security review 2026-10-30-1, E33).
            text = mask_engine_keys(text, replacement)
        else:
            text = pattern.sub(replacement, text)
    return text
