# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""An exporter writes no value into its output except through the escaping layer.

The static half of the exporter escaping rule (security review 2026-10-28-1,
O1, O2): every value an exporter module interpolates into an f-string must be
a call to one of :mod:`particles.render.escape`'s functions, a number with a
numeric format, or an expression this file names as engine-made (a count, a
constant, a slug the engine already escaped, a composed fragment whose parts
were escaped where they were made). A new stored field written raw fails here
with its file and line, in the style of ``tests/test_egress_chokepoint.py``.

A string concatenated with ``+`` and appended, joined or returned is checked
operand by operand, and so are module-level statements and the methods of
nested classes; an escape counts only when called by its bare name or on the
escaping module (security review 2026-10-30-1, AB73).

What the check cannot see, and what covers it instead: a value that reaches
the output without an f-string or a concatenation (appended to a line list as
is, or joined) is not inspected, and neither is what a helper returns.
``tests/test_export_escaping.py`` covers those behaviourally: it fills every
stored field of a scratch store with hostile text, runs every exporter, and
parses what each wrote.

An f-string that never reaches exporter output is skipped: a logging call, an
exception message, a path built with ``/``, and a lookup key.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

import particles
import particles.render.escape as escape

ROOT = Path(__file__).resolve().parents[1] / "particles"


def _module_file(rel: str) -> Path:
    """``rel`` under ``ROOT``, else wherever the ``particles`` package keeps it.

    A published engine tree holds only its own modules; the Client ones (such
    as ``render/markdown.py``) come from the installed core distribution.
    """
    path = ROOT / rel
    if path.is_file():
        return path
    for base in particles.__path__:
        if (Path(base) / rel).is_file():
            return Path(base) / rel
    return path


#: Calls whose result is inert in the context it is written into.
ESCAPES = set(escape.__all__) | {
    # Escaping compositions that call the escaping layer themselves.
    "_claim_for_heading",
    "_attribution",
    "render_inline_tags",
    "_render_lint_callouts",
    "_source_link_line",
    # ``render/markdown.py``'s one-line digest form, and the sentinel escape
    # that keeps a MEMORY.md bullet inside its region.
    "_inert_line",
    "_inert_bullet",
    "neutralize_comment_delimiters",
    # A projected section's listing line (security review 2026-10-30-1, AB22).
    "inert_listing_text",
}

#: Engine-made expressions, per module: never a stored string, or a fragment
#: whose stored parts were escaped where it was built. Each entry is the
#: expression's source as ``ast.unparse`` writes it.
ENGINE_MADE: dict[str, set[str]] = {
    "exporters/anki.py": {
        # A card's front, composed from the subject name and a fixed label; the
        # whole front is written through ``anki_field`` on its card line.
        "subject_name",
        "label",
        # The deck path: the operator's ``--deck-name`` option, one line, and the
        # subject name through ``anki_deck``; the threshold the operator set.
        "' '.join(deck_name.split())",
        "full_deck",
        "min_particle_confidence",
    },
    "exporters/wiki.py": {
        "count",  # a particle count
        "suffix",  # the qualifier (escaped) and the count
        "slug",  # a narrative's filename-sanitised slug, in a path
        "rel",  # that path, listed in the summary
    },
    "exporters/obsidian/format.py": {
        "ordinal",  # a reference's number
        "sid",  # an eight-hex citation id the regex matched in the body
        "_to_superscript(ordinal)",
        # The frontmatter annotation: an input hash and a synthesis state the
        # engine computes, spliced into the note the engine rendered.
        "article_input_hash",
        "article_synthesis",
        "body",
        "extras",
        # The synthesis splice: parts of the note and the article the engine
        # already rendered (the prose is neutralized by the article renderer).
        "body.lstrip()",
        "prose",
        "existing_h1",
        "rest",
        "references",
        # The rest of the note the engine rendered, after its frontmatter.
        "note[m.end():]",
    },
    "exporters/obsidian/vault.py": {
        "yaml_label",  # a frontmatter key from the fixed property table
        "nid",  # a stored Numista id, written only when it is all digits
        "_section_titles[section]",  # a fixed section title
        "tag",  # a fixed class tag from ``_PIVOT_TAG``
        "str(len(particles) == 0).lower()",
        "count",  # a particle count
        "suffix",  # the qualifier (escaped) and the count
        # The index table's external-id cell: ``md_code_span`` spans whose
        # pipes are escaped so the cell stays one column.
        "ext",
    },
    "exporters/logseq/format.py": {
        # ``render_block`` and ``render_property`` write their content
        # verbatim; this guard checks every caller's arguments instead.
        "_INDENT * depth",
        "_INDENT * (depth + 1)",
        "content",
        "key",
        "value",
    },
    "exporters/logseq/synthesis.py": {
        # The article's prose and References, which the article renderer
        # neutralized, and the heading and properties the engine names.
        "heading",
        "indent",
        "prose_lines[0]",
        "cont",
        "_INDENT * (depth + 1)",
        "line",
        "key",
        "prefix",
        "value",
    },
    "exporters/logseq/vault.py": {
        # A narrative's page name: a filename-sanitised slug, written through
        # ``logseq_page_ref`` wherever it is a reference.
        "slug",
    },
    "render/article_synthesis/render.py": {
        # A lint callout's type and header: fixed finding types and severities.
        "callout",
        "ft",
        "body",  # the frontmatter ``yaml.safe_dump`` wrote
        "count",  # a contradiction count
        # ``apply_lint_callouts`` splices the callouts into the rendered article.
        "stripped[:pos]",
        "stripped[pos:].lstrip(chr(10))",
        "stripped[:end]",
        "stripped[end:].lstrip(chr(10))",
    },
    "operations/projection/project.py": {
        "_short_id(particle.id)",  # the hex short id of a particle's UUID
        "_sources_trailer(short_ids)",  # a comment block of those short ids
    },
    "render/markdown.py": {
        # Fixed vocabularies: the contested badge's bases and caveat, a lint
        # finding's type and severity, a particle's status, the callout types.
        "', '.join(badge.bases)",
        "badge.caveat",
        "finding.finding_type",
        "finding.severity",
        "finding_type",
        "count",
        "particle.status",
        "callout_type",
        # A MEMORY.md bullet's badge: the contested bases are badge labels, and
        # ``versus`` holds only an identifier.
        "', '.join(bases)",
        "versus",
    },
}

#: Receivers whose ``get`` / ``add`` take a lookup key or a set member, never
#: output: an f-string passed to any other ``add`` or ``get`` is checked.
_LOOKUP_RECEIVERS = {
    "known_names",  # the set of valid link targets
    "prior_synthesis_bodies",  # the prior notes, keyed by their file name
    "prior_synthesis_hashes",
}

#: The modules whose f-strings write exporter output.
GUARDED = [
    "exporters/anki.py",
    "exporters/wiki.py",
    "exporters/obsidian/format.py",
    "exporters/obsidian/narrative.py",
    "exporters/obsidian/synthesis.py",
    "exporters/obsidian/vault.py",
    "exporters/logseq/format.py",
    "exporters/logseq/narrative.py",
    "exporters/logseq/synthesis.py",
    "exporters/logseq/vault.py",
    "render/article_synthesis/render.py",
    "render/markdown.py",
    "operations/projection/project.py",
]

#: In ``render/markdown.py`` only the particle and lint renderers and the
#: MEMORY.md bullet write exported Markdown; the digest has its own one-line
#: form (``_inert_line``) and the other projection helpers write no stored text.
_MARKDOWN_RENDERERS = {
    "render_particle",
    "render_stance_callout",
    "render_contested_callout",
    "render_particles",
    "render_lint_finding",
    "render_lint_report",
    "format_memory_bullet",
}

#: Modules where only some functions write stored text into output, and which.
#: In ``operations/projection/project.py`` the section listing written without
#: synthesis carries belief text; the rest writes the operator's manifest
#: titles, paths and messages (security review 2026-10-30-1, AB22).
_SCOPED_RENDERERS = {
    "render/markdown.py": _MARKDOWN_RENDERERS,
    "operations/projection/project.py": {"_deterministic_section_body"},
}


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
    return None


#: Names the escaping module is imported under, for an ``escape.md_text(…)`` call.
_ESCAPE_MODULES = {"escape", "_escape"}


def _escape_call(node: ast.AST) -> bool:
    """Whether ``node`` calls an escape: by its bare name or on the escaping module.

    A method that only shares an escape's name (``s.notes.md_text()``) is no
    escape (security review 2026-10-30-1, AB73).
    """
    if not isinstance(node, ast.Call) or _call_name(node) not in ESCAPES:
        return False
    func = node.func
    return isinstance(func, ast.Name) or (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id in _ESCAPE_MODULES
    )


def _is_numeric_spec(value: ast.FormattedValue) -> bool:
    spec = value.format_spec
    if not isinstance(spec, ast.JoinedStr):
        return False
    text = "".join(v.value for v in spec.values if isinstance(v, ast.Constant))
    return bool(text) and text[-1] in "dfeg%"


def _skipped(tree: ast.AST) -> set[int]:
    """Ids of f-strings that never reach output, or whose output an escape wraps."""
    skip: set[int] = set()

    def mark(node: ast.AST) -> None:
        skip.update(id(n) for n in ast.walk(node) if isinstance(n, ast.JoinedStr))

    for node in ast.walk(tree):
        name = _call_name(node)
        if isinstance(node, ast.Call):
            func = node.func
            owner = func.value if isinstance(func, ast.Attribute) else None
            if _escape_call(node):
                for arg in [*node.args, *(k.value for k in node.keywords)]:
                    mark(arg)
            elif (
                name in {"get", "add"}
                and isinstance(owner, ast.Name)
                and owner.id in _LOOKUP_RECEIVERS
            ) or (isinstance(owner, ast.Name) and owner.id in {"log", "logger", "logging"}):
                mark(node)
            for kw in node.keywords:
                if kw.arg == "progress_prefix":
                    mark(kw.value)
        elif isinstance(node, ast.Raise) and node.exc is not None:
            mark(node.exc)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            mark(node.right)
        elif isinstance(node, ast.Subscript):
            mark(node.slice)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Subscript) for t in node.targets
        ):
            for t in node.targets:
                mark(t)
    # An f-string bound to a name that is read only inside escape calls (a
    # label later written through ``md_code_span``) never reaches the output.
    escaped_reads: set[int] = set()
    for node in ast.walk(tree):
        if _escape_call(node):
            assert isinstance(node, ast.Call)
            for arg in [*node.args, *(k.value for k in node.keywords)]:
                escaped_reads.update(id(n) for n in ast.walk(arg) if isinstance(n, ast.Name))
    reads: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            reads.setdefault(node.id, []).append(id(node))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.JoinedStr)
        ):
            uses = reads.get(node.targets[0].id, [])
            if uses and all(u in escaped_reads for u in uses):
                mark(node.value)
    return skip


#: In ``render/article_synthesis/render.py`` these build the synthesis prompt,
#: which goes to the model inside a data fence, not into an export.
_PROMPT_BUILDERS = {
    "_format_particles_for_prompt",
    "_format_misalignments_for_prompt",
    "_framing_block",
    "_direction_block",
    "_synthesis_user_turn",
    "_build_synthesis_prompt",
}


#: A module-level constant (``_INDENT``, ``REFERENCES_HEADING``). A name of
#: this shape counts only when the module binds it at its top level (or
#: imports it), so a local spelled in capitals is checked like any other
#: (security review 2026-10-28-2, AA63).
_CONSTANT = re.compile(r"_?[A-Z][A-Z0-9_]*")
#: Calls that return a number or a timestamp, never stored text.
_VALUE_CALLS = {"len", "sum", "isoformat"}
#: String methods that keep text as inert as it was. ``replace`` is not one:
#: with constant arguments it can still undo an escape it is applied to.
_STRING_METHODS = {"strip", "rstrip", "lstrip", "lower", "upper"}
#: Functions that write their arguments into the output verbatim, so each
#: argument is checked like an interpolated value.
_SINKS = {"render_block", "render_property"}


def _safe_value(node: ast.AST, escaped: set[str], allowed: set[str]) -> bool:
    """Whether ``node`` evaluates to text an escape made, a constant, or an allowed expression."""
    if isinstance(node, ast.Constant) or ast.unparse(node) in allowed:
        return True
    name = _call_name(node)
    if _escape_call(node) or name in _VALUE_CALLS:
        return True
    if isinstance(node, ast.Name) and node.id in escaped:
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        # ``"sep".join(<escaped items>)``, and a string method over escaped text.
        if name == "join" and isinstance(node.func.value, ast.Constant) and len(node.args) == 1:
            items = node.args[0]
            if isinstance(items, ast.GeneratorExp | ast.ListComp):
                return _safe_value(items.elt, escaped, allowed)
            if isinstance(items, ast.List | ast.Tuple):
                return all(_safe_value(e, escaped, allowed) for e in items.elts)
        if name in _STRING_METHODS and all(isinstance(a, ast.Constant) for a in node.args):
            return _safe_value(node.func.value, escaped, allowed)
    if isinstance(node, ast.IfExp):
        return _safe_value(node.body, escaped, allowed) and _safe_value(
            node.orelse, escaped, allowed
        )
    if isinstance(node, ast.JoinedStr):
        return all(
            not isinstance(v, ast.FormattedValue)
            or _is_numeric_spec(v)
            or _safe_value(v.value, escaped, allowed)
            for v in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _safe_value(node.left, escaped, allowed) and _safe_value(
            node.right, escaped, allowed
        )
    return False


def _escaped_names(scope: ast.AST, allowed: set[str]) -> set[str]:
    """Local names every assignment of which is safe (to a fixed point)."""
    assigns: dict[str, list[ast.AST]] = {}
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigns.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                assigns.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            assigns.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            # A comprehension's name is its own scope, so only a loop's binds here.
            assigns.setdefault(node.target.id, []).append(ast.Name(id="<loop>"))
    escaped: set[str] = set()
    while True:
        grown = {
            name
            for name, values in assigns.items()
            if all(_safe_value(v, escaped, allowed) for v in values)
        }
        if grown <= escaped:
            return escaped
        escaped |= grown


def _module_constants(tree: ast.Module) -> set[str]:
    """Constant-shaped names the module binds at its top level or imports."""
    names: set[str] = set()
    for node in tree.body:
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return {n for n in names if _CONSTANT.fullmatch(n)}


def _top_level_functions(body: list[ast.stmt]) -> list[ast.AST]:
    """The scopes of a module or class body: its functions, the methods of its
    classes at any depth, and its own statements as one more scope.

    A value written at module level or inside a nested class is checked like
    one written in a function (security review 2026-10-30-1, AB73).
    """
    found: list[ast.AST] = []
    rest: list[ast.stmt] = []
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            found.append(node)
        elif isinstance(node, ast.ClassDef):
            found.extend(_top_level_functions(node.body))
        else:
            rest.append(node)
    if rest:
        found.append(ast.Module(body=rest, type_ignores=[]))
    return found


def _concatenations(scope: ast.AST) -> list[ast.BinOp]:
    """String concatenations written to output: appended, joined or returned."""
    written: list[ast.AST] = []
    for node in ast.walk(scope):
        if isinstance(node, ast.Return) and node.value is not None:
            written.append(node.value)
        elif isinstance(node, ast.Call) and _call_name(node) in {"append", "join"}:
            for arg in node.args:
                if isinstance(arg, ast.List | ast.Tuple):
                    written.extend(arg.elts)
                elif isinstance(arg, ast.GeneratorExp | ast.ListComp):
                    written.append(arg.elt)
                else:
                    written.append(arg)
    return [
        n
        for n in written
        if isinstance(n, ast.BinOp)
        and isinstance(n.op, ast.Add)
        and any(
            isinstance(leaf, ast.JoinedStr)
            or (isinstance(leaf, ast.Constant) and isinstance(leaf.value, str))
            for leaf in ast.walk(n)
        )
    ]


def _functions(tree: ast.Module, names: set[str]) -> list[ast.AST]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name in names
    ]


def _safe_operands(node: ast.AST, escaped: set[str], allowed: set[str]) -> bool:
    """Whether every operand of a string concatenation is inert.

    An f-string operand is checked on its own, and ``"sep".join(lines)`` joins
    lines checked where they were built, so only the other operands are judged
    here.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _safe_operands(node.left, escaped, allowed) and _safe_operands(
            node.right, escaped, allowed
        )
    if isinstance(node, ast.JoinedStr):
        return True
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and isinstance(node.func.value, ast.Constant)
    ):
        return True
    return _safe_value(node, escaped, allowed)


def raw_interpolations(rel: str) -> list[str]:
    """Every value ``rel`` interpolates into an f-string with no escape or allowance."""
    path = _module_file(rel)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if rel in _SCOPED_RENDERERS:
        scopes = _functions(tree, _SCOPED_RENDERERS[rel])
    else:
        prompt = {id(f) for f in _functions(tree, _PROMPT_BUILDERS)}
        scopes = [n for n in _top_level_functions(tree.body) if id(n) not in prompt]
    allowed = ENGINE_MADE.get(rel, set()) | _module_constants(tree)
    found: list[str] = []
    for scope in scopes:
        skip = _skipped(scope)
        escaped = _escaped_names(scope, allowed)
        for node in ast.walk(scope):
            if _call_name(node) in _SINKS:
                assert isinstance(node, ast.Call)
                for arg in node.args:
                    if not isinstance(arg, ast.JoinedStr) and not _safe_value(
                        arg, escaped, allowed
                    ):
                        found.append(f"{rel}:{arg.lineno}: {_call_name(node)}({ast.unparse(arg)})")
            if not isinstance(node, ast.JoinedStr) or id(node) in skip:
                continue
            for value in node.values:
                if not isinstance(value, ast.FormattedValue):
                    continue
                if _is_numeric_spec(value) or _safe_value(value.value, escaped, allowed):
                    continue
                found.append(f"{rel}:{value.lineno}: {{{ast.unparse(value.value)}}}")
        for concat in _concatenations(scope):
            if id(concat) not in skip and not _safe_operands(concat, escaped, allowed):
                found.append(f"{rel}:{concat.lineno}: {ast.unparse(concat)}")
    return sorted(set(found))


def test_every_exporter_interpolation_is_escaped_or_engine_made() -> None:
    """No exporter module formats a value into its output outside the escaping layer."""
    found = [line for rel in GUARDED for line in raw_interpolations(rel)]
    assert not found, (
        "These values reach exporter output with no particles.render.escape function. "
        "Write a stored value through the function for its context; add an expression "
        "to ENGINE_MADE only when the engine makes it:\n" + "\n".join(found)
    )


def test_the_guard_flags_a_raw_stored_field() -> None:
    """The check fails on an f-string that writes a stored attribute raw."""
    tree = ast.parse('def f(s):\n    return f"# {s.canonical_name}"\n')
    values = [
        ast.unparse(v.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.JoinedStr) and id(n) not in _skipped(tree)
        for v in n.values
        if isinstance(v, ast.FormattedValue) and not _escape_call(v.value)
    ]
    assert values == ["s.canonical_name"]


def test_every_guarded_module_exists() -> None:
    """A renamed exporter module is not silently dropped from the check."""
    missing = [rel for rel in GUARDED if not _module_file(rel).is_file()]
    assert not missing, missing


#: One raw stored-field write in each syntactic form the check walks.
_RAW_WRITE_FORMS = {
    "concatenation": (
        "def render(s):\n"
        "    lines = []\n"
        "    lines.append('# ' + s.canonical_name)\n"
        "    return '\\n'.join(lines)\n"
    ),
    "module_level": "import sys\nHEADING = f'# {sys.argv[1]}'\n",
    "nested_class": (
        "class Outer:\n"
        "    class Inner:\n"
        "        def render(self, s):\n"
        "            return f'# {s.canonical_name}'\n"
    ),
    "escape_name_as_method": "def render(s):\n    return f'# {s.notes.md_text()}'\n",
}


@pytest.mark.parametrize("form", sorted(_RAW_WRITE_FORMS))
def test_the_guard_reports_a_raw_write_in_every_form(
    form: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check reports a raw stored-field write whatever syntactic form carries it (AB73)."""
    rel = f"exporters/proof_{form}.py"
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text(_RAW_WRITE_FORMS[form], encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    assert raw_interpolations(rel), f"{form}: the raw write passed the guard"


def test_the_guard_accepts_an_escape_called_on_the_escaping_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``escape.md_text(…)`` counts as an escape; only a look-alike method does not."""
    rel = "exporters/proof_module_call.py"
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text(
        "from particles.render import escape\n\n"
        "def render(s):\n    return f'# {escape.md_text(s.canonical_name)}'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    assert raw_interpolations(rel) == []
