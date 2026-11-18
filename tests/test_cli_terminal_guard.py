# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""No CLI verb or script prints a stored text field to the terminal raw.

A claim, a subject name, a sheet row or a corpus passage is text that a source
author, an agent key or an imported bundle chose. Printed raw, a control
character in it reaches the operator's terminal and is acted on there. Every
such print goes through ``terminal_safe``, ``terminal_field`` or
``terminal_echo`` (``particles/api/cli/_terminal.py``). The behavioural tests
(``test_cli_terminal.py``, ``test_security_cli_output_*.py``) prove that for
the verbs they drive; this guard is the class fix the CI security review of
the re-read labeller asked for (S1, Low), so the next print of a stored field
fails here instead of in a later review.

The check is an AST scan, and it is deliberately narrow so that it lands with
a short reviewed allowlist instead of a wall of exemptions:

* **Sources** are reads of a field whose name says it holds stored text: an
  attribute, a subscript or a ``.get()`` key in :data:`STORED_FIELDS`, or a key
  ending ``_text`` / ``_content`` (a sheet column such as ``survivor_text``).
  Generic names (``name``, ``text``, ``label``, ``value``) are left out: they
  name file paths, enum members and the repository's own prose far more often
  than stored text, and including them is what made a wider check noisy.
* **Taint** follows a source through assignments, loop and comprehension
  targets, f-strings, string methods, slicing, the string builtins in
  :data:`PASSING_CALLS`, and calls to a function of the same module whose
  result carries its arguments' text (``word_diff`` in the labeller did,
  until it ran its readings through ``terminal_safe``). It reads one function
  at a time and in text order: a use sees the binding before it, so a name
  bound again to a clean value is clean from there on, but only where that
  binding dominates (a clean ``else`` does not clear what the ``if`` bound),
  and a loop's back edge is not followed. A comparison is a bool and carries nothing. It does not
  follow a value through a call into another module: a renderer imported
  from ``particles.operations`` is that module's concern, and following every
  call made the scan flag cost estimates computed from a passage's length.
* **Sinks** are ``print`` (unless its ``file`` is not a standard stream),
  ``typer.echo`` / ``typer.secho``, ``click.echo``, a rich ``console.print``
  and ``sys.stdout.write`` / ``sys.stderr.write``, with every argument,
  keyword ones included. A value is clean once it
  passes through a sanitizer in :data:`SANITIZERS` (the terminal helpers,
  ``repr`` and a ``!r`` conversion, which escape every non-printable
  character, and a JSON serializer, which a machine reads).

What it cannot see: a stored value that reaches a print through a renderer in
another module, a method or a nested function (only top-level functions of
the module are followed), an object attribute set elsewhere, or a field name
outside the vocabulary. Those stay with the behavioural tests and review.
"""

from __future__ import annotations

import ast
import functools
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests import _source_tree

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Field names that hold stored text: particle content, subject names and
#: aliases, claims and passages. A key ending ``_text`` or ``_content`` counts
#: too (:func:`_is_stored_key`).
STORED_FIELDS = frozenset(
    {
        "aliases",
        "canonical_name",
        "claim",
        "claim_text",
        "content",
        "description",
        "excerpt",
        "passage",
        "snippet",
        "statement",
    }
)

#: Calls whose result is clean whatever their arguments hold.
SANITIZERS = frozenset(
    {
        "terminal_safe",
        "terminal_field",
        "terminal_echo",
        "repr",
        "dumps",
        "model_dump_json",
        "len",
        "bool",
        "int",
        "float",
        "round",
        "hash",
        "id",
    }
)

#: Calls (by name) that hand their arguments' text on to their result.
PASSING_CALLS = frozenset(
    {
        "str",
        "format",
        "join",
        "clip",
        "indented",
        "shorten",
        "fill",
        "wrap",
        "dedent",
        "indent",
        "list",
        "tuple",
        "sorted",
        "reversed",
        "enumerate",
        "zip",
        "min",
        "max",
        "findall",
        "sub",
        "split",
    }
)

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class RawPrint:
    """One print of a stored field that no sanitizer covers."""

    path: str
    function: str
    line: int
    value: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.path, self.function)


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_stored_key(key: object) -> bool:
    return isinstance(key, str) and (
        key in STORED_FIELDS or key.endswith("_text") or key.endswith("_content")
    )


_STREAMS = ("sys.stdout", "sys.stderr")


def _is_sink(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name) and func.id == "print":
        # ``print(..., file=fh)`` writes a file, not the terminal.
        stream = next((kw.value for kw in call.keywords if kw.arg == "file"), None)
        return stream is None or ast.unparse(stream) in _STREAMS
    if isinstance(func, ast.Attribute):
        base = ast.unparse(func.value)
        if func.attr in ("echo", "secho"):
            return base in ("typer", "click")
        if func.attr == "write":
            return base in _STREAMS
        if func.attr == "print":
            return "console" in base.lower()  # a rich Console
    return False


def _printed(call: ast.Call) -> list[ast.expr]:
    """The values a sink prints: its arguments, keyword ones included, but not ``file``."""
    return [*call.args, *(kw.value for kw in call.keywords if kw.arg != "file")]


def _scope_nodes(scope: ast.AST) -> list[ast.AST]:
    """Every node of ``scope`` outside the functions, lambdas and classes it defines."""
    nodes: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if isinstance(node, (*_FUNCTIONS, ast.Lambda, ast.ClassDef)):
            continue
        nodes.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return nodes


def _names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {n for elt in target.elts for n in _names(elt)}
    if isinstance(target, ast.Starred):
        return _names(target.value)
    return set()


_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)

Pos = tuple[int, int]

#: The chain of statement lists a node sits in, outermost first: each link is
#: (id of the compound statement, the field holding the list).
Block = tuple[tuple[int, str], ...]

_BLOCK_FIELDS = ("body", "orelse", "finalbody", "handlers", "cases")


@dataclass(frozen=True)
class _Binding:
    """A name bound at ``pos``; ``additive`` when it only adds (``lines.append(x)``).

    ``block`` is where the binding sits: a clean rebinding clears a name only
    for the uses it dominates, those in its own block or one nested in it.
    ``within`` holds the nodes of a comprehension, the only uses that see its
    target.
    """

    pos: Pos
    value: ast.AST | None
    block: Block = ()
    additive: bool = False
    seed: bool = False
    within: frozenset[int] | None = None


def _end(node: ast.AST) -> Pos:
    return (getattr(node, "end_lineno", 0) or 0, getattr(node, "end_col_offset", 0) or 0)


def _start(node: ast.AST) -> Pos:
    return (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))


def _blocks(scope: ast.AST) -> dict[int, Block]:
    """The block of every node in ``scope``, by node id, nested definitions left out."""
    out: dict[int, Block] = {}

    def walk(node: ast.AST, block: Block) -> None:
        out[id(node)] = block
        for field, value in ast.iter_fields(node):
            inner = (*block, (id(node), field)) if field in _BLOCK_FIELDS else block
            for child in value if isinstance(value, list) else [value]:
                if isinstance(child, ast.AST) and not isinstance(
                    child, (*_FUNCTIONS, ast.Lambda, ast.ClassDef)
                ):
                    walk(child, inner)

    for child in ast.iter_child_nodes(scope):
        if not isinstance(child, (*_FUNCTIONS, ast.Lambda, ast.ClassDef)):
            walk(child, ())
    return out


def _bindings(
    scope: ast.AST, nodes: list[ast.AST], blocks: dict[int, Block]
) -> dict[str, list[_Binding]]:
    """Every binding in one function body, with the value whose taint it takes.

    A binding takes effect where its statement ends, so ``x = f(x)`` reads the
    ``x`` bound before it, and a comprehension's target where the comprehension
    starts, so its element sees it. A loop's back edge is not followed: a use
    sees the bindings that precede it in the text.
    """
    out: dict[str, list[_Binding]] = {}

    def bind(names: set[str], binding: _Binding) -> None:
        for name in names:
            out.setdefault(name, []).append(binding)

    if isinstance(scope, _FUNCTIONS):
        args = scope.args
        params = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
        bind({a.arg for a in params if a is not None}, _Binding(_start(scope), None, seed=True))
    for node in nodes:
        block = blocks.get(id(node), ())
        match node:
            case ast.Assign(targets=targets, value=value):
                for target in targets:
                    bind(_names(target), _Binding(_end(node), value, block))
            case ast.AnnAssign(target=target, value=value) if value is not None:
                bind(_names(target), _Binding(_end(node), value, block))
            case ast.AugAssign(target=target, value=value):
                bind(_names(target), _Binding(_end(node), value, block, additive=True))
            case ast.NamedExpr(target=target, value=value):
                bind(_names(target), _Binding(_end(node), value, block))
            case ast.For(target=target, iter=it) | ast.AsyncFor(target=target, iter=it):
                bind(_names(target), _Binding(_end(it), it, block))
            case ast.withitem(context_expr=expr, optional_vars=target) if target is not None:
                bind(_names(target), _Binding(_end(target), expr, block))
            case ast.Expr(
                value=ast.Call(func=ast.Attribute(value=ast.Name(id=name), attr=attr), args=args)
            ) if attr in ("append", "extend", "insert", "add", "update"):
                added = ast.Tuple(elts=list(args))
                bind({name}, _Binding(_end(node), added, block, additive=True))
            case _ if isinstance(node, _COMPREHENSIONS):
                inside = frozenset(id(n) for n in ast.walk(node))
                for gen in node.generators:
                    bind(_names(gen.target), _Binding(_start(node), gen.iter, within=inside))
    for bindings in out.values():
        bindings.sort(key=lambda b: b.pos)
    return out


class _Taint:
    """The stored text an expression carries, within one function body."""

    def __init__(
        self,
        scope: ast.AST,
        nodes: list[ast.AST],
        passing: set[str],
        seeded: bool = False,
    ) -> None:
        self.passing = passing
        self.seeded = seeded
        self.blocks = _blocks(scope)
        self.bindings = _bindings(scope, nodes, self.blocks)
        self._cache: dict[int, str | None] = {}
        self._active: set[int] = set()

    def _name(self, node: ast.Name) -> str | None:
        """The name, when a binding that reaches this use carries stored text.

        The bindings before the use are read latest first. A carrying one
        taints the use; a clean one stops the search only when it dominates
        the use (its block holds the use's), so a name bound clean in one
        branch still carries the text another branch bound it to.
        """
        pos = _start(node)
        block = self.blocks.get(id(node), ())
        for binding in reversed(self.bindings.get(node.id, [])):
            if binding.pos >= pos:
                continue
            if binding.within is not None and id(node) not in binding.within:
                continue
            if binding.seed:
                return node.id if self.seeded else None
            if self._binding(binding) is not None:
                return node.id
            if not binding.additive and block[: len(binding.block)] == binding.block:
                return None
        return None

    def _binding(self, binding: _Binding) -> str | None:
        key = id(binding)
        if key in self._cache:
            return self._cache[key]
        if key in self._active:  # a binding whose value reads itself
            return None
        self._active.add(key)
        try:
            found = self.of(binding.value)
        finally:
            self._active.discard(key)
        self._cache[key] = found
        return found

    def of(self, node: ast.AST | None) -> str | None:
        """Source text of the first stored value ``node`` carries, or ``None``."""
        if node is None:
            return None
        match node:
            case ast.Name():
                return self._name(node)
            case ast.Attribute(attr=attr) if _is_stored_key(attr):
                return ast.unparse(node)
            case ast.Subscript(slice=ast.Constant(value=key)) if _is_stored_key(key):
                return ast.unparse(node)
            case ast.FormattedValue(value=value, conversion=conversion):
                # ``!r`` escapes every non-printable character.
                return None if conversion == ord("r") else self.of(value)
            case ast.Compare() | ast.UnaryOp(op=ast.Not()):
                return None  # a bool
            case ast.Call():
                return self._of_call(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.comprehension):
                continue  # its target is bound in ``_bindings``; its iter is that binding's value
            found = self.of(child)
            if found is not None:
                return found
        return None

    def _of_call(self, call: ast.Call) -> str | None:
        func = call.func
        name = _call_name(func)
        if name in SANITIZERS:
            return None
        if (
            name == "get"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and _is_stored_key(call.args[0].value)
        ):
            return ast.unparse(call)
        # A method of a carrying value (``claim.strip()``, ``text[:80].upper()``).
        if isinstance(func, ast.Attribute):
            found = self.of(func.value)
            if found is not None:
                return found
        if name in PASSING_CALLS or (isinstance(func, ast.Name) and name in self.passing):
            for arg in [*call.args, *(kw.value for kw in call.keywords)]:
                found = self.of(arg)
                if found is not None:
                    return found
        return None


def _passing_functions(tree: ast.Module) -> set[str]:
    """The module's own functions whose result carries the text of their arguments.

    Each is read with every parameter carrying stored text, to a fixed point
    (a helper may call another). ``word_diff`` in the re-read labeller passes
    its readings on unless it runs them through ``terminal_safe`` first.
    """
    functions = {n.name: n for n in tree.body if isinstance(n, _FUNCTIONS)}
    passing: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, fn in functions.items():
            if name in passing:
                continue
            nodes = _scope_nodes(fn)
            taint = _Taint(fn, nodes, passing, seeded=True)
            returns = [n.value for n in nodes if isinstance(n, ast.Return)]
            if any(taint.of(value) for value in returns):
                passing.add(name)
                changed = True
    return passing


def _direct_defs(scope: ast.AST) -> list[ast.AST]:
    """The functions and classes ``scope`` defines itself, not those inside them."""
    defs: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if isinstance(node, (*_FUNCTIONS, ast.ClassDef)):
            defs.append(node)
            continue
        stack.extend(ast.iter_child_nodes(node))
    return defs


def raw_prints(source: str, path: str) -> list[RawPrint]:
    """Every print in ``source`` of a stored field that no sanitizer covers."""
    tree = _source_tree.parse(source)
    passing = _passing_functions(tree)
    found: list[RawPrint] = []

    def visit(scope: ast.AST, qualname: str) -> None:
        nodes = _scope_nodes(scope)
        taint = _Taint(scope, nodes, passing)
        for node in nodes:
            if isinstance(node, ast.Call) and _is_sink(node):
                for arg in _printed(node):
                    value = taint.of(arg)
                    if value is not None:
                        found.append(RawPrint(path, qualname, node.lineno, value))
                        break
        for child in _direct_defs(scope):
            assert isinstance(child, (*_FUNCTIONS, ast.ClassDef))
            visit(child, f"{qualname}.{child.name}" if qualname else child.name)

    visit(tree, "")
    return found


#: Reviewed prints of a field the vocabulary names that is not stored text,
#: keyed on (path, function), each with the reason it stands.
ALLOWED: dict[tuple[str, str], str] = {
    ("particles/api/cli/mcp.py", "mcp_tools_cmd"): (
        "the first line of each MCP tool's description, which is this package's own docstring"
    ),
    ("particles/api/cli/mcp.py", "mcp_resources_cmd"): (
        "the first line of each MCP resource's description, this package's own text"
    ),
    ("particles/api/cli/memory.py", "memory_tools_cmd"): (
        "the first line of each memory-compat tool's description, this package's own docstring"
    ),
}

#: The trees the guard scans: every CLI verb, and the repository's scripts.
SCAN_ROOTS = ("particles/api/cli", "scripts")


@functools.cache
def _scan() -> tuple[RawPrint, ...]:
    """Every raw print under :data:`SCAN_ROOTS`, scanned once per session."""
    found: list[RawPrint] = []
    for root in SCAN_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for file in sorted(base.rglob("*.py")):
            rel = file.relative_to(REPO_ROOT).as_posix()
            found.extend(raw_prints(file.read_text(encoding="utf-8"), rel))
    return found


def test_no_stored_field_is_printed_raw() -> None:
    """Every print of a stored field goes through a terminal helper or is allowlisted."""
    raw = [r for r in _scan() if r.key not in ALLOWED]
    assert not raw, (
        "a stored text field reaches the terminal raw; wrap it in terminal_safe / "
        "terminal_field (particles/api/cli/_terminal.py), or, if it is not stored "
        "text, add a reviewed entry to ALLOWED:\n"
        + "\n".join(f"  {r.path}:{r.line} in {r.function or '<module>'}: {r.value}" for r in raw)
    )


def test_every_allowlist_entry_still_matches_a_print() -> None:
    """An allowlist entry whose print was fixed or removed is deleted, not kept.

    An entry for a file this tree does not hold (a published tree carries only
    some of the scanned roots) is not judged.
    """
    seen = {r.key for r in _scan()}
    stale = sorted(k for k in ALLOWED if k not in seen and (REPO_ROOT / k[0]).is_file())
    assert not stale, f"ALLOWED entries no print matches any more: {stale}"


@pytest.mark.parametrize(
    "source",
    [
        # The re-read labeller's shape (security review S1): a sheet column,
        # through a module-local helper, printed in an f-string.
        """
import sys
def word_diff(first, second):
    a, b = re.findall(r"\\S+", first), re.findall(r"\\S+", second)
    return " ".join(a), " ".join(b)
def run(row):
    first, second = word_diff(row.get("survivor_text", ""), row.get("copy_text", ""))
    print(f"first reading\\n  {first}\\n")
""",
        "import typer\ndef show(p):\n    typer.echo(f'  {p.content[:80]}')\n",
        "def show(s):\n    name = s.canonical_name.strip()\n    print('Subject: ' + name)\n",
        "import sys\ndef show(rows):\n    for r in rows:\n        sys.stdout.write(r['claim'])\n",
        # A clean ``else`` does not clear what the ``if`` bound.
        "def show(p, v):\n    if v:\n        line = p.content\n    else:\n        line = ''\n"
        "    print(line)\n",
        # A comprehension's target is its own: it does not clear the outer name.
        "def show(p, xs):\n    row = p.content\n    ids = [row for row in xs]\n    print(row)\n",
        "import typer\ndef show(p):\n    typer.echo(message=p.content)\n",
        "def show(console, p):\n    console.print(p.content)\n",
        "import sys\ndef show(p):\n    print(p.content, file=sys.stderr)\n",
        "def show(ps):\n    lines = []\n    for p in ps:\n        lines.append(p.content)\n"
        "    print('\\n'.join(lines))\n",
    ],
)
def test_the_guard_flags_a_raw_print_of_a_stored_field(source: str) -> None:
    assert raw_prints(source, "x.py"), source


@pytest.mark.parametrize(
    "source",
    [
        "from x import terminal_safe\ndef run(row):\n    print(terminal_safe(row['claim']))\n",
        "from x import terminal_echo\ndef run(p):\n    terminal_echo(f'  {p.content}')\n",
        "def run(p):\n    print(f'  {p.content!r}')\n",
        "import json\ndef run(p):\n    print(json.dumps({'c': p.content}))\n",
        "def run(p):\n    print(len(p.content))\n",
        # A call into another module is not followed.
        "from x import estimate\ndef run(p):\n    cost = estimate(p.content)\n    print(cost)\n",
        "def run(path):\n    print(path.name)\n",
        # A file is not the terminal.
        "def run(p, fh):\n    print(p.content, file=fh)\n",
        # A name bound again to a clean value is clean from there on.
        "def run(ps):\n    for row in ps:\n        row = {'c': row.content}\n"
        "    for row in totals():\n        print(row)\n",
        # The labeller once #1501 lands: the helper sanitizes what it returns.
        """
def word_diff(first, second):
    a = re.findall(r"\\S+", terminal_safe(first))
    return " ".join(a), ""
def run(row):
    first, second = word_diff(row.get("survivor_text", ""), "")
    print(first)
""",
        # A comparison is a bool, whatever it compares.
        "def run(rows):\n    copied = [r['content'] in seen for r in rows]\n"
        "    print(sum(copied))\n",
    ],
)
def test_the_guard_passes_a_sanitized_or_unrelated_print(source: str) -> None:
    assert not raw_prints(source, "x.py"), source
