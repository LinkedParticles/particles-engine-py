# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""The bounded YAML loader: alias expansion is held to a budget linear in the text."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import pytest
import yaml

from particles.core.bounded_yaml import (
    YAMLExpansionError,
    YAMLItemLimitError,
    load,
    load_sequence,
)


def _fan_out(levels: int, fan: int) -> str:
    lines = ["a0: &a0 [x]"]
    for i in range(1, levels + 1):
        lines.append(f"a{i}: &a{i} [" + ", ".join([f"*a{i - 1}"] * fan) + "]")
    lines.append(f"top: *a{levels}")
    return "\n".join(lines) + "\n"


def _mapping_fan_out(levels: int, fan: int, *, alias_as_key: bool) -> str:
    """The fan-out of ``_fan_out`` built from mappings, the alias as each value or each key."""
    lines = ["a0: &a0 {x: 1}"]
    for i in range(1, levels + 1):
        if alias_as_key:
            pairs = [f"? *a{i - 1}\n  : k{j}" for j in range(fan)]
            lines.append(f"a{i}: &a{i}\n  " + "\n  ".join(pairs))
        else:
            pairs = [f"k{j}: *a{i - 1}" for j in range(fan)]
            lines.append(f"a{i}: &a{i} {{" + ", ".join(pairs) + "}")
    return "\n".join(lines) + "\n"


def test_plain_yaml_loads_as_safe_load_does() -> None:
    text = "a: 1\nb: [x, y, {k: v}]\nc:\nd: 2026-05-12\n"
    assert load(text) == yaml.safe_load(text)


def test_a_modest_alias_still_resolves() -> None:
    assert load("base: &b {k: v}\nuse: *b\n") == {"base": {"k": "v"}, "use": {"k": "v"}}


def test_an_alias_fan_out_past_the_budget_is_refused() -> None:
    text = _fan_out(levels=7, fan=9)
    assert len(text) < 1024
    with pytest.raises(YAMLExpansionError):
        load(text)


def test_a_recursive_alias_is_refused() -> None:
    with pytest.raises(YAMLExpansionError):
        load("a: &a [*a]\n")


def test_the_refusal_is_a_yaml_error() -> None:
    assert issubclass(YAMLExpansionError, yaml.YAMLError)


def test_an_explicit_budget_is_honoured() -> None:
    with pytest.raises(YAMLExpansionError):
        load("a: &a [x, y]\nb: [*a, *a, *a]\n", max_nodes=8)


@pytest.mark.parametrize("alias_as_key", [False, True], ids=["value", "key"])
def test_a_mapping_fan_out_past_the_budget_is_refused(alias_as_key: bool) -> None:
    """An alias to a mapping counts every key and value it stands for."""
    text = _mapping_fan_out(levels=7, fan=9, alias_as_key=alias_as_key)
    assert len(text) < 1024
    with pytest.raises(YAMLExpansionError):
        load(text)


@pytest.mark.parametrize(
    "text",
    ["a: &a {x: 1}\nb: *a\n", "a: &a [x, y]\nb: *a\n"],
    ids=["mapping", "sequence"],
)
def test_the_budget_is_exact(text: str) -> None:
    """Six composed nodes plus an alias charged at its three: nine fit, eight do not."""
    assert load(text, max_nodes=9) == yaml.safe_load(text)
    with pytest.raises(YAMLExpansionError):
        load(text, max_nodes=8)


_DEEP = "k: " + "[" * 3000 + "]" * 3000
_LONG_INT = "k: " + "1" * 5000


@pytest.mark.parametrize("text", [_DEEP, _LONG_INT], ids=["deep", "long-int"])
def test_a_document_that_cannot_be_built_is_a_yaml_error(text: str) -> None:
    """Nesting past the stack or a number past the digit limit reads as malformed YAML."""
    with pytest.raises(yaml.YAMLError):
        load(text)


def test_a_base_sixty_number_costs_time_linear_in_its_length() -> None:
    """A long colon-separated number is refused before it is built."""
    import time

    start = time.perf_counter()
    with pytest.raises(YAMLExpansionError):
        load("k: 1" + ":1" * 64_000)
    assert time.perf_counter() - start < 0.5


def test_ordinary_nesting_and_numbers_still_load() -> None:
    nested = "k: " + "[" * 50 + "1" + "]" * 50
    assert load(nested) == yaml.safe_load(nested)
    assert load("n: 123456789012345678901234567890\nf: 1.5e+10\nt: 1:30\n") == {
        "n": 123456789012345678901234567890,
        "f": 1.5e10,
        "t": 90,
    }


def test_nesting_at_the_depth_bound_loads_and_one_past_is_refused() -> None:
    at_bound = "[" * 100 + "]" * 100
    assert load(at_bound) == yaml.safe_load(at_bound)
    with pytest.raises(YAMLExpansionError, match="nests deeper"):
        load("[" * 101 + "]" * 101)


def test_finished_siblings_do_not_reset_the_depth() -> None:
    """Each level holds an empty list before its deeper sibling; the depth still counts."""
    text = "[[], " * 150 + "[]" + "]" * 150
    with pytest.raises(YAMLExpansionError, match="nests deeper"):
        load(text)


def test_an_alias_at_the_depth_bound_loads_and_one_past_is_refused() -> None:
    """An alias three levels down counts its anchor's levels: 3 + 97 fits, 3 + 98 does not."""

    def doc(levels: int) -> str:
        return "a: &a " + "[" * levels + "]" * levels + "\nb: [[*a]]\n"

    assert load(doc(97)) == yaml.safe_load(doc(97))
    with pytest.raises(YAMLExpansionError, match="nests deeper"):
        load(doc(98))


def test_a_number_at_the_length_bound_loads_and_one_past_is_refused() -> None:
    assert load("n: " + "1" * 128) == {"n": int("1" * 128)}
    with pytest.raises(YAMLExpansionError, match="number scalar"):
        load("n: " + "1" * 129)


def _depth(value: object) -> int:
    depth = 0
    while isinstance(value, list) and value:
        value = value[0]
        depth += 1
    return depth


def _chained_header(anchors: int, levels: int = 98) -> str:
    lines = []
    for i in range(anchors):
        inner = f"*a{i - 1}" if i else "0"
        lines.append(f"a{i}: &a{i} " + "[" * levels + inner + "]" * levels)
    body = "\n".join(lines) + "\n"
    return "pad: '" + "x" * (8000 - len(body) - 10) + "'\n" + body


def test_a_loaded_value_nests_no_deeper_than_the_depth_bound() -> None:
    """Anchors chained through aliases never build a value past the depth bound (O64)."""
    from particles.core import bounded_yaml

    text = _chained_header(18)
    assert len(text) < 8192
    try:
        value = load(text)
    except YAMLExpansionError:
        return
    deepest = max(_depth(v) for v in value.values() if isinstance(v, list))
    assert deepest <= bounded_yaml._MAX_DEPTH, deepest


def test_an_alias_within_the_depth_bound_still_loads() -> None:
    """Control: an alias whose levels fit under the bound loads as ``safe_load`` does."""
    text = "a: &a [[[1]]]\nb: [*a, *a]\n"
    assert load(text) == yaml.safe_load(text)


@pytest.mark.parametrize(
    "text",
    [
        "base: &b {k: 1, j: 2}\nm:\n  <<: *b\n  k: 3\n",
        "x: &x {a: 1}\ny: &y {b: 2}\nm:\n  <<: [*x, *y]\n  c: 3\n",
        "x: &x {a: 1}\ny: &y {a: 2}\nm:\n  <<: [*x, *y]\n",
        "x: &x {a: 1}\nm:\n  <<: *x\n  <<: {a: 5, z: 0}\n  q: 1\n",
        "m:\n  =: v\n  k: 1\n",
    ],
)
def test_merge_keys_load_as_safe_load_does(text: str) -> None:
    assert load(text) == yaml.safe_load(text)


@pytest.mark.parametrize("text", ["m:\n  <<: 1\n", "m:\n  <<: [1]\n"])
def test_a_bad_merge_is_refused_as_safe_load_refuses_it(text: str) -> None:
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load(text)
    with pytest.raises(yaml.constructor.ConstructorError):
        load(text)


def _merge_doc(count: int) -> str:
    return "- &a {k: 1}\n- \n" + "".join("  <<: *a\n" for _ in range(count))


class _MeteredPairs(list[tuple[yaml.Node, yaml.Node]]):
    """A mapping node's pair list that charges ``meter`` the slots each operation touches.

    A walk reads every slot, a delete or insert at ``i`` moves every slot after
    ``i``, and a concatenation or extension copies every slot it adds. That move
    is where an in-place merge spends its quadratic time, which only a wall
    clock saw before. The meter sees these lists only: work done on a plain
    copy of one, or on a local list, is not counted.
    """

    def __init__(self, items: list[tuple[yaml.Node, yaml.Node]], meter: list[int]) -> None:
        super().__init__(items)
        self.meter = meter

    def _charge(self, slots: int) -> None:
        self.meter[0] += slots

    def _start(self, index: Any) -> int:
        if isinstance(index, slice):
            return index.indices(len(self))[0]
        return min(index + len(self) if index < 0 else index, len(self))

    def __iter__(self) -> Any:
        self._charge(len(self))
        return super().__iter__()

    def __reversed__(self) -> Any:
        self._charge(len(self))
        return super().__reversed__()

    def __contains__(self, item: object) -> bool:
        self._charge(len(self))
        return super().__contains__(item)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            self._charge(len(range(*index.indices(len(self)))))
        else:
            self._charge(1)
        return super().__getitem__(index)

    def __setitem__(self, index: Any, value: Any) -> None:
        if isinstance(index, slice):
            value = list(value)
            self._charge(len(self) - self._start(index) + len(value))
        else:
            self._charge(1)
        super().__setitem__(index, value)

    def __delitem__(self, index: Any) -> None:
        self._charge(len(self) - self._start(index))
        super().__delitem__(index)

    def insert(self, index: Any, value: Any) -> None:
        self._charge(len(self) - self._start(index) + 1)
        super().insert(index, value)

    def pop(self, index: Any = -1) -> Any:
        self._charge(len(self) - self._start(index))
        return super().pop(index)

    def remove(self, value: Any) -> None:
        self._charge(len(self))
        super().remove(value)

    def index(self, *args: Any) -> int:
        self._charge(len(self))
        return super().index(*args)

    def extend(self, other: Any) -> None:
        other = list(other)
        self._charge(len(other))
        super().extend(other)

    def __iadd__(self, other: Any) -> Any:
        self.extend(other)
        return self

    def copy(self) -> list[tuple[yaml.Node, yaml.Node]]:
        self._charge(len(self))
        return list(super().__iter__())

    def __add__(self, other: Any) -> Any:
        self._charge(len(self) + len(other))
        return [*super().__iter__(), *other]

    def __radd__(self, other: Any) -> Any:
        self._charge(len(self) + len(other))
        return [*other, *super().__iter__()]


def _merge_work(
    count: int, monkeypatch: pytest.MonkeyPatch, *, flatten: Callable[..., None] | None = None
) -> int:
    """Slots the loader touches in mapping-node pair lists to load ``_merge_doc(count)``."""
    from particles.core import bounded_yaml

    loader = bounded_yaml._BoundedSafeLoader
    compose = loader.compose_mapping_node
    meter = [0]

    def metered(self: Any, anchor: Any) -> yaml.MappingNode:
        node = compose(self, anchor)
        node.value = _MeteredPairs(node.value, meter)
        return node

    with monkeypatch.context() as patch:
        patch.setattr(loader, "compose_mapping_node", metered)
        if flatten is not None:
            patch.setattr(loader, "flatten_mapping", flatten)
        assert load(_merge_doc(count)) == yaml.safe_load(_merge_doc(count))
    return meter[0]


def test_merge_keys_cost_time_linear_in_their_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Four times the merge keys costs about four times the work, as the node budget
    assumes (O24).

    The work is counted, not timed: timing two such loads (60,000 and 240,000
    keys) read a ratio of 6.3 against this bound of 6 under the full pre-commit
    run's load, and passed on a retry.
    """
    small = _merge_work(1_000, monkeypatch)
    large = _merge_work(4_000, monkeypatch)
    assert large / small < 6.0, (small, large)


def test_the_merge_meter_sees_an_in_place_merge_go_quadratic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: PyYAML's own merge, which deletes each key in place, reads quadratic.

    If this fails, the meter has gone blind to the cost the linearity test
    above guards against, or PyYAML has made its own merge linear.
    """
    flatten = yaml.SafeLoader.flatten_mapping
    small = _merge_work(1_000, monkeypatch, flatten=flatten)
    large = _merge_work(4_000, monkeypatch, flatten=flatten)
    assert large / small > 12.0, (small, large)


# --- load_sequence: a top-level sequence built one item at a time (AA16) ----


@pytest.mark.parametrize(
    "text",
    [
        "",
        "- a\n- b: 1\n  c: [x, y]\n",
        "- &u {k: v}\n- *u\n- {<<: *u, j: 2}\n",
        "[1, 2, {a: b}]\n",
        "plain\n",
        "!!seq [1, 2]\n",
    ],
)
def test_load_sequence_builds_what_load_builds(text: str) -> None:
    assert load_sequence(text, max_items=10) == load(text)


def test_an_alias_to_an_earlier_item_resolves_to_the_same_object() -> None:
    items = load_sequence("- &u {k: [1, 2]}\n- {ref: *u}\n", max_items=10)
    assert items[1]["ref"] is items[0]


def test_a_sequence_past_its_item_limit_is_refused_while_it_is_read() -> None:
    with pytest.raises(YAMLItemLimitError):
        load_sequence("- 1\n- 2\n- 3\n", max_items=2)
    assert issubclass(YAMLItemLimitError, yaml.YAMLError)


def test_a_tagged_top_level_sequence_is_refused() -> None:
    with pytest.raises(yaml.YAMLError):
        load_sequence("!!omap [a: 1]\n", max_items=10)


def test_load_sequence_keeps_every_refusal_load_makes() -> None:
    with pytest.raises(YAMLExpansionError):
        load_sequence("- " + _fan_out(8, 10).replace("\n", "\n  "), max_items=10)
    with pytest.raises(YAMLExpansionError):
        load_sequence("- &a [*a]\n", max_items=10)
    with pytest.raises(YAMLExpansionError):
        load_sequence("- " + "[" * 200 + "]" * 200 + "\n", max_items=10)


# --- load_sequence holds peak memory near what it builds (AB16) -------------

_UNIT = {"@type": "Particle", "content": "c" * 40, "tags": ["a", "b"], "properties": {"n": 1}}


def _indent(block: str) -> str:
    return "".join("  " + line + "\n" for line in block.splitlines())


def _peak_per_byte(text: str) -> float:
    import tracemalloc

    tracemalloc.start()
    try:
        with contextlib.suppress(yaml.YAMLError):
            load_sequence(text, max_items=100_000)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return peak / len(text.encode())


def _units(n: int = 3000) -> list[dict[str, object]]:
    return [dict(_UNIT, id=f"p-{i}", tags=["a", "b"], properties={"n": i}) for i in range(n)]


def test_anchored_items_peak_no_higher_than_plain_ones() -> None:
    """Items that each carry an anchor peak near the same items without one.

    Each anchored item's node tree used to stay alive for a later alias, tens
    of bytes for each byte of input (security review 2026-10-30, AB16). Peak
    per byte does not depend on the item count, so a thousand items show it as
    well as three thousand, at a third of the cost under ``tracemalloc``.
    """
    units = _units(1000)
    plain = yaml.safe_dump(units, sort_keys=False, default_flow_style=False)
    anchored = "".join(
        f"- &u{i}\n" + _indent(yaml.safe_dump(u, sort_keys=False, default_flow_style=False))
        for i, u in enumerate(units)
    )
    assert _peak_per_byte(anchored) <= 2 * _peak_per_byte(plain)


def test_a_top_level_mapping_reads_as_not_a_sequence_without_being_composed() -> None:
    """A mapping at the top level reads as an empty mapping, never composed whole (AB16)."""
    body = yaml.safe_dump(_units(), sort_keys=False, default_flow_style=False)
    text = "root:\n" + _indent(body)
    assert load_sequence(text, max_items=10) == {}
    assert load_sequence("a: 1\n", max_items=10) == {}
    assert _peak_per_byte(text) < 4.0


@pytest.mark.parametrize(
    "text",
    [
        "- &u {k: [1, 2], m: {a: b}}\n- {<<: *u, j: 2}\n- {<<: [*u, {z: 1}], k: 3}\n",
        "- &a {x: 1}\n- &b {y: 2}\n- &l [*a, *b]\n- {<<: *l}\n- {<<: [*b, *a], x: 9}\n",
        "- {<<: &m {p: 1}, q: 2}\n- {r: *m}\n",
        "- &s [1, &t {u: 1}]\n- [*s, *t]\n- {<<: *t}\n",
    ],
)
def test_aliases_and_merges_of_released_items_build_what_load_builds(text: str) -> None:
    """An alias or merge of an earlier item's anchor builds what a whole-document load does."""
    assert load_sequence(text, max_items=10) == load(text)


def test_an_alias_to_a_released_item_keeps_its_depth_and_size_charges() -> None:
    """A released item's cached size and depth still charge a later alias in full."""
    nested = "[" * 60 + "]" * 60
    with pytest.raises(YAMLExpansionError):
        load_sequence(f"- &d {nested}\n- " + "[" * 50 + "*d" + "]" * 50 + "\n", max_items=10)
    fan = "- &a [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]\n"
    for i in range(1, 8):
        fan += f"- &{chr(97 + i)} [" + ", ".join([f"*{chr(96 + i)}"] * 10) + "]\n"
    with pytest.raises(YAMLExpansionError):
        load_sequence(fan, max_items=20)


def test_many_anchored_items_build_in_time_linear_in_their_count() -> None:
    """Forgetting each item's objects no longer walks every earlier anchor (AB16)."""
    import os
    import subprocess
    import sys

    # Measured in a fresh interpreter, on the fastest of three runs of each
    # size. Inside a long-lived test worker the same two loads read a ratio of
    # 10.4 on CI twice running, against 4.0 here, so the reading depended on
    # the worker's accumulated state rather than on the loader. A linear
    # loader reads about 4 and a quadratic one about 16.
    script = r"""
import time
from particles.core.bounded_yaml import load_sequence

def elapsed(n):
    text = "".join(f"- &u{i} {{k: {i}}}\n" for i in range(n))
    runs = []
    for _ in range(3):
        start = time.perf_counter()
        load_sequence(text, max_items=n)
        runs.append(time.perf_counter() - start)
    return min(runs)

print(elapsed(5_000), elapsed(20_000))
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("COV_CORE")}
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=120
    )
    assert result.returncode == 0, result.stderr
    small, large = (float(x) for x in result.stdout.split())
    assert large / small < 8.0, (small, large)


# --- merges of released items, and the bounds load_sequence shares


@pytest.mark.parametrize(
    "text",
    [
        "- &s !!set {a, b}\n- {<<: *s}\n",
        "- &s !!set {z, a: 1, m}\n- {<<: *s, q: 2}\n- {<<: [*s, {k: 0}]}\n",
        "- &l [!!set {x: 2}, {y: 3}]\n- {<<: *l}\n",
    ],
)
def test_a_merge_of_a_released_set_builds_what_load_builds(text: str) -> None:
    """A ``!!set`` merged by a later item folds in the mapping it was written as."""
    streamed, whole = load_sequence(text, max_items=10), load(text)
    assert streamed == whole == yaml.safe_load(text)
    # Key order follows the document, not the set's hash order.
    assert [list(v) for v in streamed if isinstance(v, dict)] == [
        list(v) for v in whole if isinstance(v, dict)
    ]


def test_a_bad_merge_of_a_released_sequence_stays_a_yaml_error() -> None:
    """A released list of scalars merged into a mapping is refused as ``load`` refuses it."""
    text = "- &l [1, 2]\n- {<<: *l}\n"
    with pytest.raises(yaml.constructor.ConstructorError):
        load(text)
    with pytest.raises(yaml.constructor.ConstructorError):
        load_sequence(text, max_items=10)


def test_load_sequence_honours_an_explicit_budget() -> None:
    with pytest.raises(YAMLExpansionError):
        load_sequence("- [a, b, c, d]\n", max_items=10, max_nodes=3)
    assert load_sequence("- [a, b, c, d]\n", max_items=10, max_nodes=6) == [["a", "b", "c", "d"]]


@pytest.mark.parametrize("empty", ["{}", "[]"])
def test_an_empty_merge_adds_nothing(empty: str) -> None:
    text = f"m:\n  <<: {empty}\n  a: 1\n"
    assert load(text) == yaml.safe_load(text) == {"m": {"a": 1}}
    assert load_sequence(f"- <<: {empty}\n  a: 1\n", max_items=10) == [{"a": 1}]


def test_a_scalar_anchor_does_not_stop_the_release_of_its_item() -> None:
    """An anchored scalar ahead of an anchored mapping in one item still lets the mapping go.

    Release skips the anchored nodes it cannot release; stopping at the first
    one would keep every later anchored node's tree alive (AB16): about four
    times the peak per byte of the same items whose mappings carry no anchor.
    """

    def items(mapping_anchor: bool) -> str:
        return "".join(
            f"- - &s{i} x\n"
            + _indent(
                (f"- &u{i}\n" if mapping_anchor else "-\n")
                + _indent(yaml.safe_dump(u, sort_keys=False))
            )
            for i, u in enumerate(_units(1000))
        )

    anchored, baseline = items(True), items(False)
    assert load_sequence(anchored, max_items=2000) == load_sequence(baseline, max_items=2000)
    assert _peak_per_byte(anchored) <= 2 * _peak_per_byte(baseline)
