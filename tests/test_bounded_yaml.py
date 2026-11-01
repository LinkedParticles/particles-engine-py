# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""The bounded YAML loader: alias expansion is held to a budget linear in the text."""

from __future__ import annotations

import contextlib

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


def _elapsed(text: str) -> float:
    import time

    start = time.perf_counter()
    load(text)
    return time.perf_counter() - start


def test_merge_keys_cost_time_linear_in_their_count() -> None:
    """Four times the merge keys costs at most six times the time, as the node budget
    assumes (O24)."""
    small = _elapsed(_merge_doc(60_000))
    large = _elapsed(_merge_doc(240_000))
    assert large / small < 6.0, (small, large)


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
    import time

    def elapsed(n: int) -> float:
        text = "".join(f"- &u{i} {{k: {i}}}\n" for i in range(n))
        start = time.perf_counter()
        load_sequence(text, max_items=n)
        return time.perf_counter() - start

    small = elapsed(5_000)
    large = elapsed(20_000)
    assert large / small < 8.0, (small, large)
