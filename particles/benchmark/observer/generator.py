# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Two projects, one store: the seeded world (gate B).

The dogfood store holds one real project, so the rate at which a belief
vanishes from its own project under the two named mechanisms
cannot be read off it. This world manufactures the situation: two repositories whose
memory files talk about the same generic subjects — the repository, the test
suite, the deploy pipeline — and evolve independently, so that

* two projects state **different values** for one slot (rung 2.5 pairs them:
  the later deposit wins, the earlier project's own line is retired), and
* two projects state **the same rule** byte-for-byte (folds it into one
  particle; when one project drops the line, the cascade retires it
  for both).

A ``seed`` is the fixture: :func:`generate_world` is pure, and
:attr:`ObserverWorld.fingerprint` is pinned by the tests — bump
``GENERATOR_VERSION`` when a seed's output is meant to change.
"""

from __future__ import annotations

import hashlib
import json
import random
import re

from particles.benchmark.observer.schema import (
    GENERATOR_VERSION,
    PROJECT_A,
    PROJECT_B,
    PROJECTS,
    EventKind,
    LineKind,
    MemoryLine,
    ObserverWorld,
    WorldEvent,
)

#: ``slot → (subject, template, value pool)``. Subjects are generic on purpose:
#: both projects have a repository, a test suite, a deploy pipeline.
SLOTS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "branch": (
        "the repository",
        "The default branch of the repository is {v}.",
        ("main", "master", "trunk", "develop"),
    ),
    "runner": (
        "the test suite",
        "The test suite is run with {v}.",
        ("pytest", "unittest", "nose2", "ward"),
    ),
    "target": (
        "the deploy pipeline",
        "The deploy pipeline targets {v}.",
        ("Fly", "Heroku", "Render", "Railway"),
    ),
    "cadence": (
        "the release process",
        "Releases are cut every {v}.",
        ("Friday", "Monday", "sprint", "fortnight"),
    ),
}

#: ``rule → subject``. Slot-less, so two projects stating one produce one particle.
RULES: dict[str, str] = {
    "Every commit to the repository needs a sign-off.": "the repository",
    "The test suite must pass before a merge.": "the test suite",
    "The deploy pipeline is never run during a release freeze.": "the deploy pipeline",
    "The release process requires two approvals.": "the release process",
}

#: Per-project subjects nothing else mentions — the control lines.
OWN_SUBJECT = {PROJECT_A: "project alpha", PROJECT_B: "project beta"}
OWN_TEMPLATE = "The build of {subject} takes {v} minutes."
OWN_VALUES = ("seven", "twelve", "nineteen", "thirty")

GLOBAL_LINES = [
    MemoryLine(text="The user prefers tabs over spaces.", subject="the user", kind=LineKind.RULE),
    MemoryLine(
        text="The user reads pull requests before their description.",
        subject="the user",
        kind=LineKind.RULE,
    ),
]


#: ``global line → the project line that contests it``. Deposited once per world
#: after the last day (alpha contests the first, beta the second), so the daily
#: measurement is untouched: a project disagreeing with the operator's own claim
#: must reach review, never silently retire it.
GLOBAL_CONTESTS: dict[str, MemoryLine] = {
    GLOBAL_LINES[0].text: MemoryLine(
        text="The user prefers spaces over tabs.", subject="the user", kind=LineKind.RULE
    ),
    GLOBAL_LINES[1].text: MemoryLine(
        text="The user reads the description of a pull request before its code.",
        subject="the user",
        kind=LineKind.RULE,
    ),
}


def value_pattern(value: str) -> re.Pattern[str]:
    """Case-insensitive whole-word matcher for one value."""
    return re.compile(r"(?<![\w-])" + re.escape(value) + r"(?![\w-])", re.IGNORECASE)


def check_value_invariants() -> None:
    """Refuse a slot table the scripted probe could misread.

    Values must not appear inside any template, rule or other pool: the probe
    decides "contradiction" by finding one slot's values on both sides.
    """
    fixed = [t for _, t, _ in SLOTS.values()] + list(RULES) + [OWN_TEMPLATE]
    fixed += [line.text for line in GLOBAL_LINES]
    fixed += [line.text for line in GLOBAL_CONTESTS.values()]
    seen: dict[str, str] = {}
    for slot, (_, _, pool) in SLOTS.items():
        for v in pool:
            if v.lower() in seen:
                raise ValueError(f"value {v!r} in slots {seen[v.lower()]!r} and {slot!r}")
            seen[v.lower()] = slot
            for text in fixed:
                if value_pattern(v).search(text.replace("{v}", "")):
                    raise ValueError(f"value {v!r} appears in fixed text {text!r}")


def fact_line(slot: str, value: str) -> MemoryLine:
    subject, template, _ = SLOTS[slot]
    return MemoryLine(
        text=template.format(v=value), subject=subject, kind=LineKind.FACT, slot=slot, value=value
    )


def rule_line(rule: str) -> MemoryLine:
    return MemoryLine(text=rule, subject=RULES[rule], kind=LineKind.RULE)


def own_line(project: str, value: str) -> MemoryLine:
    subject = OWN_SUBJECT[project]
    return MemoryLine(
        text=OWN_TEMPLATE.format(subject=subject, v=value),
        subject=subject,
        kind=LineKind.OWN,
        slot="own",
        value=value,
    )


def generate_world(seed: int, days: int = 12) -> ObserverWorld:
    """Build one world. Pure: the same ``(seed, days)`` always yields the same events.

    Day 1 gives each project a value for every slot (chosen independently, so
    some collide and some conflict), each rule with probability ¾, and one own
    line. Each later day, each project has an even chance of one event: a slot
    changes value (½), a rule is dropped or added (¼ each side), or the own line
    changes (¼).
    """
    check_value_invariants()
    rng = random.Random(seed * 7919 + days)
    events: list[WorldEvent] = []
    for project in PROJECTS:
        for slot, (_, _, pool) in SLOTS.items():
            events.append(
                WorldEvent(
                    day=1, project=project, kind=EventKind.SET, slot=slot, value=rng.choice(pool)
                )
            )
        for rule in RULES:
            if rng.random() < 0.75:
                events.append(
                    WorldEvent(day=1, project=project, kind=EventKind.ADD_RULE, rule=rule)
                )
        events.append(
            WorldEvent(day=1, project=project, kind=EventKind.SET_OWN, value=rng.choice(OWN_VALUES))
        )

    state = {p: _replay(events, p, 1) for p in PROJECTS}
    for day in range(2, days + 1):
        for project in PROJECTS:
            if rng.random() >= 0.5:
                continue
            roll = rng.random()
            lines = state[project]
            if roll < 0.5:
                slot = rng.choice(list(SLOTS))
                current = next(ln.value for ln in lines if ln.slot == slot)
                choices = [v for v in SLOTS[slot][2] if v != current]
                events.append(
                    WorldEvent(
                        day=day,
                        project=project,
                        kind=EventKind.SET,
                        slot=slot,
                        value=rng.choice(choices),
                    )
                )
            elif roll < 0.75:
                held = [ln.text for ln in lines if ln.kind is LineKind.RULE]
                missing = [r for r in RULES if r not in held]
                if held and (not missing or rng.random() < 0.5):
                    events.append(
                        WorldEvent(
                            day=day,
                            project=project,
                            kind=EventKind.DROP_RULE,
                            rule=rng.choice(held),
                        )
                    )
                elif missing:
                    events.append(
                        WorldEvent(
                            day=day,
                            project=project,
                            kind=EventKind.ADD_RULE,
                            rule=rng.choice(missing),
                        )
                    )
            else:
                current = next(ln.value for ln in lines if ln.kind is LineKind.OWN)
                events.append(
                    WorldEvent(
                        day=day,
                        project=project,
                        kind=EventKind.SET_OWN,
                        value=rng.choice([v for v in OWN_VALUES if v != current]),
                    )
                )
            state[project] = _replay(events, project, day)

    world = ObserverWorld(seed=seed, days=days, events=events, global_lines=list(GLOBAL_LINES))
    world.fingerprint = _fingerprint(world)
    return world


def _replay(events: list[WorldEvent], project: str, through_day: int) -> list[MemoryLine]:
    slots: dict[str, str] = {}
    rules: list[str] = []
    own: str | None = None
    for ev in events:
        if ev.project != project or ev.day > through_day:
            continue
        match ev.kind:
            case EventKind.SET:
                assert ev.slot is not None and ev.value is not None
                slots[ev.slot] = ev.value
            case EventKind.ADD_RULE:
                assert ev.rule is not None
                if ev.rule not in rules:
                    rules.append(ev.rule)
            case EventKind.DROP_RULE:
                assert ev.rule is not None
                rules = [r for r in rules if r != ev.rule]
            case EventKind.SET_OWN:
                own = ev.value
    lines = [fact_line(slot, slots[slot]) for slot in SLOTS if slot in slots]
    lines += [rule_line(r) for r in RULES if r in rules]
    if own is not None:
        lines.append(own_line(project, own))
    return lines


def lines_on_day(world: ObserverWorld, project: str, day: int) -> list[MemoryLine]:
    """A project's memory lines as they stand at the end of ``day``."""
    return _replay(world.events, project, day)


def render_memory_file(lines: list[MemoryLine]) -> str:
    """The ``MEMORY.md`` text a harness would find: one bullet per line."""
    return "# Memory\n\n" + "".join(f"- {line.text}\n" for line in lines)


def days_with_changes(world: ObserverWorld, project: str) -> set[int]:
    return {ev.day for ev in world.events if ev.project == project}


def _fingerprint(world: ObserverWorld) -> str:
    payload = json.dumps(
        [e.model_dump(mode="json") for e in world.events] + [GENERATOR_VERSION],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
