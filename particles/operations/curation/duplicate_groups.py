# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""One curation card per connected group of duplicate candidates.

The pure half: the shell gathers the co-evidential finder's pairs, the
store's ``CO_EVIDENTIAL`` components, the standing group rulings and each
claim's election key, and these functions decide the cards (D2).

* :func:`group_pairs` forms the cards at build time (§1): withheld pairs out,
  already-linked claims collapsed to one node (a *linked set*), connected
  components over the rest, one card per component of two or more nodes.
* :func:`regroup` re-applies the same rules to one served card (§6), so a
  link, a ruling or a retirement after the build shows at once.
* :class:`GroupRulings` is the partition reading of the standing ``split`` /
  ``dismiss`` rulings (§5): a pair is withheld when one ruling names both
  claims and puts them in different classes.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from particles.core.schema import JudgeVerdictKind

from .cards import CardKind, CurationCard, DuplicateEdge, DuplicateVerdict, gestures_for

#: An election key: lower sorts first.
ElectionKey = tuple[bool, datetime | None, datetime | None, str]

#: The most members ``--members all`` accepts, and the most the queue slice
#: briefs whole (§6).
ALL_MEMBERS_LIMIT = 10

#: How many members besides the survivor a larger group's slice briefs.
LARGE_GROUP_SAMPLE = 5


def election_key(
    pid: str,
    *,
    subject_linked: bool,
    first_read: datetime | None,
    asserted_at: datetime | None,
) -> ElectionKey:
    """§3's subject-linked term, then first reading.

    A missing reading or assertion time sorts last within its term (``None``
    is mapped by :func:`_sortable`).
    """
    return (not subject_linked, first_read, asserted_at, pid)


def _sortable(key: ElectionKey) -> tuple[bool, bool, datetime | str, bool, datetime | str, str]:
    """``key`` with each optional time made comparable, ``None`` last."""
    unlinked, first, asserted, pid = key
    return (
        unlinked,
        first is None,
        first if first is not None else "",
        asserted is None,
        asserted if asserted is not None else "",
        pid,
    )


@dataclass(frozen=True)
class GroupRulings:
    """The standing ``split`` / ``dismiss`` rulings on duplicate groups.

    Each ruling is a partition of the claims it named. Plain values, read by
    the shell from the event log.
    """

    rulings: tuple[tuple[frozenset[str], ...], ...] = ()
    _index: dict[str, list[tuple[int, int]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for r, classes in enumerate(self.rulings):
            for c, members in enumerate(classes):
                for pid in members:
                    self._index.setdefault(pid, []).append((r, c))

    def withholds(self, a: str, b: str) -> bool:
        """Whether one ruling names both claims and puts them in different classes."""
        where_b = dict(self._index.get(b, ()))
        return any(r in where_b and where_b[r] != c for r, c in self._index.get(a, ()))

    def restricted(self, members: Iterable[str]) -> list[list[list[str]]]:
        """Each ruling cut to ``members``, kept when two or more classes remain.

        What a card shows as ``prior_rulings``: a cut a later claim has bridged
        into one group again.
        """
        wanted = set(members)
        out: list[list[list[str]]] = []
        for classes in self.rulings:
            kept = [sorted(c & wanted) for c in classes if c & wanted]
            if len(kept) >= 2:
                out.append(sorted(kept))
        return out


def pair_withheld(suppressed: Collection[str], rulings: GroupRulings) -> Callable[[str, str], bool]:
    """Whether a pair is withheld: by a group ruling, or by its old pair key (§5).

    ``suppressed`` is the card keys a standing ruling hides, so a
    ``duplicate_pair:<a>|<b>`` dismissal, or a snooze until it lapses, keeps
    holding its pair.
    """

    def withheld(a: str, b: str) -> bool:
        old = f"{CardKind.DUPLICATE_PAIR.value}:" + "|".join(sorted((a, b)))
        return old in suppressed or rulings.withholds(a, b)

    return withheld


def node_of(pid: str, linked: Mapping[str, frozenset[str]]) -> str:
    """The id that stands for ``pid``'s linked set: its smallest member."""
    component = linked.get(pid)
    return min(component) if component else pid


def group_pairs(
    edges: Sequence[DuplicateEdge],
    *,
    linked: Mapping[str, frozenset[str]],
    withheld: Callable[[str, str], bool],
    rank: Mapping[str, ElectionKey],
    rulings: GroupRulings,
    subject_names: Mapping[str, str] | None = None,
) -> list[CurationCard]:
    """One ``duplicate_group`` card per connected group of unwithheld pairs (§1).

    ``linked`` maps a claim to its ``CO_EVIDENTIAL`` component, so claims an
    earlier ``merge`` joined are one node. ``rank`` holds each claim's election
    key; a claim without one ranks after every claim with one, by id.
    """
    kept = [
        e for e in edges if not withheld(e.a, e.b) and node_of(e.a, linked) != node_of(e.b, linked)
    ]
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in kept:
        ra, rb = find(node_of(e.a, linked)), find(node_of(e.b, linked))
        if ra != rb:
            parent[ra] = rb
    by_root: dict[str, list[DuplicateEdge]] = {}
    for e in kept:
        by_root.setdefault(find(node_of(e.a, linked)), []).append(e)
    # A linked set is shown whole: every claim of it any pair named, including
    # one whose only pair now falls inside the set.
    claims_of: dict[str, set[str]] = {}
    for e in edges:
        for pid in (e.a, e.b):
            claims_of.setdefault(node_of(pid, linked), set()).add(pid)
    cards = [
        _card(
            group,
            claims={
                pid for e in group for end in (e.a, e.b) for pid in claims_of[node_of(end, linked)]
            },
            linked=linked,
            rank=rank,
            rulings=rulings,
            subject_names=subject_names or {},
        )
        for group in by_root.values()
    ]
    return sorted(cards, key=lambda c: c.key)


def sort_key(pid: str, rank: Mapping[str, ElectionKey]) -> tuple[object, ...]:
    """``pid``'s place in the election; a claim without a key sorts last, by id."""
    key = rank.get(pid)
    return (0, *_sortable(key)) if key is not None else (1, pid)


def _card(
    edges: list[DuplicateEdge],
    *,
    claims: set[str],
    linked: Mapping[str, frozenset[str]],
    rank: Mapping[str, ElectionKey],
    rulings: GroupRulings,
    subject_names: Mapping[str, str],
) -> CurationCard:
    """The card for one component's edges and the claims its nodes hold."""
    members = sorted(claims, key=lambda p: sort_key(p, rank))
    survivor = members[0]
    weakest: dict[str, float] = {}
    for e in edges:
        for pid in (e.a, e.b):
            weakest[pid] = min(weakest.get(pid, 1.0), e.similarity)
    nodes: dict[str, list[str]] = {}
    for pid in members:  # election order within each linked set
        nodes.setdefault(node_of(pid, linked), []).append(pid)
    survivor_node = node_of(survivor, linked)

    def node_order(node: str) -> tuple[object, ...]:
        held = nodes[node]
        return (min(weakest.get(p, 1.0) for p in held), sort_key(held[0], rank))

    ordered = [survivor_node] + sorted((n for n in nodes if n != survivor_node), key=node_order)
    particle_ids = [pid for node in ordered for pid in nodes[node]]
    subject_ids = list(dict.fromkeys(e.subject_id for e in edges if e.subject_id))
    sims = [e.similarity for e in edges]
    names = [subject_names.get(s, s) for s in subject_ids[:2]]
    where = " in " + ", ".join(f"'{n}'" for n in names) if names else ""
    if len(subject_ids) > 2:
        where += f" and {len(subject_ids) - 2} more"
    span = f"{min(sims):.2f}" if min(sims) == max(sims) else f"{min(sims):.2f}–{max(sims):.2f}"
    pairs = sorted(edges, key=lambda e: (e.a, e.b))
    return CurationCard(
        kind=CardKind.DUPLICATE_GROUP,
        particle_ids=particle_ids,
        subject_ids=subject_ids,
        diagnostic=(
            f"{len(particle_ids)} possible duplicates{where} "
            f"({len(pairs)} similar pair{'s' if len(pairs) != 1 else ''}, similarity {span})"
        ),
        suggested_gestures=gestures_for(CardKind.DUPLICATE_GROUP),
        verdict=_group_verdict(pairs),
        pairs=pairs,
        linked_sets=[sorted(c) for n, c in nodes.items() if len(c) > 1] or None,
        prior_rulings=rulings.restricted(particle_ids) or None,
        survivor_id=survivor,
    )


def _group_verdict(pairs: list[DuplicateEdge]) -> DuplicateVerdict | None:
    """A one-pair group carries its pair's verdict, as a pair card did."""
    return pairs[0].verdict if len(pairs) == 1 else None


def all_distinct(card: CurationCard) -> bool:
    """Every carded pair the judge cleared as DISTINCT."""
    pairs = card.pairs or []
    return bool(pairs) and all(
        e.verdict is not None and e.verdict.verdict is JudgeVerdictKind.DISTINCT for e in pairs
    )


def regroup(
    card: CurationCard,
    *,
    active: set[str] | None,
    linked: Mapping[str, frozenset[str]],
    rulings: GroupRulings,
    rank: Mapping[str, ElectionKey] | None = None,
    withheld: Callable[[str, str], bool] | None = None,
) -> CurationCard | None:
    """A served group card with post-build links, rulings and retirements applied (§6).

    Pairs with a member outside ``active`` (when given) or a pair a ruling
    withholds drop out, and members since linked become one node. The card
    keeps the component holding its survivor, so a split's named side waits
    for the next build. ``None`` when fewer than two nodes remain. When the
    survivor itself has gone, ``rank`` (the election keys of the remaining
    members) decides which component stands; without it the card's own order
    does. Leverage and the diagnostic are the build's. The input is not
    mutated. ``withheld`` defaults to the rulings' own reading; the shell
    passes one that also reads the old ``duplicate_pair`` keys (§5).
    """
    order = {pid: i for i, pid in enumerate(card.particle_ids)}
    edges = [e for e in card.pairs or [] if active is None or (e.a in active and e.b in active)]
    if card.survivor_id is not None and (active is None or card.survivor_id in active):
        keys: Mapping[str, ElectionKey] = {
            pid: (pid != card.survivor_id, None, None, f"{order.get(pid, len(order)):09d}")
            for pid in order
        }
    elif rank is not None:
        keys = rank
    else:
        keys = {pid: (False, None, None, f"{i:09d}") for pid, i in order.items()}
    cards = group_pairs(
        edges, linked=linked, withheld=withheld or rulings.withholds, rank=keys, rulings=rulings
    )
    if not cards:
        return None
    survivor = card.survivor_id
    chosen = next((c for c in cards if survivor in c.particle_ids), None) if survivor else None
    if chosen is None:
        chosen = min(cards, key=lambda c: sort_key(c.particle_ids[0], keys))
    unchanged = (
        chosen.particle_ids == card.particle_ids
        and chosen.linked_sets == card.linked_sets
        and chosen.prior_rulings == card.prior_rulings
    )
    if unchanged:
        return card
    return card.model_copy(
        update={
            "particle_ids": chosen.particle_ids,
            "pairs": chosen.pairs,
            "linked_sets": chosen.linked_sets,
            "prior_rulings": chosen.prior_rulings,
            "survivor_id": chosen.survivor_id,
            "subject_ids": chosen.subject_ids,
            "verdict": chosen.verdict,
            "particles": [],
        }
    )


def brief_ids(card: CurationCard, *, full: bool) -> list[str]:
    """Every member of a group of ten or fewer; else the survivor and five more (§6)."""
    if full or len(card.particle_ids) <= ALL_MEMBERS_LIMIT:
        return card.particle_ids
    return card.particle_ids[: 1 + LARGE_GROUP_SAMPLE]


def nodes_of(card: CurationCard) -> list[list[str]]:
    """The card's members as nodes: each linked set once, every other claim alone."""
    sets = {pid: tuple(s) for s in card.linked_sets or [] for pid in s}
    seen: set[str] = set()
    out: list[list[str]] = []
    for pid in card.particle_ids:
        if pid in seen:
            continue
        group = list(sets.get(pid, (pid,)))
        seen.update(group)
        out.append(group)
    return out


def resolve_members(card: CurationCard, named: Sequence[str]) -> list[str]:
    """The claims ``named`` picks out of ``card``, each linked set whole (§3).

    Each value is a full id or a unique id prefix of a current member; ``all``
    names every member of a group of at most :data:`ALL_MEMBERS_LIMIT`.
    Raises ``ValueError`` naming the first value that does not resolve.
    """
    members = list(card.particle_ids)
    values = [v.strip() for v in named if v.strip()]
    if not values:
        raise ValueError("Name the members with --members (ids from the card, or all).")
    if [v.lower() for v in values] == ["all"]:
        if len(members) > ALL_MEMBERS_LIMIT:
            raise ValueError(
                f"--members all applies to a group of {ALL_MEMBERS_LIMIT} or fewer; this one "
                f"has {len(members)}. Name the members from `particles curate --card "
                f"{card.key}`."
            )
        return members
    picked: set[str] = set()
    for value in values:
        matches = [pid for pid in members if pid == value or pid.startswith(value)]
        if len(matches) != 1:
            why = "matches no member" if not matches else "matches more than one member"
            raise ValueError(f"{value!r} {why} of {card.key}.")
        picked.add(matches[0])
    for node in nodes_of(card):
        if picked & set(node):
            picked.update(node)
    return [pid for pid in members if pid in picked]


def split_card(
    card: CurationCard, keep_edge: Callable[[DuplicateEdge], bool]
) -> list[CurationCard]:
    """``card`` regrouped over the pairs ``keep_edge`` keeps; each component a card.

    The audit's reading: a pair it sets aside can break a group in
    two. The card's own order stands in for the election, so the survivor of
    each part is its best-placed member. Leverage is the caller's to rescore.
    """
    order = {pid: (False, None, None, f"{i:09d}") for i, pid in enumerate(card.particle_ids)}
    linked = {pid: frozenset(s) for s in card.linked_sets or [] for pid in s}
    rulings = GroupRulings()
    return group_pairs(
        [e for e in card.pairs or [] if keep_edge(e)],
        linked=linked,
        withheld=lambda _a, _b: False,
        rank=order,
        rulings=rulings,
    )
