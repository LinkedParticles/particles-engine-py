/*
 * Gesture → endpoint mapping.
 *
 * The single source of truth for which gestures a card offers, whether each is
 * v1-backed by a shipped endpoint or deferred (shown disabled with a note, NO
 * invented endpoint), and how the v1 ones map to UI intent. The PWA adds NO
 * operation logic — each v1 gesture is one authenticated call to an
 * already-shipped endpoint (dispatched in feed.ts). It is the phone-shaped twin
 * of `particles curate apply <gesture> <card-key>`.
 *
 * The curation write surface is closed, so the previously-deferred
 * gestures now have endpoints:
 *   - affirm → POST /curation/affirm (BELIEF_AFFIRMED); belief-snooze /
 *     belief-dismiss → POST /curation/snooze (CURATION_CARD_SNOOZED).
 *   - per-particle retract of an extracted/operator belief → operator
 *     POST /particles/{id}/retract; edit-as-supersede on an extracted belief →
 *     operator POST /particles/{id}/supersede.
 *   - assign-subject (NO_SUBJECT card) → POST /particles/{id}/subjects.
 *   - relink (GATED_SUBJECTS batch card) → POST /subjects/relink-gated.
 * Still deferred:
 *   - vouch — proposed, not active — not offered at all.
 */

/** How the UI should treat a gesture button when it is rendered on a card. */
export type GestureAvailability =
  | { kind: "v1" }
  | { kind: "deferred"; note: string; pdr: string };

/**
 * Resolve the availability of a gesture name on a card of a given CardKind. The
 * gesture names come straight from the engine's `card.suggested_gestures`
 * — the PWA invents none. `dismiss` is the one gesture whose
 * availability depends on the card kind (v1 only for uncited_url via the URL
 * dismiss endpoint; belief-snooze on other kinds is the deferred path).
 */
export function gestureAvailability(
  gesture: string,
  _kind: string,
): GestureAvailability {
  switch (gesture) {
    case "comment":
    case "resolve": // POST /review/{inconsistency_id}
    case "merge":
    case "deposit":
    case "supersede":
    case "reindex":
      return { kind: "v1" };
    case "retract":
      // v1: operator per-particle retract (POST /particles/{id}/retract
      //) for an extracted belief, with whole-source retract as the
      // fallback when no belief id is on the card.
      return { kind: "v1" };
    case "assign-subject":
      return { kind: "v1" }; // POST /particles/{id}/subjects
    case "relink":
      return { kind: "v1" }; // POST /subjects/relink-gated
    case "dismiss":
      // v1 both ways now: uncited_url via POST /corpus/links/dismiss, belief
      // cards via POST /curation/snooze (permanent dismiss).
      return { kind: "v1" };
    case "affirm":
      return { kind: "v1" }; // POST /curation/affirm
    case "snooze":
      return { kind: "v1" }; // POST /curation/snooze
    case "vouch":
      return {
        kind: "deferred",
        note: "Vouch awaits the endorsement primitive (not active)",
        pdr: "ADR-0140",
      };
    default:
      // An unknown gesture from a future engine: render it disabled rather than
      // guess an endpoint.
      return {
        kind: "deferred",
        note: "Not supported by this client version",
        pdr: "",
      };
  }
}

/** The dominant safe gesture to surface as the primary swipe, per kind. */
export function primaryGesture(kind: string, offered: string[]): string | null {
  // Prefer the cheapest card-resolving gesture that has a v1 backing; fall back
  // to the first v1-backed gesture offered.
  const preference: Record<string, string[]> = {
    stale: ["affirm", "supersede", "retract"],
    confidence_decay: ["affirm", "supersede"],
    // an open conflict is its own card, cleared by resolving it.
    inconsistency: ["resolve"],
    // A contested card now fires only on an observer signal; an older engine
    // may still send `comment` for its conflict.
    contested: ["comment", "affirm"],
    contradiction: ["supersede", "retract"],
    retraction_cascade: ["supersede", "retract"],
    broken_provenance: ["supersede", "retract"],
    no_subject: ["assign-subject", "supersede", "retract"],
    gated_subjects: ["relink"],
    duplicate_pair: ["merge"],
    uncited_url: ["deposit", "dismiss"],
    failed_snapshots: ["reindex"],
  };
  const prefs = preference[kind] ?? [];
  for (const g of prefs) {
    if (offered.includes(g) && gestureAvailability(g, kind).kind === "v1") {
      return g;
    }
  }
  for (const g of offered) {
    if (gestureAvailability(g, kind).kind === "v1") return g;
  }
  return null;
}

/**
 * Human label for a gesture button. `comment` is the engine's name for the
 * gesture that resolves an INCONSISTENCY (`POST /review/{id}`), so it is
 * labelled for what it does, not what it was once imagined to be.
 */
export function gestureLabel(gesture: string, kind = ""): string {
  if (kind === "contested" && gesture === "affirm") return "Stands anyway";
  const labels: Record<string, string> = {
    affirm: "Still true",
    snooze: "Snooze",
    dismiss: "Dismiss",
    comment: "Resolve…",
    resolve: "Resolve…",
    merge: "Merge",
    deposit: "Deposit",
    supersede: "Edit",
    retract: "Retract",
    reindex: "Reindex",
    "assign-subject": "Assign subject",
    relink: "Link subjects",
    vouch: "Vouch",
  };
  return labels[gesture] ?? gesture;
}

/**
 * One line saying what a gesture does, in the curator's terms. The web twin
 * of the CLI's per-gesture help (`describe_gesture` in the engine), phrased
 * for buttons rather than flags. Empty for a gesture with nothing to add.
 */
export function gestureHint(gesture: string, kind = ""): string {
  const byKind: Record<string, string> = {
    "inconsistency:resolve":
      "Settle the conflict: keep one side, keep both, or discard both.",
    "inconsistency:snooze": "Decide later. Hides the card for a while; the conflict stays open.",
    "contested:comment":
      "Settle the conflict: keep one side, keep both, discard both, or defer.",
    "contested:affirm":
      "This belief stands despite the disagreement. Hides the card for good; a conflict it is in keeps its own card.",
    "duplicate_pair:dismiss": "They are different claims. Hides the card for good.",
    "uncited_url:dismiss": "Not worth depositing. Stops suggesting this URL.",
  };
  const generic: Record<string, string> = {
    affirm: "The belief is still correct. Records that and hides the card for good.",
    snooze: "Decide later. Hides the card for a while.",
    dismiss: "Not a real problem. Hides the card for good.",
    comment: "Resolve the underlying conflict.",
    resolve: "Resolve the conflict.",
    merge: "Same claim stated twice. Links the two; both stay active.",
    deposit: "Fetch the URL into the corpus for extraction.",
    supersede: "Replace the belief with a corrected one.",
    retract: "The belief is wrong. Retracts it.",
    reindex: "Re-extract the failed snapshots.",
    "assign-subject": "Attach the belief to a subject.",
    relink:
      "Link every belief on this card to the files, records or commands it names, scoped by project.",
  };
  return byKind[`${kind}:${gesture}`] ?? generic[gesture] ?? "";
}

/** Gestures whose CSS should mark them destructive. */
export function isDangerGesture(gesture: string): boolean {
  return gesture === "retract";
}
