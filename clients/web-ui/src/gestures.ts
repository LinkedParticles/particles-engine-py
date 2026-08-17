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
    contested: ["affirm", "comment"],
    contradiction: ["comment"],
    retraction_cascade: ["supersede", "retract"],
    broken_provenance: ["supersede", "retract"],
    no_subject: ["assign-subject", "supersede", "retract"],
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

/** Human label + the CSS class for a gesture button. */
export function gestureLabel(gesture: string): string {
  const labels: Record<string, string> = {
    affirm: "Still true",
    snooze: "Snooze",
    dismiss: "Dismiss",
    comment: "Comment",
    merge: "Merge",
    deposit: "Deposit",
    supersede: "Edit",
    retract: "Retract",
    reindex: "Reindex",
    "assign-subject": "Assign subject",
    vouch: "Vouch",
  };
  return labels[gesture] ?? gesture;
}

/** Gestures whose CSS should mark them destructive. */
export function isDangerGesture(gesture: string): boolean {
  return gesture === "retract";
}
