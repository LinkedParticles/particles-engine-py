/*
 * The swipeable, leverage-ranked card feed + gesture dispatch
 * (§5). Fetches GET /curation and renders the CurationQueueResponse as a
 * single-column swipeable card stack: the top card is interactive, the next
 * peeks behind it. Each card surfaces exactly the engine-computed
 * `suggested_gestures` — the PWA invents none — with the dominant
 * safe gesture as a primary swipe and the full set as tap targets.
 *
 * Read-only-degrade (§5): a 403 on any write means the engine has not opted into
 * belief writes (`mcp.write.enabled_stores` default-deny); the feed
 * re-renders read-only with the write gestures hidden, never controls that fail.
 */
import {
  ApiError,
  CurationCard,
  DuplicateVerdict,
  ParticleBrief,
  ParticlesApiClient,
  QueueOptions,
} from "./api";
import {
  gestureAvailability,
  gestureLabel,
  isDangerGesture,
  primaryGesture,
} from "./gestures";
import { routeHash } from "./router";
import { confirmSheet, openSheet } from "./sheet";

export interface FeedDeps {
  client: ParticlesApiClient;
  reviewerId: string;
  onNeedsSettings: () => void;
}

const KINDS = [
  "",
  "stale",
  "confidence_decay",
  "contested",
  "contradiction",
  "retraction_cascade",
  "broken_provenance",
  "no_subject",
  "duplicate_pair",
  "uncited_url",
  "failed_snapshots",
];

export class CurationFeed {
  private deps: FeedDeps;
  private root: HTMLElement;
  private cards: CurationCard[] = [];
  private options: QueueOptions = {};
  private readOnly = false;
  private semanticSkipped = false;
  private loaded = false;
  private loading = false;

  constructor(root: HTMLElement, deps: FeedDeps) {
    this.root = root;
    this.deps = deps;
  }

  updateDeps(deps: FeedDeps): void {
    this.deps = deps;
  }

  /**
   * Re-target the feed at a new view container (the shell hands each route
   * render a fresh element). The feed object outlives route switches so a
   * slow queue build (a large store computes it fresh per request) is not
   * thrown away by navigating: while away it completes into the detached old
   * container, and re-attaching shows the finished queue from memory.
   */
  attach(root: HTMLElement): void {
    this.root = root;
    if (this.loading) {
      this.renderLoading();
    } else if (this.loaded) {
      this.render();
    } else {
      void this.refresh();
    }
  }

  /** Fetch the queue and render. Fail-closed: no token ⇒ settings screen. */
  async refresh(): Promise<void> {
    if (!this.deps.client.isConfigured()) {
      this.deps.onNeedsSettings();
      return;
    }
    this.loading = true;
    this.renderLoading();
    try {
      const resp = await this.deps.client.curation(this.options);
      this.cards = resp.cards ?? [];
      // `resp.count` is deliberately not kept: it is len(cards) at fetch time,
      // so it only ever restated the number the header already shows — and as
      // gestures resolve cards it becomes a second, wrong count.
      this.semanticSkipped = resp.semantic_skipped ?? false;
      this.loaded = true;
      this.render();
    } catch (e) {
      this.renderError(e);
    } finally {
      this.loading = false;
    }
  }

  // --- Rendering ----------------------------------------------------------

  private renderLoading(): void {
    this.root.innerHTML = "";
    const el = document.createElement("div");
    el.className = "empty";
    el.textContent = "Building today's queue…";
    const note = document.createElement("div");
    note.className = "hint";
    // Honest about the cost: the queue is computed fresh over the whole
    // store per request (every finder runs — no cached result yet), so a
    // large store legitimately takes a while. Precomputing/caching it is an
    // engine-side decision.
    note.textContent =
      "The queue is computed fresh over the whole store, so a large store " +
      "can take a minute or two. You can switch tabs — the build keeps " +
      "running and the queue will be here when you come back.";
    el.appendChild(note);
    this.root.appendChild(el);
  }

  private render(): void {
    this.root.innerHTML = "";
    this.root.appendChild(this.renderHeader());
    this.root.appendChild(this.renderControls());

    if (this.readOnly) {
      this.root.appendChild(
        banner(
          "readonly",
          "This engine is read-only (belief writes disabled). Reviewing without write gestures.",
        ),
      );
    }

    if (this.semanticSkipped) {
      // the engine's LLM circuit breaker is open (account-level
      // failure — bad key / no permission / out of credits), so the LLM-assisted
      // finders were skipped. Say so rather than implying a clean queue.
      this.root.appendChild(
        banner(
          "info",
          "Semantic finders unavailable (LLM error — check the engine's API key / credit balance). Showing structural cards only.",
        ),
      );
    }

    if (this.cards.length === 0) {
      const done = document.createElement("div");
      done.className = "empty";
      done.innerHTML =
        '<div class="big">✓</div><div>Queue clear. Nothing to curate right now.</div>';
      this.root.appendChild(done);
      return;
    }

    const stack = document.createElement("div");
    stack.className = "stack";
    // Render the top card (and let it be removed as gestures resolve).
    stack.appendChild(this.renderCard(this.cards[0]));
    this.root.appendChild(stack);
  }

  /**
   * The queue's own line: how much is in front of you, and nothing else.
   *
   * No title — the nav's active tab already says Curate, and the internal name
   * for the model named nobody's concept but the author's. The count says "in
   * queue", not "today": the cap is per *fetch* (`curation.session_size`), so
   * a refresh hands you a fresh worklist and nothing resets at midnight —
   * "today" claimed a cadence the engine does not implement.
   *
   * It counts this worklist, not the store's backlog. Those differ whenever
   * the finders had more to say than the cap allows, and the queue is capped
   * on purpose: a finite list is a habit, an infinite one is a
   * chore. `GET /curation` returns only the slice, so the backlog is a number
   * this client does not have and will not imply.
   */
  private renderHeader(): HTMLElement {
    const header = document.createElement("div");
    header.className = "header";
    const count = document.createElement("span");
    count.className = "session-count";
    const n = this.cards.length;
    count.textContent = n === 1 ? "1 left in queue" : `${n} left in queue`;
    header.append(count);
    return header;
  }

  private renderControls(): HTMLElement {
    const controls = document.createElement("div");
    controls.className = "controls";

    const kindSel = document.createElement("select");
    for (const k of KINDS) {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = k === "" ? "All kinds" : k.replace(/_/g, " ");
      if (k === (this.options.kind ?? "")) opt.selected = true;
      kindSel.appendChild(opt);
    }
    kindSel.onchange = () => {
      this.options.kind = kindSel.value || undefined;
      void this.refresh();
    };

    const semLabel = document.createElement("label");
    const sem = document.createElement("input");
    sem.type = "checkbox";
    sem.checked = this.options.semantic ?? false;
    sem.onchange = () => {
      this.options.semantic = sem.checked;
      void this.refresh();
    };
    semLabel.append(sem, document.createTextNode("Semantic finders"));

    const refresh = document.createElement("button");
    refresh.textContent = "Refresh";
    refresh.onclick = () => void this.refresh();

    controls.append(kindSel, semLabel, refresh);
    return controls;
  }

  private renderCard(card: CurationCard): HTMLElement {
    const el = document.createElement("div");
    el.className = "card";

    const kind = document.createElement("div");
    kind.className = "kind";
    const dot = document.createElement("span");
    dot.className = "dot";
    kind.append(dot, document.createTextNode((card.kind ?? "").replace(/_/g, " ")));
    if (typeof card.leverage === "number") {
      const lev = document.createElement("span");
      lev.className = "leverage";
      lev.textContent = `leverage ${card.leverage.toFixed(2)}`;
      kind.appendChild(lev);
    }
    el.appendChild(kind);

    const diag = document.createElement("div");
    diag.className = "diagnostic";
    diag.textContent = card.diagnostic ?? "";
    el.appendChild(diag);

    // A CONTESTED card whose inconsistency basis fired carries the full
    // INCONSISTENCY id structurally (the diagnostic prose truncates it) —
    // link it to the contradiction's evidence graph, so "what is
    // this inconsistent with?" is one tap, not a copy-paste hunt.
    if (card.inconsistency_id) {
      const evidence = document.createElement("a");
      evidence.className = "subject-link";
      evidence.textContent = "show the conflicting beliefs (graph) →";
      evidence.href = routeHash("browse", {
        scope: "inconsistency",
        inconsistency_id: card.inconsistency_id,
      });
      const row = document.createElement("div");
      row.className = "diagnostic";
      row.appendChild(evidence);
      el.appendChild(row);
    }

    const refs = document.createElement("div");
    refs.className = "refs";
    refs.append(...this.renderRefs(card));
    el.appendChild(refs);

    el.appendChild(this.renderGestures(card, el));
    this.attachSwipe(card, el);
    return el;
  }

  private renderRefs(card: CurationCard): Node[] {
    const nodes: Node[] = [];
    const briefs = card.particles ?? [];
    if (briefs.length > 0) {
      // show what each belief actually says (claim + subject +
      // effective confidence + status) so the gesture — e.g. which of a
      // duplicate pair to keep — can be judged from the card alone.
      for (const b of briefs) {
        nodes.push(renderBrief(b));
      }
    } else {
      const ids = card.particle_ids ?? [];
      if (ids.length > 0) {
        nodes.push(document.createTextNode(`particles: ${ids.join(", ")}`));
      }
    }
    // on a duplicate-pair card the LLM judge ran (semantic finders on),
    // show its same-claim verdict under the brief so the merge decision is
    // informed by the model's read, not raw cosine. Absent in REPORT mode / when
    // the LLM was unavailable — the card then falls back to the brief.
    if (card.verdict) {
      nodes.push(renderVerdict(card.verdict));
    }
    if (card.corpus_url) {
      if (nodes.length) nodes.push(document.createElement("br"));
      const a = document.createElement("a");
      a.href = card.corpus_url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = card.corpus_url;
      nodes.push(a);
    }
    return nodes;
  }

  private renderGestures(card: CurationCard, cardEl: HTMLElement): HTMLElement {
    const wrap = document.createElement("div");
    wrap.className = "gestures";
    const offered = card.suggested_gestures ?? [];
    const kind = card.kind ?? "";
    const primary = primaryGesture(kind, offered);

    for (const g of offered) {
      const avail = gestureAvailability(g, kind);
      const btn = document.createElement("button");
      btn.className = "gesture";
      if (g === primary) btn.classList.add("primary");
      if (isDangerGesture(g)) btn.classList.add("danger");

      const label = document.createElement("span");
      label.textContent = gestureLabel(g);
      btn.appendChild(label);

      if (avail.kind === "deferred") {
        // Read-only-degrade hides write gestures entirely; deferred gestures are
        // shown disabled with their PDR note (honest about the gap, §5).
        if (this.readOnly) continue;
        btn.disabled = true;
        const note = document.createElement("span");
        note.className = "note";
        note.textContent = avail.pdr ? `${avail.note} (${avail.pdr})` : avail.note;
        btn.appendChild(note);
      } else {
        // v1 gesture. Hidden in read-only mode (a 403 means it would fail).
        if (this.readOnly) continue;
        btn.onclick = () => void this.dispatch(g, card, cardEl);
      }
      wrap.appendChild(btn);
    }
    return wrap;
  }

  // --- Swipe --------------------------------------------------------------

  private attachSwipe(card: CurationCard, el: HTMLElement): void {
    let startX = 0;
    let dx = 0;
    let active = false;

    const onStart = (x: number): void => {
      startX = x;
      dx = 0;
      active = true;
    };
    const onMove = (x: number): void => {
      if (!active) return;
      dx = x - startX;
      el.style.transform = `translateX(${dx}px) rotate(${dx / 40}deg)`;
    };
    const onEnd = (): void => {
      if (!active) return;
      active = false;
      const threshold = 96;
      const offered = card.suggested_gestures ?? [];
      const kind = card.kind ?? "";
      if (dx > threshold && !this.readOnly) {
        // Swipe right → the dominant safe (primary) gesture.
        const primary = primaryGesture(kind, offered);
        if (primary) {
          void this.dispatch(primary, card, el);
          return;
        }
      }
      if (dx < -threshold) {
        // Swipe left → snooze/dismiss the card out of the session. URL dismiss
        // and belief-snooze are both v1 now: URL → /corpus/links/dismiss,
        // belief cards → /curation/snooze. Read-only / no-write falls back to a
        // local advance.
        if (this.readOnly) {
          this.advance(el);
          return;
        }
        if (kind === "uncited_url" && card.corpus_url) {
          void this.dispatch("dismiss", card, el);
          return;
        }
        if (offered.includes("snooze")) {
          void this.dispatch("snooze", card, el);
          return;
        }
        this.advance(el);
        return;
      }
      el.style.transform = "";
    };

    el.addEventListener("touchstart", (e) => onStart(e.touches[0].clientX), {
      passive: true,
    });
    el.addEventListener("touchmove", (e) => onMove(e.touches[0].clientX), {
      passive: true,
    });
    el.addEventListener("touchend", onEnd);
    // Pointer (desktop) parity.
    el.addEventListener("pointerdown", (e) => onStart(e.clientX));
    el.addEventListener("pointermove", (e) => {
      if (e.buttons === 1) onMove(e.clientX);
    });
    el.addEventListener("pointerup", onEnd);
  }

  /** Drop the current card and render the next (local advance, no server write). */
  private advance(el?: HTMLElement): void {
    if (el) el.style.opacity = "0";
    this.cards.shift();
    this.render();
  }

  // --- Gesture dispatch -------------------------------------

  private async dispatch(
    gesture: string,
    card: CurationCard,
    el: HTMLElement,
  ): Promise<void> {
    try {
      const handled = await this.runGesture(gesture, card);
      if (handled) {
        this.advance(el);
      } else {
        el.style.transform = "";
      }
    } catch (e) {
      if (e instanceof ApiError && e.kind === "forbidden") {
        // Read-only engine: degrade the whole feed (§5).
        this.readOnly = true;
        this.render();
        return;
      }
      el.style.transform = "";
      this.root.prepend(banner("error", apiErrorMessage(e)));
    }
  }

  /** Returns true if the card was resolved (advance), false to keep it. */
  private async runGesture(gesture: string, card: CurationCard): Promise<boolean> {
    const client = this.deps.client;
    const ids = card.particle_ids ?? [];
    switch (gesture) {
      case "comment": {
        if (ids.length === 0) return false;
        const out = await openSheet({
          title: "Comment / resolve",
          message:
            "Resolve the inconsistency: pick how to treat the two beliefs and add a note.",
          fields: [
            {
              name: "action",
              label: "Action (PREFER_A / PREFER_B / BOTH_VALID / DEFER)",
              type: "text",
              placeholder: "DEFER",
              value: "DEFER",
            },
            { name: "note", label: "Note", type: "textarea" },
          ],
          confirmLabel: "Submit",
        });
        if (!out) return false;
        if (!this.deps.reviewerId) {
          throw new ApiError(
            "not-configured",
            "Set a reviewer id in settings before commenting.",
          );
        }
        const action = (out.action || "DEFER") as
          | "PREFER_A"
          | "PREFER_B"
          | "BOTH_VALID"
          | "DEFER";
        await client.review(ids[0], action, this.deps.reviewerId, out.note || undefined);
        return true;
      }
      case "merge": {
        if (ids.length < 2) return false;
        await client.link(ids[0], ids[1]);
        return true;
      }
      case "deposit": {
        // The card already carries the cited URL, so let the engine fetch +
        // extract it (POST /corpus/deposit/url) — one tap, no paste. That path
        // also reconciles the prior citing mentions to the new entry,
        // which is what actually clears this card; a content-only depositText
        // carries no URL and leaves the card standing. Fall back to a manual
        // paste only when the engine can't reach the URL (paywall / 403 /
        // SSRF-blocked → HTTP 400).
        if (card.corpus_url) {
          try {
            await client.depositUrl(card.corpus_url);
            return true;
          } catch (e) {
            if (!(e instanceof ApiError && e.kind === "http")) throw e;
            // fetch failed — fall through to the manual-paste path below.
          }
        }
        const out = await openSheet({
          title: "Deposit a source",
          message: card.corpus_url
            ? `The engine couldn't fetch ${card.corpus_url}. Paste its content to deposit it manually.`
            : "Paste source text to deposit.",
          fields: [{ name: "text", label: "Source text", type: "textarea" }],
          confirmLabel: "Deposit",
        });
        if (!out || !out.text) return false;
        await client.depositText(out.text);
        return true;
      }
      case "supersede": {
        if (ids.length === 0) return false;
        const out = await openSheet({
          title: "Edit (supersede)",
          message:
            "An edit is a supersession. The old belief is retired and this replaces it.",
          fields: [
            { name: "content", label: "Revised claim", type: "textarea" },
            { name: "subjects", label: "Subjects (comma-separated)", type: "text" },
            { name: "confidence", label: "Confidence (0–1)", type: "text", value: "0.8" },
          ],
          confirmLabel: "Supersede",
        });
        if (!out || !out.content) return false;
        // Operator-scoped supersede: works on the extracted beliefs
        // that fill the queue, not just own beliefs. POST /particles/{id}/supersede.
        await client.operatorSupersede(ids[0], {
          content: out.content,
          subject_names: (out.subjects || "")
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
          confidence: Number(out.confidence) || 0.8,
        });
        return true;
      }
      case "assign-subject": {
        // attach a subject to a NO_SUBJECT orphan via a
        // provenance-preserving operator-supersede. POST /particles/{id}/subjects.
        if (ids.length === 0) return false;
        const out = await openSheet({
          title: "Assign subject",
          message:
            "Attach a subject to this orphaned claim. The claim's confidence + provenance are preserved; only the subject linkage is added.",
          fields: [
            {
              name: "subject",
              label: "Subject name (resolved) or subject id",
              type: "text",
            },
          ],
          confirmLabel: "Assign",
        });
        if (!out || !out.subject) return false;
        // A 36-char UUID with dashes is treated as an explicit subject id;
        // anything else is a name run through the engine's standard resolver.
        const sval = out.subject.trim();
        const isId = /^[0-9a-fA-F-]{36}$/.test(sval);
        await client.assignSubject(ids[0], isId ? { subject_id: sval } : { subject_name: sval });
        return true;
      }
      case "retract": {
        // operator per-particle retract when the card names a belief
        // (the common queue case — an extracted belief). Falls back to
        // whole-source retract only when no belief id is present.
        if (ids.length === 1) {
          const out = await openSheet({
            title: "Retract belief",
            message: "Retract this single belief (ACTIVE → RETRACTED).",
            fields: [
              {
                name: "reason",
                label: "Reason",
                type: "text",
                value: "curation retract",
              },
            ],
            confirmLabel: "Retract",
          });
          if (!out) return false;
          await client.operatorRetract(ids[0], out.reason || "curation retract");
          return true;
        }
        const entryId = await this.resolveEntryId();
        if (!entryId) {
          this.root.prepend(
            banner(
              "info",
              "Whole-source retract needs the source entry id; supersede (edit) the belief instead.",
            ),
          );
          return false;
        }
        const plan = await client.retractCorpusEntry(entryId, "curation retract", true);
        const n = plan.retracted_ids?.length ?? 0;
        const ok = await confirmSheet(
          "Retract whole source",
          `This retracts all ${n} live belief(s) from this source. This cannot be undone here.`,
          "Retract all",
        );
        if (!ok) return false;
        await client.retractCorpusEntry(entryId, "curation retract", false);
        return true;
      }
      case "reindex": {
        await client.reindex();
        return true;
      }
      case "affirm": {
        // "still true" → POST /curation/affirm (BELIEF_AFFIRMED),
        // suppressing the card without touching confidence.
        await client.affirm(ids[0] ?? "", card.key);
        return true;
      }
      case "snooze": {
        // belief-snooze → POST /curation/snooze (CURATION_CARD_SNOOZED).
        await client.snoozeCard(card.key, ids);
        return true;
      }
      case "dismiss": {
        if (card.kind === "uncited_url" && card.corpus_url) {
          await client.dismissUrl(card.corpus_url);
          return true;
        }
        // permanent dismiss of a belief card → POST /curation/snooze
        // with no window (snooze_days omitted).
        await client.snoozeCard(card.key, ids);
        return true;
      }
      default:
        return false;
    }
  }

  /**
   * The curation card does not carry a corpus entry id (the queue is built from
   * belief-level finders), and there is no per-card source handle in v1, so
   * whole-source retract prompts the operator for the entry id. (A future card
   * field carrying the source entry would remove this prompt — out of v1 scope.)
   */
  private async resolveEntryId(): Promise<string | null> {
    const out = await openSheet({
      title: "Source entry id",
      message:
        "Whole-source retract operates per corpus entry. Enter the entry id of the source to retract.",
      fields: [{ name: "entry_id", label: "Corpus entry id", type: "text" }],
      confirmLabel: "Continue",
    });
    if (!out || !out.entry_id) return null;
    return out.entry_id;
  }

  private renderError(e: unknown): void {
    this.root.innerHTML = "";
    this.root.appendChild(this.renderHeader());
    if (e instanceof ApiError && e.kind === "not-configured") {
      this.deps.onNeedsSettings();
      return;
    }
    this.root.appendChild(banner("error", apiErrorMessage(e)));
  }
}

function banner(cls: "error" | "info" | "readonly", text: string): HTMLElement {
  const el = document.createElement("div");
  el.className = `banner ${cls}`;
  el.textContent = text;
  return el;
}

function apiErrorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  return `Unexpected error: ${String(e)}`;
}

/**
 * One particle brief: the claim text, then a meta line of subject(s)
 * · effective confidence · status — enough to judge the card's gesture without
 * a `particles show <id>` round-trip.
 */
function renderBrief(b: ParticleBrief): HTMLElement {
  const item = document.createElement("div");
  item.className = "particle";

  const claim = document.createElement("div");
  claim.className = "claim";
  claim.textContent = b.content ?? "";
  item.appendChild(claim);

  const bits: string[] = [];
  const subjects = b.subject_labels ?? [];
  if (subjects.length) bits.push(subjects.join(", "));
  if (typeof b.effective_confidence === "number") {
    bits.push(`conf ${b.effective_confidence.toFixed(2)}`);
  }
  if (b.status) bits.push(b.status);
  if (bits.length) {
    const meta = document.createElement("div");
    meta.className = "particle-meta";
    meta.textContent = bits.join(" · ");
    item.appendChild(meta);
  }
  return item;
}

/**
 * The LLM judge's advisory same-claim verdict on a duplicate-pair card
 *. Rendered under the brief, e.g. "LLM: same claim — safe to merge"
 * or "LLM: not a duplicate — <rationale>". Advisory only: the operator still
 * taps Merge / Dismiss; a DISTINCT verdict has already demoted the card.
 */
function renderVerdict(v: DuplicateVerdict): HTMLElement {
  const el = document.createElement("div");
  el.className = "verdict";

  let summary: string;
  switch (v.verdict) {
    case "PARAPHRASE":
      summary = "same claim — safe to merge";
      break;
    case "DISTINCT":
      summary = "not a duplicate";
      break;
    default:
      summary = "unsure";
      break;
  }
  let text = `LLM: ${summary}`;
  if (v.rationale) text += ` — ${v.rationale}`;
  el.textContent = text;
  return el;
}
