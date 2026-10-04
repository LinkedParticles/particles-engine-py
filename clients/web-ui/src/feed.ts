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
  ConflictBrief,
  CurationCard,
  DuplicateVerdict,
  ParticleBrief,
  ParticlesApiClient,
  QueueOptions,
  ResolutionAction,
} from "./api";
import {
  gestureAvailability,
  gestureHint,
  gestureLabel,
  isDangerGesture,
  primaryGesture,
} from "./gestures";
import { routeHash } from "./router";
import { confirmSheet, openSheet } from "./sheet";
import { isQuarantined, reasonNote, statusText } from "./status";

export interface FeedDeps {
  client: ParticlesApiClient;
  reviewerId: string;
  onNeedsSettings: () => void;
}

const KINDS = [
  "",
  "stale",
  "confidence_decay",
  "inconsistency",
  "contested",
  "contradiction",
  "retraction_cascade",
  "broken_provenance",
  "no_subject",
  "gated_subjects",
  "duplicate_pair",
  "uncited_url",
  "failed_snapshots",
];

export class CurationFeed {
  private deps: FeedDeps;
  private root: HTMLElement;
  private cards: CurationCard[] = [];
  /** Open cards in all (the server's `open_count`), of which `cards` is the head. */
  private openCount = 0;
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
      // `resp.count` is deliberately not kept: it is len(cards) at fetch
      // time, which the header already shows. `open_count` is the backlog the
      // batch was cut from; an older engine without it reads as the batch.
      this.openCount = Math.max(resp.open_count ?? 0, this.cards.length);
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
      const big = document.createElement("div");
      big.className = "big";
      big.textContent = "✓";
      const line = document.createElement("div");
      line.textContent =
        this.openCount > 0
          ? `Batch done. ${this.openCount} more open, ranked below this batch. ` +
            "Refresh for the next batch."
          : "Queue clear. Nothing to curate right now.";
      done.append(big, line);
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
   * The queue's own line: this batch, and the backlog it was cut from.
   *
   * The batch is capped on purpose (`curation.session_size`): a
   * short session does not fatigue. The cap is per *fetch*, so a refresh
   * hands you the next batch; nothing resets at midnight. Showing only the
   * batch size read as the whole backlog ("7 left" when 154 were open), so the
   * header names both ("Top 7 of 154 cards open"): `open_count` comes from the
   * engine, which is the only party that knows it.
   */
  private renderHeader(): HTMLElement {
    const header = document.createElement("div");
    header.className = "header";
    const count = document.createElement("span");
    count.className = "session-count";
    const n = this.cards.length;
    const open = this.openCount;
    const cards = (k: number): string =>
      `${k.toLocaleString()} ${k === 1 ? "card" : "cards"}`;
    count.textContent =
      open > n
        ? `Top ${n.toLocaleString()} of ${cards(open)} open, by leverage`
        : `${cards(n)} open`;
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
    kind.append(
      dot,
      document.createTextNode(card.title ?? (card.kind ?? "").replace(/_/g, " ")),
    );
    if (typeof card.leverage === "number") {
      const lev = document.createElement("span");
      lev.className = "leverage";
      lev.textContent = `leverage ${card.leverage.toFixed(2)}`;
      kind.appendChild(lev);
    }
    el.appendChild(kind);

    // What the card asks, before the evidence (the engine's KIND_TITLES).
    if (card.question) {
      const q = document.createElement("div");
      q.className = "question";
      q.textContent = card.question;
      el.appendChild(q);
    }

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
    if (card.conflict && (card.conflict.a || card.conflict.b)) {
      // A contested card's belief is one side of a conflict; show both, in
      // review's A / B order, so "does it stand?" can be judged here.
      // A contested card is about one of the sides; a conflict card is the
      // record itself, so none of its sides is singled out.
      const flagged = card.kind === "contested" ? (card.particle_ids ?? []) : [];
      nodes.push(...renderConflict(card.conflict, flagged));
    } else if (briefs.length > 0) {
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
      label.textContent = gestureLabel(g, kind);
      btn.appendChild(label);
      const hint = gestureHint(g, kind);
      if (hint) btn.title = hint;

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
    const holder = document.createElement("div");
    holder.appendChild(wrap);
    const help = this.renderGestureHelp(offered, kind);
    if (help) holder.appendChild(help);
    return holder;
  }

  /**
   * "What do these do?": one line per offered gesture. A tooltip is invisible
   * on touch, and "Still true" vs "Snooze" is the choice the curator has to
   * make, so the meanings are one tap away on every card.
   */
  private renderGestureHelp(offered: string[], kind: string): HTMLElement | null {
    const rows = offered
      .filter((g) => !this.readOnly && gestureAvailability(g, kind).kind === "v1")
      .map((g) => [gestureLabel(g, kind), gestureHint(g, kind)] as const)
      .filter(([, hint]) => hint);
    if (rows.length === 0) return null;
    const details = document.createElement("details");
    details.className = "gesture-help";
    const summary = document.createElement("summary");
    summary.textContent = "What do these do?";
    const dl = document.createElement("dl");
    for (const [label, hint] of rows) {
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = hint;
      dl.append(dt, dd);
    }
    details.append(summary, dl);
    return details;
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

  /**
   * Drop the current card and render the next. `resolved` is true when a
   * gesture wrote to the engine (the card left the backlog too), false for a
   * local skip in read-only mode (it will be back on the next fetch).
   */
  private advance(el?: HTMLElement, resolved = false): void {
    if (el) el.style.opacity = "0";
    this.cards.shift();
    if (resolved) this.openCount = Math.max(this.openCount - 1, this.cards.length);
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
        this.advance(el, true);
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
      case "resolve":
      case "comment": {
        // Resolve the INCONSISTENCY behind the card. The review
        // route takes the record's id, never a member belief's (which 404s).
        // `comment` is the name an engine before 1.162 sends for the same thing.
        const conflict = card.conflict;
        const inconsistencyId = card.inconsistency_id ?? conflict?.inconsistency_id;
        if (!inconsistencyId) {
          this.root.prepend(
            banner("info", "This card names no INCONSISTENCY to resolve; snooze or affirm it instead."),
          );
          return false;
        }
        if (!this.deps.reviewerId) {
          throw new ApiError(
            "not-configured",
            "Set a reviewer id in settings before resolving.",
          );
        }
        const quote = (b: ParticleBrief | null | undefined): string =>
          b ? `“${truncate(b.content ?? "", 90)}”` : "(no longer in the store)";
        // Offer exactly the engine's `resolve_actions` (PARITY rule 3): it
        // withholds "Keep A" when A is no longer active, for instance, since
        // that choice would close the conflict with neither claim standing.
        // An older engine sends none, so every action is offered as before.
        const offered = card.resolve_actions ?? [
          "PREFER_A",
          "PREFER_B",
          "BOTH_VALID",
          "DISCARD",
          "DEFER",
        ];
        const options = resolveOptions(conflict).filter((o) => offered.includes(o.value));
        const withheld = RESOLVE_OPTIONS.filter(
          (o) => o.value !== "DEFER" && !offered.includes(o.value),
        );
        const note = withheld.map((o) => `\n\n${withheldNote(o, conflict)}`).join("");
        const out = await openSheet({
          title: "Resolve the conflict",
          message:
            `A: ${quote(conflict?.a)}\n` +
            `B: ${quote(conflict?.b)}` +
            note +
            (card.resolve_actions ? "\n\nTo decide later, cancel and snooze the card." : ""),
          fields: [
            {
              name: "action",
              label: "Resolution",
              type: "choice",
              options,
            },
            { name: "note", label: "Note (optional)", type: "textarea" },
          ],
          confirmLabel: "Resolve",
        });
        if (!out || !out.action) return false;
        await client.review(
          inconsistencyId,
          out.action as ResolutionAction,
          this.deps.reviewerId,
          out.note || undefined,
        );
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
            { name: "reason", label: "Reason", type: "text", value: "curation edit" },
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
          // Recorded on the PARTICLE_SUPERSEDED event; required
          // on the operator path.
          reason: (out.reason || "").trim() || "curation edit",
        });
        return true;
      }
      case "assign-subject": {
        // Attach a subject to a NO_SUBJECT orphan, in place.
        // POST /particles/{id}/subjects.
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
      case "relink": {
        // one batch over every recoverable orphan. Preview first,
        // then confirm, the way whole-source retract shows its blast radius.
        const plan = await client.relinkGated(true);
        const ok = await confirmSheet(
          "Link subjects",
          `Link ${plan.recoverable} belief(s) to the project-scoped subjects they name? Nothing else about them changes.`,
          "Link",
        );
        if (!ok) return false;
        await client.relinkGated(false);
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
  if (b.status) bits.push(statusText(b.status, b.status_reason));
  if (b.asserted_at) bits.push(`asserted ${String(b.asserted_at).slice(0, 10)}`);
  if (bits.length) {
    const meta = document.createElement("div");
    meta.className = "particle-meta";
    meta.textContent = bits.join(" · ");
    item.appendChild(meta);
  }
  // The raw reason stays above, verbatim; this is its plain reading.
  const note = reasonNote(b.status_reason);
  if (note) {
    const why = document.createElement("div");
    why.className = "particle-meta";
    why.textContent = `${note[0].toUpperCase()}${note.slice(1)}.`;
    item.appendChild(why);
  }
  // Where it came from: without a subject, the source is the only context.
  if (b.source_uri) {
    const src = document.createElement("div");
    src.className = "particle-meta";
    src.appendChild(document.createTextNode("source: "));
    if (/^https?:\/\//.test(b.source_uri)) {
      const a = document.createElement("a");
      a.href = b.source_uri;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = b.source_uri;
      src.appendChild(a);
    } else {
      src.appendChild(document.createTextNode(b.source_uri));
    }
    item.appendChild(src);
  }
  return item;
}

/**
 * Both sides of an INCONSISTENCY, labelled A and B in the order the Resolve
 * sheet's "Keep A" / "Keep B" name them. On a contested card the card's own
 * belief is outlined so the curator knows which side the card is about. A
 * census record's further members follow their side.
 */
function renderConflict(conflict: ConflictBrief, flaggedIds: string[]): Node[] {
  const nodes: Node[] = [];
  const sides: [string, ParticleBrief | null | undefined][] = [
    ["A", conflict.a],
    ...(conflict.further_a ?? []).map((b): [string, ParticleBrief] => ["A", b]),
    ["B", conflict.b],
    ...(conflict.further_b ?? []).map((b): [string, ParticleBrief] => ["B", b]),
  ];
  for (const [side, brief] of sides) {
    const label = document.createElement("div");
    label.className = "side-label";
    const flagged = !!brief && flaggedIds.includes(brief.particle_id);
    label.textContent = flagged ? `Claim ${side} · this card's belief` : `Claim ${side}`;
    nodes.push(label);
    if (brief) {
      const el = renderBrief(brief);
      if (flagged) el.classList.add("flagged");
      nodes.push(el);
    } else {
      const gone = document.createElement("div");
      gone.className = "particle-meta";
      gone.textContent = "No longer in the store.";
      nodes.push(gone);
    }
  }
  return nodes;
}

/** The resolve sheet's choices, in menu order; filtered per card. */
const RESOLVE_OPTIONS: { value: string; label: string; detail: string }[] = [
  {
    value: "PREFER_A",
    label: "Keep A",
    detail: "A is right. B is demoted, and A's source may gain trust over B's.",
  },
  {
    value: "PREFER_B",
    label: "Keep B",
    detail: "B is right. A is demoted, and B's source may gain trust over A's.",
  },
  {
    value: "BOTH_VALID",
    label: "Both valid",
    detail: "They do not really conflict. Both stay, and the conflict closes.",
  },
  {
    value: "DISCARD",
    label: "Discard both",
    detail: "Neither is worth keeping. Both are retracted.",
  },
  {
    value: "DEFER",
    label: "Decide later",
    detail: "Record a note and leave the conflict open.",
  },
];

type SideFate = "active" | "restored" | "inactive";

/** What a resolution that keeps this claim does to it: an ACTIVE claim stays,
 * a quarantined one becomes ACTIVE, and any other inactive claim stays as it is
 * (review never re-promotes a demoted claim). */
function fateOf(b: ParticleBrief | null | undefined): SideFate {
  if (!b || b.status === "ACTIVE") return "active";
  return isQuarantined(b.status, b.status_reason) ? "restored" : "inactive";
}

/** RESOLVE_OPTIONS with "Keep B" and "Both valid" worded for this record's
 * claims, since what they do depends on whether each side is ACTIVE. */
function resolveOptions(
  conflict: ConflictBrief | null | undefined,
): { value: string; label: string; detail: string }[] {
  const a = fateOf(conflict?.a);
  const b = fateOf(conflict?.b);
  const fate = (side: string, f: SideFate): string =>
    f === "active"
      ? `${side} stays active.`
      : f === "restored"
        ? `${side} becomes an active belief.`
        : `${side} stays inactive.`;
  return RESOLVE_OPTIONS.map((o) => {
    if (o.value === "PREFER_B" && (a !== "active" || b === "restored")) {
      const kept = b === "restored" ? "B is right and becomes an active belief." : "B is right.";
      const lost = a === "active" ? "A is demoted" : "A stays inactive";
      return { ...o, detail: `${kept} ${lost}, and B's source may gain trust over A's.` };
    }
    if (o.value === "BOTH_VALID" && (a !== "active" || b !== "active")) {
      return {
        ...o,
        detail: `They do not really conflict, and the conflict closes. ${fate("A", a)} ${fate("B", b)}`,
      };
    }
    return o;
  });
}

/** Why one resolution is withheld, naming the claim's recorded status. */
function withheldNote(
  option: { value: string; label: string },
  conflict: ConflictBrief | null | undefined,
): string {
  const side = option.value === "PREFER_A" ? "A" : "B";
  const brief = side === "A" ? conflict?.a : conflict?.b;
  if (!brief) {
    return `Not offered: ${option.label}. Claim ${side} is no longer in the store.`;
  }
  const why = reasonNote(brief.status_reason);
  return (
    `Not offered: ${option.label}. Claim ${side} is ` +
    `${statusText(brief.status, brief.status_reason)}` +
    (why ? ` (${why})` : "") +
    ", and resolving this conflict cannot make it active again."
  );
}

function truncate(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`;
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
