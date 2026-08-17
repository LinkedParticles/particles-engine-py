/*
 * Typed HTTP client of the Particles FastAPI engine (§6).
 *
 * Every request/response shape is imported from `./openapi`, the file
 * `openapi-typescript` generates from the committed `artifacts/openapi.json`
 *. The PWA never hand-writes CurationCard / ReviewRequest / … shapes,
 * so a breaking contract change surfaces as a TypeScript compile error rather
 * than a runtime parse failure. Run `npm run gen-api` (done automatically by
 * `npm run build` / `npm run typecheck`) to refresh `src/openapi.d.ts`.
 *
 * The client is store-free: it holds no schema logic and no reconciliation. It
 * presents the client bearer on every call and parses responses back
 * through the generated types. Because the engine serves this PWA same-origin
 *, the calls carry no `Origin` mismatch and trigger no CORS
 * preflight — the engine's lack of CORS middleware is a non-issue.
 */
import type { components } from "./openapi";

type Schemas = components["schemas"];

export type Subject = Schemas["Subject"];
export type CurationCard = Schemas["CurationCard"];
export type CurationQueueResponse = Schemas["CurationQueueResponse"];
export type CardKind = Schemas["CardKind"];
export type ParticleBrief = Schemas["ParticleBrief"];
/** the LLM judge's advisory same-claim verdict on a duplicate-pair card. */
export type DuplicateVerdict = Schemas["DuplicateVerdict"];
export type ReviewRequest = Schemas["ReviewRequest"];
export type ResolutionAction = Schemas["ResolutionAction"];
export type LinkRequest = Schemas["LinkRequest"];
export type ParticleRelation = Schemas["ParticleRelation"];
export type SubjectMergeRequest = Schemas["SubjectMergeRequest"];
export type SubjectMergeResponse = Schemas["SubjectMergeResponse"];
export type DepositTextRequest = Schemas["DepositTextRequest"];
export type DepositUrlRequest = Schemas["DepositUrlRequest"];
export type DepositResponse = Schemas["DepositResponse"];
export type ParticleSupersedeRequest = Schemas["ParticleSupersedeRequest"];
export type AgentWriteResult = Schemas["AgentWriteResult"];
export type CorpusRetractRequest = Schemas["CorpusRetractRequest"];
export type CorpusRetractResponse = Schemas["CorpusRetractResponse"];
/** operator-scoped supersede of a belief the operator does not own. */
export type OperatorSupersedeRequest = Schemas["OperatorSupersedeRequest"];
/** operator-scoped per-particle retract. */
export type OperatorRetractRequest = Schemas["OperatorRetractRequest"];
export type ParticleRetractEndpointResponse =
  Schemas["ParticleRetractEndpointResponse"];
/** assign a subject to a NO_SUBJECT orphan (provenance-preserving). */
export type ParticleSubjectAssignRequest = Schemas["ParticleSubjectAssignRequest"];
/** affirm / snooze curation operator-event writes. */
export type CurationAffirmRequest = Schemas["CurationAffirmRequest"];
export type CurationSnoozeRequest = Schemas["CurationSnoozeRequest"];
export type CurationEventResponse = Schemas["CurationEventResponse"];
export type ReindexRequest = Schemas["ReindexRequest"];
export type ReindexResponse = Schemas["ReindexResponse"];
export type CorpusLinksDismissRequest = Schemas["CorpusLinksDismissRequest"];
export type CorpusLinksDismissResponse = Schemas["CorpusLinksDismissResponse"];
export type HealthResponse = Schemas["HealthResponse"];
/** the query surface drives the existing POST /query contract. */
export type QueryRequest = Schemas["QueryRequest"];
export type QueryResponse = Schemas["QueryResponse"];
/** the scoped epistemic subgraph contract. */
export type GraphData = Schemas["GraphData"];
export type GraphNode = Schemas["GraphNode"];
export type GraphEdge = Schemas["GraphEdge"];
export type GraphParticleInfo = Schemas["GraphParticleInfo"];

/**
 * Categorised failure surfaced from an engine call (mirrors client).
 *
 * - `unauthorized` (401): the engine rejected the bearer token
 *. Surfaced as a clear "engine rejected the token" error.
 * - `forbidden` (403): belief writes are disabled on this store
 *   (`mcp.write.enabled_stores` default-deny). The feed degrades to
 *   a read-only review mode (write gestures hidden).
 * - `unavailable` (503): the engine is in dev-key mode and refuses a
 *   non-loopback caller. Surfaced as "engine refuses unauthenticated
 *   non-loopback access".
 * - `not-configured`: no engine URL or token configured — the fail-closed
 *   client posture. The PWA makes no call at all.
 */
export type ApiErrorKind =
  | "unauthorized"
  | "forbidden"
  | "unavailable"
  | "not-configured"
  | "not-found"
  | "network"
  | "http";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;

  constructor(kind: ApiErrorKind, message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

export interface EngineSettings {
  /**
   * The `engine.base_url` analog. Empty string = same-origin (the
   * default when served from the engine's `/app` mount): calls go
   * to the origin the app was loaded from, so no cross-origin preflight.
   */
  baseUrl: string;
  /** The client bearer, checked against the engine's PARTICLES_API_KEY. */
  bearerToken: string;
  /**
   * Proceed with no token, because this engine was *asked* and answered that it
   * serves unauthenticated — the loopback dev-key case, where the
   * bearer check is skipped entirely and any value would do.
   *
   * Set only by a successful `probeAuth()` on the settings screen, never
   * inferred and never a default. That keeps the fail-closed posture
   * intact where it means something — the app still makes no
   * call until the operator has been through settings — while removing the
   * absurdity it produced against a keyless engine, where the only way in was
   * to type a value the engine would ignore.
   */
  allowUnauthenticated: boolean;
}

/**
 * What the engine answers when asked, without a token, for something gated.
 *
 * The three defined outcomes are the states: a real key set (401), the
 * dev-key skip refusing a non-loopback caller (503), and the dev-key skip
 * serving a loopback caller (200).
 */
export type AuthProbe =
  | { kind: "open" }
  | { kind: "token-required" }
  | { kind: "refuses-unauthenticated" }
  | { kind: "unreachable"; detail: string };

/** Curation queue session controls — the existing `GET /curation` query params. */
export interface QueueOptions {
  /** One-run `curation.session_size` override (today's N). */
  limit?: number;
  /** Restrict to a single CardKind (e.g. "stale", "contested"). */
  kind?: string;
  /** Run the LLM-assisted finders (defaults to `curation.semantic`). */
  semantic?: boolean;
}

/** GET /graph wire params — also the #/browse deep-link params. */
export interface GraphParams {
  scope: "subject" | "query" | "inconsistency" | "projection";
  subject_id?: string;
  q?: string;
  /** A contradiction's evidence scope: the INCONSISTENCY id. */
  inconsistency_id?: string;
  /** Projection scope: engine-host manifest path + section. */
  manifest?: string;
  section?: string;
  hops?: number;
  history?: boolean;
  /** ISO-8601 past instant (as-of lens). */
  as_of?: string;
  max_nodes?: number;
}

/** Minimal injectable fetch so the client is testable without the global. */
export type FetchFn = typeof fetch;

export class ParticlesApiClient {
  private settings: EngineSettings;
  private readonly fetchFn: FetchFn;

  constructor(settings: EngineSettings, fetchFn?: FetchFn) {
    this.settings = settings;
    this.fetchFn = fetchFn ?? ((...args) => fetch(...args));
  }

  updateSettings(settings: EngineSettings): void {
    this.settings = settings;
  }

  /**
   * True when a bearer token is configured. The base URL may legitimately be
   * empty (same-origin serving), so it is NOT required — only the
   * bearer is. Fail-closed: with no token the PWA makes no call.
   */
  isConfigured(): boolean {
    return this.settings.bearerToken.trim().length > 0 || this.settings.allowUnauthenticated;
  }

  /**
   * Ask the engine, without a token, whether it needs one.
   *
   * Drives the empty-token path on the settings screen: rather than demanding
   * a value the operator may not have — or may not need — the app puts the
   * question to the engine, which is the only party that knows. `GET /events`
   * is the probe because it is gated unconditionally (not merely when
   * `api.require_auth_for_reads` is on, which would let a keyed engine answer
   * "open"), and it is a cheap read that drives no LLM call. `limit=1` keeps
   * the body small; the body is discarded either way.
   */
  async probeAuth(): Promise<AuthProbe> {
    try {
      await this.request<unknown>("GET", "/events?limit=1", undefined, { open: true });
      return { kind: "open" };
    } catch (e) {
      if (e instanceof ApiError) {
        if (e.kind === "unauthorized") return { kind: "token-required" };
        if (e.kind === "unavailable") return { kind: "refuses-unauthenticated" };
        return { kind: "unreachable", detail: e.message };
      }
      return { kind: "unreachable", detail: String(e) };
    }
  }

  private requireConfigured(): void {
    if (!this.isConfigured()) {
      throw new ApiError(
        "not-configured",
        "Configure the engine bearer token in settings to start curating.",
      );
    }
  }

  private url(path: string): string {
    // Empty baseUrl ⇒ same-origin relative path (the default served mount).
    const base = this.settings.baseUrl.replace(/\/+$/, "");
    return `${base}${path}`;
  }

  private headers(withBody: boolean): Record<string, string> {
    const h: Record<string, string> = {
      Accept: "application/json",
    };
    // The client bearer on every request. Omitted
    // only when no token is held at all, which reaches the wire on the one
    // open route (GET /health, called before onboarding): an empty
    // `Authorization: Bearer ` is a malformed header, and /health takes no
    // credential. Every other route is refused client-side before this runs.
    if (this.settings.bearerToken) {
      h.Authorization = `Bearer ${this.settings.bearerToken}`;
    }
    if (withBody) {
      h["Content-Type"] = "application/json";
    }
    return h;
  }

  /**
   * Core request helper. Fail-closed both ways: refuses to send
   * without a token, and maps 401 → unauthorized, 403 → forbidden, 503 →
   * unavailable, 404 → not-found, other non-2xx → http with the engine's
   * `ErrorResponse.detail` when present.
   *
   * `open` waives the client-side token check for a route the engine serves
   * unauthenticated. Only `GET /health` qualifies, and only because the
   * engine gates it nowhere (serves the app shell on the same
   * footing) — do not reach for this to work around a 401.
   */
  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    opts: { open?: boolean } = {},
  ): Promise<T> {
    if (!opts.open) this.requireConfigured();
    let resp: Response;
    try {
      resp = await this.fetchFn(this.url(path), {
        method,
        headers: this.headers(body !== undefined),
        body: body !== undefined ? JSON.stringify(body) : undefined,
      });
    } catch (e) {
      throw new ApiError(
        "network",
        `Could not reach the Particles engine${
          this.settings.baseUrl ? ` at ${this.settings.baseUrl}` : ""
        }: ${String(e)}`,
      );
    }

    if (resp.status === 401) {
      throw new ApiError(
        "unauthorized",
        this.settings.bearerToken
          ? "The engine rejected the bearer token. Check the token in settings."
          : "This engine requires a bearer token and none is set — it may have " +
            "been restarted with a PARTICLES_API_KEY. Add it in settings.",
        401,
      );
    }
    if (resp.status === 403) {
      throw new ApiError(
        "forbidden",
        "Belief writes are disabled on this engine (read-only).",
        403,
      );
    }
    if (resp.status === 503) {
      throw new ApiError(
        "unavailable",
        await this.detail(resp, "The engine refuses unauthenticated non-loopback access."),
        503,
      );
    }
    if (resp.status === 404) {
      throw new ApiError("not-found", await this.detail(resp, "Not found."), 404);
    }
    if (!resp.ok) {
      throw new ApiError(
        "http",
        await this.detail(resp, `Engine returned HTTP ${resp.status}.`),
        resp.status,
      );
    }
    return (await resp.json()) as T;
  }

  /**
   * Pull the `detail` message when the body carries one. Handles both shapes
   * the engine sends: the plain-string `ErrorResponse.detail`, and FastAPI's
   * structured request-validation detail (a list of `{loc, msg}` objects on a
   * 422) — previously the latter fell through to a useless generic
   * "Engine returned HTTP 422." with the actual reason discarded.
   */
  private async detail(resp: Response, fallback: string): Promise<string> {
    try {
      const data = (await resp.json()) as { detail?: unknown };
      const d = data ? data.detail : undefined;
      if (typeof d === "string") {
        return d;
      }
      if (Array.isArray(d) && d.length) {
        const parts = d.map((e) => {
          if (e && typeof e === "object") {
            const err = e as { loc?: unknown; msg?: unknown };
            const loc = Array.isArray(err.loc)
              ? err.loc.filter((x) => x !== "query" && x !== "body").join(".")
              : "";
            const msg = typeof err.msg === "string" ? err.msg : JSON.stringify(e);
            return loc ? `${loc}: ${msg}` : msg;
          }
          return String(e);
        });
        return parts.join(" · ");
      }
    } catch {
      /* fall through to the fallback message */
    }
    return fallback;
  }

  // --- Read ---------------------------------------------------------------

  /**
   * GET /health — liveness + SDK version.
   *
   * The one call made before a bearer is saved: the engine serves it
   * unauthenticated, and its `version` is what identifies the software this
   * app is talking to, so the footer can state it on the settings screen —
   * where knowing what you are pointed at matters most.
   */
  async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>("GET", "/health", undefined, { open: true });
  }

  /** GET /curation — the leverage-ranked, finite "today's N" queue. */
  async curation(opts: QueueOptions = {}): Promise<CurationQueueResponse> {
    const params = new URLSearchParams();
    if (opts.limit !== undefined) params.set("limit", String(opts.limit));
    if (opts.kind) params.set("kind", opts.kind);
    if (opts.semantic !== undefined) params.set("semantic", String(opts.semantic));
    const qs = params.toString();
    return this.request<CurationQueueResponse>(
      "GET",
      `/curation${qs ? `?${qs}` : ""}`,
    );
  }

  /**
   * POST /query — tag-aware semantic query with the NL cited answer
   * (non-streaming v1 — streaming is deferred).
   * Bearer-gated + rate-limited engine-side: this drives a paid embedding +
   * completion per call.
   */
  async query(question: string): Promise<QueryResponse> {
    const req: QueryRequest = { question };
    return this.request<QueryResponse>("POST", "/query", req);
  }

  /**
   * GET /graph — one scoped epistemic subgraph.
   * The params mirror the wire contract one-to-one (they are also the
   * #/browse deep-link params); every epistemic quantity in the response is
   * server-computed — this client only renders.
   */
  async graph(params: GraphParams): Promise<GraphData> {
    const sp = new URLSearchParams();
    sp.set("scope", params.scope);
    if (params.subject_id) sp.set("subject_id", params.subject_id);
    if (params.q) sp.set("q", params.q);
    if (params.inconsistency_id) sp.set("inconsistency_id", params.inconsistency_id);
    if (params.manifest) sp.set("manifest", params.manifest);
    if (params.section) sp.set("section", params.section);
    if (params.hops !== undefined) sp.set("hops", String(params.hops));
    if (params.history) sp.set("history", "true");
    if (params.as_of) sp.set("as_of", params.as_of);
    if (params.max_nodes !== undefined) sp.set("max_nodes", String(params.max_nodes));
    return this.request<GraphData>("GET", `/graph?${sp.toString()}`);
  }

  /**
   * GET /subjects — list subjects; `order: "degree"` returns hub subjects
   * first (descending ACTIVE-link count, server-side). `limit: 1` on top is
   * the Browse route's seed call: "the most-connected subject".
   */
  async subjects(opts: { order?: "name" | "degree"; limit?: number } = {}): Promise<Subject[]> {
    const sp = new URLSearchParams();
    if (opts.order && opts.order !== "name") sp.set("order", opts.order);
    if (opts.limit !== undefined) sp.set("limit", String(opts.limit));
    const qs = sp.toString();
    return this.request<Subject[]>("GET", `/subjects${qs ? `?${qs}` : ""}`);
  }

  // --- Write gestures — each one authenticated call ----------

  /** comment → POST /review/{id} (resolve an INCONSISTENCY with an action + note). */
  async review(
    particleId: string,
    action: ResolutionAction,
    reviewerId: string,
    note?: string,
  ): Promise<unknown> {
    const req: ReviewRequest = {
      action,
      reviewer_id: reviewerId,
      ...(note !== undefined ? { note } : {}),
    };
    return this.request<unknown>(
      "POST",
      `/review/${encodeURIComponent(particleId)}`,
      req,
    );
  }

  /** merge (same-subject duplicate) → POST /links (CO_EVIDENTIAL). */
  async link(particleA: string, particleB: string): Promise<ParticleRelation> {
    const req: LinkRequest = {
      particle_a: particleA,
      particle_b: particleB,
      relation_type: "CO_EVIDENTIAL",
    };
    return this.request<ParticleRelation>("POST", "/links", req);
  }

  /** merge (cross-subject) → POST /subjects/{source_id}/merge. */
  async mergeSubjects(
    sourceId: string,
    targetId: string,
  ): Promise<SubjectMergeResponse> {
    const req: SubjectMergeRequest = { target_id: targetId };
    return this.request<SubjectMergeResponse>(
      "POST",
      `/subjects/${encodeURIComponent(sourceId)}/merge`,
      req,
    );
  }

  /** deposit → POST /corpus/deposit/text. */
  async depositText(text: string): Promise<DepositResponse> {
    const req: DepositTextRequest = {
      text,
      source_type: "CONVERSATION",
      deposited_by: "web-ui",
      tags: [],
    };
    return this.request<DepositResponse>("POST", "/corpus/deposit/text", req);
  }

  /**
   * deposit a file → POST /corpus/deposit/file (multipart — the
   * shipped endpoint gains its first UI consumer). The browser sets the
   * multipart boundary itself, so this bypasses the JSON `request` helper;
   * the same bearer + error mapping apply.
   */
  async depositFile(file: File): Promise<DepositResponse> {
    this.requireConfigured();
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("deposited_by", "web-ui");
    let resp: Response;
    try {
      resp = await this.fetchFn(this.url("/corpus/deposit/file"), {
        method: "POST",
        // No Content-Type: the browser sets multipart/form-data + boundary.
        headers: {
          Authorization: `Bearer ${this.settings.bearerToken}`,
          Accept: "application/json",
        },
        body: form,
      });
    } catch (e) {
      throw new ApiError(
        "network",
        `Could not reach the Particles engine${
          this.settings.baseUrl ? ` at ${this.settings.baseUrl}` : ""
        }: ${String(e)}`,
      );
    }
    if (!resp.ok) {
      throw new ApiError(
        resp.status === 401
          ? "unauthorized"
          : resp.status === 403
            ? "forbidden"
            : resp.status === 503
              ? "unavailable"
              : "http",
        await this.detail(resp, `Engine returned HTTP ${resp.status}.`),
        resp.status,
      );
    }
    return (await resp.json()) as DepositResponse;
  }

  /**
   * deposit-from-URL → POST /corpus/deposit/url. The engine fetches + extracts
   * the URL itself (importer routing + SSRF guard) and reconciles the prior
   * citing mentions to the new entry, which is what clears an
   * uncited-URL card — a content-only `depositText` cannot, since it carries no
   * source URL to bind. Throws `ApiError("http", …, 400)` when the engine cannot
   * fetch the URL (paywall / 403 / SSRF-blocked); the caller falls back to a
   * manual paste.
   */
  async depositUrl(url: string): Promise<DepositResponse> {
    const req: DepositUrlRequest = {
      url,
      deposited_by: "web-ui",
      tags: [],
    };
    return this.request<DepositResponse>("POST", "/corpus/deposit/url", req);
  }

  /**
   * supersede / edit → POST /particles/supersede (own-beliefs-only
   * §4a/§6: rejects HUMAN_REVIEW + other identities; 403 when writes disabled).
   */
  async supersede(req: ParticleSupersedeRequest): Promise<AgentWriteResult> {
    return this.request<AgentWriteResult>("POST", "/particles/supersede", req);
  }

  /**
   * operator edit-as-supersede → POST /particles/{id}/supersede. Works
   * on a belief the operator does NOT own (incl. extracted beliefs that fill a
   * curation queue): skips only the ownership check, keeps the HUMAN_REVIEW +
   * ACTIVE guards, and stays gated by `mcp.write.enabled_stores` (403 when off).
   */
  async operatorSupersede(
    particleId: string,
    req: OperatorSupersedeRequest,
  ): Promise<AgentWriteResult> {
    return this.request<AgentWriteResult>(
      "POST",
      `/particles/${encodeURIComponent(particleId)}/supersede`,
      req,
    );
  }

  /**
   * operator per-particle retract → POST /particles/{id}/retract
   *. Retracts one extracted/operator-sourced belief by id (vs the
   * blunt whole-source `retractCorpusEntry`).
   */
  async operatorRetract(
    particleId: string,
    reason: string,
  ): Promise<ParticleRetractEndpointResponse> {
    const req: OperatorRetractRequest = { reason };
    return this.request<ParticleRetractEndpointResponse>(
      "POST",
      `/particles/${encodeURIComponent(particleId)}/retract`,
      req,
    );
  }

  /**
   * assign-subject → POST /particles/{id}/subjects. Attaches a subject
   * to a NO_SUBJECT orphan via a provenance-preserving operator-supersede: the
   * successor keeps the predecessor's confidence + extractor_ref + source, only
   * the subject linkage is corrected. Resolution by explicit subject_id or by
   * subject_name through the standard resolver.
   */
  async assignSubject(
    particleId: string,
    req: ParticleSubjectAssignRequest,
  ): Promise<AgentWriteResult> {
    return this.request<AgentWriteResult>(
      "POST",
      `/particles/${encodeURIComponent(particleId)}/subjects`,
      req,
    );
  }

  /** affirm ("still true") → POST /curation/affirm (BELIEF_AFFIRMED). */
  async affirm(
    particleId: string,
    cardKey?: string,
  ): Promise<CurationEventResponse> {
    const req: CurationAffirmRequest = {
      particle_id: particleId,
      ...(cardKey !== undefined ? { card_key: cardKey } : {}),
    };
    return this.request<CurationEventResponse>("POST", "/curation/affirm", req);
  }

  /**
   * snooze / dismiss a belief card → POST /curation/snooze (
   * CURATION_CARD_SNOOZED). `snoozeDays` omitted ⇒ permanent dismiss.
   */
  async snoozeCard(
    cardKey: string,
    particleIds: string[],
    snoozeDays?: number,
  ): Promise<CurationEventResponse> {
    const req: CurationSnoozeRequest = {
      card_key: cardKey,
      particle_ids: particleIds,
      ...(snoozeDays !== undefined ? { snooze_days: snoozeDays } : {}),
    };
    return this.request<CurationEventResponse>("POST", "/curation/snooze", req);
  }

  /**
   * retract (whole source) → POST /corpus/{entry_id}/retract. Retracts ALL live
   * particles of the entry; the UI confirms the blast radius via a dry-run first
   * ("this retracts all N beliefs" confirm).
   */
  async retractCorpusEntry(
    entryId: string,
    reason: string,
    dryRun = false,
  ): Promise<CorpusRetractResponse> {
    const req: CorpusRetractRequest = { reason, dry_run: dryRun };
    return this.request<CorpusRetractResponse>(
      "POST",
      `/corpus/${encodeURIComponent(entryId)}/retract`,
      req,
    );
  }

  /** reindex → POST /reindex (over FAILED snapshots). */
  async reindex(): Promise<ReindexResponse> {
    const req: ReindexRequest = { include_failed: true, rate_limit_per_minute: 100 };
    return this.request<ReindexResponse>("POST", "/reindex", req);
  }

  /** dismiss (uncited_url) → POST /corpus/links/dismiss. */
  async dismissUrl(
    url: string,
    snoozeDays?: number,
  ): Promise<CorpusLinksDismissResponse> {
    const req: CorpusLinksDismissRequest = {
      url,
      ...(snoozeDays !== undefined ? { snooze_days: snoozeDays } : {}),
    };
    return this.request<CorpusLinksDismissResponse>(
      "POST",
      "/corpus/links/dismiss",
      req,
    );
  }
}
