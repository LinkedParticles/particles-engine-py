/*
 * Typed HTTP client of the Particles FastAPI engine (§6).
 *
 * Every request/response shape is imported from `./openapi`, the file
 * `openapi-typescript` generates from the committed `artifacts/openapi.json`.
 * The PWA never hand-writes CurationCard / ReviewRequest / … shapes,
 * so a breaking contract change surfaces as a TypeScript compile error rather
 * than a runtime parse failure. Run `npm run gen-api` (done automatically by
 * `npm run build` / `npm run typecheck`) to refresh `src/openapi.d.ts`.
 *
 * The client is store-free: it holds no schema logic and no reconciliation. It
 * presents the client bearer on every call and parses responses back
 * through the generated types. Because the engine serves this PWA same-origin,
 * the calls carry no `Origin` mismatch and trigger no CORS
 * preflight — the engine's lack of CORS middleware is a non-issue.
 */
import type { components } from "./openapi";
import {
  ApiError,
  CurationClient,
  type ConnectionFacts,
  type WireRequest,
  type WireResponse,
  detailOf,
  errorForStatus,
  networkError,
} from "../../curation-core/src/index.ts";

export { ApiError, type ApiErrorKind } from "../../curation-core/src/index.ts";

type Schemas = components["schemas"];

export type Subject = Schemas["Subject"];
export type CurationCard = Schemas["CurationCard"];
export type CurationQueueResponse = Schemas["CurationQueueResponse"];
export type CardKind = Schemas["CardKind"];
export type ParticleBrief = Schemas["ParticleBrief"];
/** which projects a briefed belief is in view for (display only). */
export type ObserverScopeBrief = Schemas["ObserverScopeBrief"];
/** Both sides (claims A and B) of the INCONSISTENCY behind a contested card. */
export type ConflictBrief = Schemas["ConflictBrief"];
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
export type RelinkGatedRequest = Schemas["RelinkGatedRequest"];
export type RelinkReport = Schemas["RelinkReport"];
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
  /** One-run `curation.session_size` override (the batch size). */
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
  /**
   * The curation calls, made by the shared curation core over this client's
   * connection: the same bearer, the same fail-closed check.
   */
  readonly core: CurationClient;

  constructor(settings: EngineSettings, fetchFn?: FetchFn) {
    this.settings = settings;
    this.fetchFn = fetchFn ?? ((...args) => fetch(...args));
    this.core = new CurationClient({
      transport: (req) => {
        this.requireConfigured();
        return this.send(req);
      },
      facts: () => this.facts(),
      depositedBy: "web-ui",
    });
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

  private facts(): ConnectionFacts {
    return { hasToken: !!this.settings.bearerToken, baseUrl: this.settings.baseUrl };
  }

  /** Put one request on the wire; the body is parsed, undefined when none. */
  private async send(req: WireRequest): Promise<WireResponse> {
    let resp: Response;
    try {
      resp = await this.fetchFn(this.url(req.path), {
        method: req.method,
        headers: this.headers(req.body !== undefined),
        body: req.body !== undefined ? JSON.stringify(req.body) : undefined,
      });
    } catch (e) {
      throw networkError(this.facts(), e);
    }
    let body: unknown;
    try {
      body = await resp.json();
    } catch {
      body = undefined;
    }
    return { status: resp.status, body };
  }

  /**
   * Core request helper. Fail-closed both ways: refuses to send
   * without a token, and maps each failure with the curation core's rules
   * (401 unauthorized, 403 forbidden, 503 unavailable, 404 not-found, other
   * non-2xx http with the engine's detail).
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
    const resp = await this.send(body === undefined ? { method, path } : { method, path, body });
    const failure = errorForStatus(resp.status, resp.body, this.facts());
    if (failure) throw failure;
    return resp.body as T;
  }

  private async detail(resp: Response, fallback: string): Promise<string> {
    try {
      return detailOf(await resp.json(), fallback);
    } catch {
      return fallback;
    }
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

  /**
   * POST /query — tag-aware semantic query with the NL cited answer
   * (non-streaming v1 — streaming is deferred).
   * Bearer-gated + rate-limited engine-side: this drives a paid embedding +
   * completion per call.
   */
  async query(question: string): Promise<QueryResponse> {
    // a refused answer withholds its nearest beliefs by default.
    // This view is the person-facing display they were kept for,
    // so it asks for them; a re-query would pay for a second answer.
    const req: QueryRequest = { question, include_refused_candidates: true };
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
  depositText(text: string): Promise<DepositResponse> {
    return this.core.depositText(text);
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
  depositUrl(url: string): Promise<DepositResponse> {
    return this.core.depositUrl(url);
  }
}
