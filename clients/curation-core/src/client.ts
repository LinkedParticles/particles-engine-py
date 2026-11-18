/*
 * The engine calls behind every curation action.
 *
 * A client hands the core a `Transport`: the one function that puts a request
 * on the wire (the web UI's fetch, the Claude Code pane's `$.http.fetch`). The
 * core builds every path and body and maps every answer, so each client sends
 * exactly the same requests and reports failures the same way.
 */
import type {
  AgentWriteResult,
  CorpusLinksDismissRequest,
  CorpusLinksDismissResponse,
  CorpusRetractRequest,
  CorpusRetractResponse,
  CurationAffirmRequest,
  CurationCard,
  CurationEventResponse,
  CurationGestureResponse,
  CurationQueueResponse,
  CurationSnoozeRequest,
  DepositResponse,
  DepositTextRequest,
  Subject,
  SubjectSuggestionItem,
  DepositUrlRequest,
  DuplicateGroupGestureRequest,
  HealthResponse,
  LinkRequest,
  OperatorRetractRequest,
  OperatorSupersedeRequest,
  ParticleRelation,
  ParticleRetractEndpointResponse,
  ParticleSubjectAssignRequest,
  ReindexRequest,
  ReindexResponse,
  RelinkGatedRequest,
  RelinkReport,
  ResolutionAction,
  ReviewRequest,
} from "./schema.ts";
import { ApiError, type ConnectionFacts, errorForStatus, networkError } from "./errors.ts";

/** One request as the core states it: a method, a path with its query, a JSON body. */
export interface WireRequest {
  method: string;
  path: string;
  body?: unknown;
}

/** The answer: the status and the parsed JSON body, undefined when there is none. */
export interface WireResponse {
  status: number;
  body: unknown;
}

/**
 * Sends one request. Throws for a request that never got an answer; an
 * `ApiError` it throws (a client refusing to call with no token) passes
 * through unchanged.
 */
export type Transport = (req: WireRequest) => Promise<WireResponse>;

/** Curation queue controls: the `GET /curation` query parameters. */
export interface QueueOptions {
  limit?: number;
  kind?: string;
  semantic?: boolean;
}

export interface CurationClientOptions {
  transport: Transport;
  /** Read at call time, for the failure messages. */
  facts: () => ConnectionFacts;
  /** Recorded as `deposited_by` on a deposit: the client's own name. */
  depositedBy: string;
}

export class CurationClient {
  private readonly opts: CurationClientOptions;

  constructor(opts: CurationClientOptions) {
    this.opts = opts;
  }

  /** Send one request and map its answer (outcomes). */
  async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    let resp: WireResponse;
    try {
      resp = await this.opts.transport(body === undefined ? { method, path } : { method, path, body });
    } catch (e) {
      if (e instanceof ApiError) throw e;
      throw networkError(this.opts.facts(), e);
    }
    const failure = errorForStatus(resp.status, resp.body, this.opts.facts());
    if (failure) throw failure;
    return resp.body as T;
  }

  /** GET /health: liveness and the engine's version. */
  health(): Promise<HealthResponse> {
    return this.request<HealthResponse>("GET", "/health");
  }

  /** GET /curation: the leverage-ranked batch and its backlog's `open_count`. */
  curation(opts: QueueOptions = {}): Promise<CurationQueueResponse> {
    const params = new URLSearchParams();
    if (opts.limit !== undefined) params.set("limit", String(opts.limit));
    if (opts.kind) params.set("kind", opts.kind);
    if (opts.semantic !== undefined) params.set("semantic", String(opts.semantic));
    const qs = params.toString();
    return this.request<CurationQueueResponse>("GET", `/curation${qs ? `?${qs}` : ""}`);
  }

  /** GET /curation/cards/{key}: one open card, every belief briefed. */
  curationCard(key: string): Promise<CurationCard> {
    return this.request<CurationCard>("GET", `/curation/cards/${encodeURIComponent(key)}`);
  }

  /** Resolve an INCONSISTENCY: POST /review/{id}, the record's id, never a member's. */
  review(
    inconsistencyId: string,
    action: ResolutionAction,
    reviewerId: string,
    note?: string,
  ): Promise<unknown> {
    const req: ReviewRequest = {
      action,
      reviewer_id: reviewerId,
      ...(note !== undefined ? { note } : {}),
    };
    return this.request<unknown>("POST", `/review/${encodeURIComponent(inconsistencyId)}`, req);
  }

  /** merge: POST /links with a CO_EVIDENTIAL relation. */
  link(particleA: string, particleB: string): Promise<ParticleRelation> {
    const req: LinkRequest = {
      particle_a: particleA,
      particle_b: particleB,
      relation_type: "CO_EVIDENTIAL",
    };
    return this.request<ParticleRelation>("POST", "/links", req);
  }

  /** deposit pasted text: POST /corpus/deposit/text. */
  depositText(text: string): Promise<DepositResponse> {
    const req: DepositTextRequest = {
      text,
      source_type: "CONVERSATION",
      deposited_by: this.opts.depositedBy,
      tags: [],
    };
    return this.request<DepositResponse>("POST", "/corpus/deposit/text", req);
  }

  /** deposit a cited URL: POST /corpus/deposit/url (the engine fetches it). */
  depositUrl(url: string): Promise<DepositResponse> {
    const req: DepositUrlRequest = { url, deposited_by: this.opts.depositedBy, tags: [] };
    return this.request<DepositResponse>("POST", "/corpus/deposit/url", req);
  }

  /** edit: POST /particles/{id}/supersede, operator-scoped. */
  operatorSupersede(particleId: string, req: OperatorSupersedeRequest): Promise<AgentWriteResult> {
    return this.request<AgentWriteResult>(
      "POST",
      `/particles/${encodeURIComponent(particleId)}/supersede`,
      req,
    );
  }

  /** retract one belief: POST /particles/{id}/retract. */
  operatorRetract(particleId: string, reason: string): Promise<ParticleRetractEndpointResponse> {
    const req: OperatorRetractRequest = { reason };
    return this.request<ParticleRetractEndpointResponse>(
      "POST",
      `/particles/${encodeURIComponent(particleId)}/retract`,
      req,
    );
  }

  /** assign-subject: POST /particles/{id}/subjects. */
  assignSubject(particleId: string, req: ParticleSubjectAssignRequest): Promise<AgentWriteResult> {
    return this.request<AgentWriteResult>(
      "POST",
      `/particles/${encodeURIComponent(particleId)}/subjects`,
      req,
    );
  }

  /** The subjects whose name or alias matches `q`, closest first: GET /subjects/search, a read. */
  searchSubjects(q: string, limit: number): Promise<Subject[]> {
    const qs = new URLSearchParams({ q, limit: String(limit), order: "relevance" }).toString();
    return this.request<Subject[]>("GET", `/subjects/search?${qs}`);
  }

  /** The subjects a belief is most likely about, best first: GET /particles/{id}/subject-suggestions, a read. */
  subjectSuggestions(particleId: string): Promise<SubjectSuggestionItem[]> {
    return this.request<SubjectSuggestionItem[]>(
      "GET",
      `/particles/${encodeURIComponent(particleId)}/subject-suggestions`,
    );
  }

  /** affirm: POST /curation/affirm (BELIEF_AFFIRMED). */
  affirm(particleId: string, cardKey?: string): Promise<CurationEventResponse> {
    const req: CurationAffirmRequest = {
      particle_id: particleId,
      ...(cardKey !== undefined ? { card_key: cardKey } : {}),
    };
    return this.request<CurationEventResponse>("POST", "/curation/affirm", req);
  }

  /** snooze or dismiss a card: POST /curation/snooze; no window is permanent. */
  snoozeCard(cardKey: string, particleIds: string[], snoozeDays?: number): Promise<CurationEventResponse> {
    const req: CurationSnoozeRequest = {
      card_key: cardKey,
      particle_ids: particleIds,
      ...(snoozeDays !== undefined ? { snooze_days: snoozeDays } : {}),
    };
    return this.request<CurationEventResponse>("POST", "/curation/snooze", req);
  }

  /** merge, split or dismiss members of a duplicate group: POST /curation/duplicate-group. */
  duplicateGroup(req: DuplicateGroupGestureRequest): Promise<CurationGestureResponse> {
    return this.request<CurationGestureResponse>("POST", "/curation/duplicate-group", req);
  }

  /** retract a whole source: POST /corpus/{entry_id}/retract, dry run first. */
  retractCorpusEntry(entryId: string, reason: string, dryRun = false): Promise<CorpusRetractResponse> {
    const req: CorpusRetractRequest = { reason, dry_run: dryRun };
    return this.request<CorpusRetractResponse>(
      "POST",
      `/corpus/${encodeURIComponent(entryId)}/retract`,
      req,
    );
  }

  /** relink: POST /subjects/relink-gated, dry run first. */
  relinkGated(dryRun: boolean): Promise<RelinkReport> {
    const req: RelinkGatedRequest = { dry_run: dryRun };
    return this.request<RelinkReport>("POST", "/subjects/relink-gated", req);
  }

  /** reindex: POST /reindex over the FAILED snapshots. */
  reindex(): Promise<ReindexResponse> {
    const req: ReindexRequest = { include_failed: true, rate_limit_per_minute: 100 };
    return this.request<ReindexResponse>("POST", "/reindex", req);
  }

  /** dismiss an uncited URL: POST /corpus/links/dismiss. */
  dismissUrl(url: string, snoozeDays?: number): Promise<CorpusLinksDismissResponse> {
    const req: CorpusLinksDismissRequest = {
      url,
      ...(snoozeDays !== undefined ? { snooze_days: snoozeDays } : {}),
    };
    return this.request<CorpusLinksDismissResponse>("POST", "/corpus/links/dismiss", req);
  }
}
