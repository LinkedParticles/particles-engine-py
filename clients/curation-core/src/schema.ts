/*
 * The engine's request and response shapes the curation core uses, from the
 * file `openapi-typescript` generates out of `artifacts/openapi.json`.
 * Nothing here is hand-written.
 */
import type { components } from "./openapi";

type Schemas = components["schemas"];

export type CurationCard = Schemas["CurationCard"];
export type CurationQueueResponse = Schemas["CurationQueueResponse"];
export type ParticleBrief = Schemas["ParticleBrief"];
export type ObserverScopeBrief = Schemas["ObserverScopeBrief"];
export type ConflictBrief = Schemas["ConflictBrief"];
export type DuplicateVerdict = Schemas["DuplicateVerdict"];
export type DuplicateEdge = Schemas["DuplicateEdge"];
export type DuplicateGroupGestureRequest = Schemas["DuplicateGroupGestureRequest"];
export type CurationGestureResponse = Schemas["CurationGestureResponse"];
export type ResolutionAction = Schemas["ResolutionAction"];
export type ReviewRequest = Schemas["ReviewRequest"];
export type LinkRequest = Schemas["LinkRequest"];
export type ParticleRelation = Schemas["ParticleRelation"];
export type DepositTextRequest = Schemas["DepositTextRequest"];
export type DepositUrlRequest = Schemas["DepositUrlRequest"];
export type DepositResponse = Schemas["DepositResponse"];
export type AgentWriteResult = Schemas["AgentWriteResult"];
export type CorpusRetractRequest = Schemas["CorpusRetractRequest"];
export type CorpusRetractResponse = Schemas["CorpusRetractResponse"];
export type OperatorSupersedeRequest = Schemas["OperatorSupersedeRequest"];
export type OperatorRetractRequest = Schemas["OperatorRetractRequest"];
export type ParticleRetractEndpointResponse = Schemas["ParticleRetractEndpointResponse"];
export type ParticleSubjectAssignRequest = Schemas["ParticleSubjectAssignRequest"];
export type Subject = Schemas["Subject"];
export type SubjectSuggestionItem = Schemas["SubjectSuggestionItem"];
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
