export const RERANK_REFERENCE_TRANSCRIPT_SCHEMA_ID: 'doppler.rerank-reference-transcript/v1';
export interface RerankReferenceRow {
  index: number; document: string; tokenIds: number[];
  trueLogit: number; falseLogit: number; score: number; probability: number;
}
export interface RerankObservation {
  input: { query: string; documents: string[] };
  scoringConfig: Record<string, unknown>;
  outputs: RerankReferenceRow[];
}
export interface RerankReference extends RerankObservation {
  schema: 'doppler.rerank-source-reference/v1';
  source: { checkpointId: string; repository: string; revision: string; engine: string;
    files: Array<{ path: string; hash: `sha256:${string}` }>; [key: string]: unknown };
  tolerances: { logitMaxAbs: number; scoreMaxAbs: number; probabilityMaxAbs: number; ranking: 'exact' };
}
export interface RerankReferenceTranscript {
  schema: typeof RERANK_REFERENCE_TRANSCRIPT_SCHEMA_ID; operation: 'rerank';
  modelId: string; surface: string; manifestHash: `sha256:${string}`; executionGraphHash: `sha256:${string}`;
  source: { kind: string; path: string; hash: `sha256:${string}` };
  reference: RerankReference; referenceDigest: `sha256:${string}`; observation: RerankObservation;
}
export function assertRerankReference(value: unknown): RerankReference;
export function assertRerankSourceIdentity(identity: unknown, reference: RerankReference): void;
export function evaluateRerankReference(reference: RerankReference, observation: RerankObservation): {
  passed: boolean; checks: Array<{ id: string; passed: boolean; [key: string]: unknown }>;
};
export function assertRerankReferenceTranscript(value: unknown): RerankReferenceTranscript;
