import type { PackEmbeddingContract } from './embedding-contract.js';

export const EMBEDDING_REFERENCE_TRANSCRIPT_SCHEMA_ID: 'doppler.embedding-reference-transcript/v1';
export interface EmbeddingObservation {
  input: { texts: string[] };
  embeddingContract: PackEmbeddingContract;
  outputs: Array<{ text: string; tokenIds: number[]; embedding: number[] }>;
}
export interface EmbeddingReference extends EmbeddingObservation {
  schema: 'doppler.embedding-source-reference/v1';
  source: { checkpointId: string; repository: string; revision: string; engine: string;
    files: Array<{ path: string; hash: `sha256:${string}` }>; [key: string]: unknown };
  tolerances: { embeddingMaxAbs: number; tokenIds: 'exact' };
}
export interface EmbeddingReferenceTranscript {
  schema: typeof EMBEDDING_REFERENCE_TRANSCRIPT_SCHEMA_ID; operation: 'embed';
  modelId: string; surface: string; manifestHash: `sha256:${string}`; executionGraphHash: `sha256:${string}`;
  source: { kind: string; path: string; hash: `sha256:${string}` };
  reference: EmbeddingReference; referenceDigest: `sha256:${string}`; observation: EmbeddingObservation;
}
export function assertEmbeddingReference(value: unknown): EmbeddingReference;
export function assertEmbeddingSourceIdentity(identity: unknown, reference: EmbeddingReference): void;
export function evaluateEmbeddingReference(reference: EmbeddingReference, observation: EmbeddingObservation): {
  passed: boolean; checks: Array<{ id: string; passed: boolean; [key: string]: unknown }>;
};
export function assertEmbeddingReferenceTranscript(value: unknown): EmbeddingReferenceTranscript;
