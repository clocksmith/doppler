export const SEQUENCE_REFERENCE_TRANSCRIPT_SCHEMA_ID: 'doppler.sequence-reference-transcript/v1';
export interface SequenceReferenceTranscript {
  schema: typeof SEQUENCE_REFERENCE_TRANSCRIPT_SCHEMA_ID;
  operation: 'encodeSequence';
  modelId: string;
  surface: string;
  executionGraphHash: `sha256:${string}`;
  manifestHash: `sha256:${string}`;
  source: { kind: string; path: string; hash: `sha256:${string}` };
  reference: {
    digest: `sha256:${string}`;
    path: string;
    source: { checkpointId: string; repository: string; revision: string; [key: string]: unknown };
    input: { sequence: string; alphabet: string; tokenIds: number[] };
    tolerances: { pooledEmbeddingMaxAbs: number; tokenEmbeddingMaxAbs: number };
  };
  options: { includeTokenEmbeddings: true; includeLogits: false };
  output: {
    embeddingDim: number;
    tokenCount: number;
    digests: { pooledEmbedding: `sha256:${string}`; tokenEmbeddings: `sha256:${string}`; logits: null };
  };
  checks: Array<{ id: string; passed: boolean; [key: string]: unknown }>;
}
export function validateSequenceReferenceTranscript(value: unknown): { ok: boolean; errors: string[] };
export function assertSequenceReferenceTranscript(value: unknown): SequenceReferenceTranscript;
