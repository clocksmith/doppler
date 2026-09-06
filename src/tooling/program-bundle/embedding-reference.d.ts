import type { EmbeddingReferenceTranscript } from '../../config/embedding-reference.js';
export function buildEmbeddingReferenceTranscript(report: Record<string, any>, artifact: Record<string, any>, executionGraphHash: string): {
  artifact: Record<string, any>;
  transcript: EmbeddingReferenceTranscript;
  adapter: Record<string, unknown>;
};
