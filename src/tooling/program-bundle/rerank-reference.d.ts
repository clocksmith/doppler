import type { RerankReferenceTranscript } from '../../config/rerank-reference.js';
export function buildRerankReferenceTranscript(report: any, artifact: any, executionGraphHash: string): {
  artifact: any; transcript: RerankReferenceTranscript; adapter: { source: string; surface: string; deviceInfo: unknown };
};
