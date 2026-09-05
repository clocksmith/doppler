import type { DopplerRerankScore } from './model-session.js';
export function collectModelRerankScores(pipeline: any, query: string, documents: string[], options: { benchmark?: boolean }): Promise<{
  normalizedQuery: string; normalizedDocuments: string[]; scores: DopplerRerankScore[];
  ranking: Array<DopplerRerankScore & { rank: number }>;
}>;
