/**
 * Expert router for multi-model execution.
 *
 * @module inference/expert-router
 */

export interface ExpertProfile {
  id: string;
  embedding?: number[];
  metadata?: Record<string, unknown>;
}

export declare class ExpertRouter {
  private experts;

  constructor();

  registerExpert(profile: ExpertProfile): void;

  removeExpert(id: string): void;

  listExperts(): ExpertProfile[];

  selectByEmbedding(embedding: number[], topK?: number): ExpertProfile[];

  private cosineSimilarity(a: number[], b: number[]): number;
}
