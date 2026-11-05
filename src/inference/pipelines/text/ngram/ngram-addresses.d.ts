import type {
  ResolvedNgramEmbedding,
  ResolvedNgramEmbeddingLayer,
} from '../../../../config/schema/ngram-embedding.schema.js';

export interface NgramHistoryCheckpoint {
  readonly epoch: number;
  readonly length: number;
}

/**
 * Per-sequence n-gram token history of one PLE layer. Appends return exact
 * table row addresses, [count, headCount], in head order (bigram heads first).
 */
export declare class NgramTokenHistory {
  constructor(contract: ResolvedNgramEmbedding, layer: ResolvedNgramEmbeddingLayer, initialCapacity?: number);
  readonly contract: ResolvedNgramEmbedding;
  readonly layer: ResolvedNgramEmbeddingLayer;
  /** ngramSize - 1 leading boundary tokens of context. */
  readonly context: number;
  readonly length: number;
  readonly epoch: number;
  append(tokens: ArrayLike<number>): Uint32Array;
  /** Shorten the history; invalidates earlier checkpoints. */
  truncate(length: number): void;
  reset(): void;
  checkpoint(): NgramHistoryCheckpoint;
  /** Restore a checkpoint taken since the last truncate, reset or restore. */
  restore(checkpoint: NgramHistoryCheckpoint): void;
}
