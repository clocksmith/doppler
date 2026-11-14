import type {
  HyperConnectionsSchema,
  IndexedAttentionSchema,
} from '../config/schema/hyper-connection-decoder.schema.js';
import type { NgramEmbeddingSchema } from '../config/schema/ngram-embedding.schema.js';

export interface Qwen4ExpSourceContracts {
  hyperConnections: HyperConnectionsSchema;
  indexedAttention: IndexedAttentionSchema | null;
  ngramEmbedding: NgramEmbeddingSchema | null;
}

/** Whether the source config (top-level or text_config model_type) is qwen4_exp. */
export declare function isQwen4ExpSource(rawConfig: Record<string, unknown> | null | undefined): boolean;

/**
 * Lower a qwen4_exp source config into Doppler contracts, resolving omitted
 * fields from the pinned Qwen4ExpTextConfig defaults. Null for other families.
 */
export declare function resolveQwen4ExpSourceContracts(
  rawConfig: Record<string, unknown> | null | undefined
): Qwen4ExpSourceContracts | null;

