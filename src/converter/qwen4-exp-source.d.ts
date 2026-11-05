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

/** Throw when authored contracts are missing or differ from the qwen4_exp source. */
export declare function assertConvertedQwen4ExpContracts(
  inference: Record<string, unknown> | null | undefined,
  rawConfig: Record<string, unknown> | null | undefined,
  modelId: string
): void;

export interface Qwen4ExpSourceBufferReceipt {
  tensor: string;
  dtype: 'I64';
  sha256: string;
  values: string[];
}

/**
 * Verify the checkpoint's I64 n-gram buffers against the authored contract and
 * record them in converterConfig.manifest.conversion.sourceBuffers.
 */
export declare function authenticateQwen4ExpSourceBuffers(
  tensors: ReadonlyArray<{ name: string; dtype: string }>,
  converterConfig: {
    inference?: { ngramEmbedding?: import('../config/schema/ngram-embedding.schema.js').NgramEmbeddingSchema | null } | null;
    manifest?: { conversion?: Record<string, unknown> | null } & Record<string, unknown>;
  },
  readTensor: (tensor: { name: string; dtype: string }) => Promise<Uint8Array | ArrayBuffer>
): Promise<Qwen4ExpSourceBufferReceipt[] | null>;
