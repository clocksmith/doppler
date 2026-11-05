import type {
  NgramEmbeddingSchema,
  NgramEmbeddingSessionSchema,
  ResolvedNgramEmbedding,
  ResolvedNgramEmbeddingSession,
} from './ngram-embedding.schema.js';

export type HyperConnectionSemantics = 'qwen4_exp_gated_residual';
export type IndexedAttentionSemantics = 'qwen4_exp_qsa';
export type IndexedAttentionTieOrder = 'lower_block_index';

export declare const HYPER_CONNECTION_SEMANTICS: readonly HyperConnectionSemantics[];
export declare const INDEXED_ATTENTION_SEMANTICS: readonly IndexedAttentionSemantics[];
export declare const INDEXED_ATTENTION_TIE_ORDERS: readonly IndexedAttentionTieOrder[];

/** Manifest `inference.hyperConnections`: null for single-stream residual decoders. */
export interface HyperConnectionsSchema {
  semantics: HyperConnectionSemantics;
  /** Residual streams (qwen4_exp `hc_count`). */
  streamCount: number;
  /** Width of one stream; equals architecture.hiddenSize. */
  streamWidth: number;
  /** Low-rank input mixer rank (qwen4_exp `hc_lowrank`). */
  projectionRank: number;
  normEps: number;
}

/** Manifest `inference.indexedAttention`: null for dense full attention. */
export interface IndexedAttentionSchema {
  semantics: IndexedAttentionSemantics;
  /** Zero-based full-attention layers that run the indexer (all of them for qwen4_exp). */
  layers: number[];
  queryHeads: number;
  /** Exactly 1 for qwen4_exp. */
  keyHeads: number;
  headDim: number;
  /** Rotary dimensions shared with full attention. */
  rotaryDim: number;
  tokenBudget: number;
  compressRatio: number;
  /** tokenBudget / compressRatio. */
  blockTopK: number;
  /** tokenBudget + compressRatio - 1: visible-token count up to which every visible token is selected. */
  denseVisibleLimit: number;
  normEps: number;
  /** Doppler-declared order among equal block scores. */
  tieOrder: IndexedAttentionTieOrder;
  /** Always false: the reference does not force a query's own token. */
  forceSelfToken: false;
}

export type ResolvedHyperConnections = Readonly<HyperConnectionsSchema>;
export interface ResolvedIndexedAttention extends Readonly<Omit<IndexedAttentionSchema, 'layers'>> {
  readonly layers: readonly number[];
}

export interface HyperConnectionDecoderGeometry {
  hiddenSize: number;
  numLayers: number;
  vocabSize: number;
  layerTypes?: readonly string[] | null;
  /** Rotary dimensions of full attention (headDim * partialRotaryFactor), when known. */
  attentionRotaryDim?: number | null;
  modelId?: string;
  /** Manifest tensor names, used to reject undeclared mechanism tensors. */
  tensorNames?: readonly string[] | null;
}

export interface HyperConnectionDecoderContracts {
  hyperConnections: ResolvedHyperConnections | null;
  indexedAttention: ResolvedIndexedAttention | null;
  ngramEmbedding: ResolvedNgramEmbedding | null;
  ngramEmbeddingSession: ResolvedNgramEmbeddingSession | null;
}

export declare function resolveHyperConnectionsContract(
  contract: HyperConnectionsSchema | null | undefined,
  geometry: { hiddenSize: number },
  label?: string
): ResolvedHyperConnections | null;

export declare function resolveIndexedAttentionContract(
  contract: IndexedAttentionSchema | null | undefined,
  geometry: { numLayers: number; layerTypes?: readonly string[] | null; attentionRotaryDim?: number | null },
  label?: string
): ResolvedIndexedAttention | null;

export declare function resolveHyperConnectionDecoderContracts(
  inference: {
    hyperConnections?: HyperConnectionsSchema | null;
    indexedAttention?: IndexedAttentionSchema | null;
    ngramEmbedding?: NgramEmbeddingSchema | null;
    session?: { ngramEmbedding?: NgramEmbeddingSessionSchema | null } | null;
    normalization?: { postAttentionNorm?: boolean; preFeedforwardNorm?: boolean; postFeedforwardNorm?: boolean };
    layerPattern?: { residualBranchScale?: number };
  },
  geometry: HyperConnectionDecoderGeometry
): HyperConnectionDecoderContracts;
