/** Supported hashed n-gram embedding semantics. */
export type NgramEmbeddingSemantics = 'qwen4_exp_ple';
/** Supported stored dtypes of the n-gram embedding table. */
export type NgramEmbeddingTableDtype = 'bf16';
export type NgramRowCacheEviction = 'lru';

export declare const NGRAM_EMBEDDING_SEMANTICS: readonly NgramEmbeddingSemantics[];
export declare const NGRAM_EMBEDDING_TABLE_DTYPES: readonly NgramEmbeddingTableDtype[];
export declare const NGRAM_ROW_CACHE_EVICTIONS: readonly NgramRowCacheEviction[];

/** One PLE decoder layer as authored in the manifest. */
export interface NgramEmbeddingLayerSchema {
  /** One-based source decoder layer id (qwen4_exp `ple_layer_ids`). */
  sourceLayerId: number;
  /** Zero-based decoder index; must equal sourceLayerId - 1. */
  decoderLayerIndex: number;
  /** Position in sorted source `ple_layer_ids`; selects primes and multipliers. */
  pleLayerIndex: number;
  /** Hash multipliers as decimal strings (values exceed 2^53). */
  multipliers: string[];
  headVocabSizes: number[];
  headOffsets: number[];
  unpaddedRows: number;
  rows: number;
}

/** Manifest `inference.ngramEmbedding`: null for models without hashed n-gram PLE. */
export interface NgramEmbeddingSchema {
  semantics: NgramEmbeddingSemantics;
  unigramVocabSize: number;
  ngramSize: number;
  headsPerNgram: number;
  /** (ngramSize - 1) * headsPerNgram. */
  headCount: number;
  embedDim: number;
  /** embedDim / headCount: width of one table row. */
  headDim: number;
  vocabSizeBase: number;
  vocabPadMultiple: number;
  /** Source seed (qwen4_exp default 1234 when the source config omits it). */
  seed: number;
  /** N-gram segment boundary and padding token; not a generation stop token list. */
  boundaryTokenId: number;
  convKernelSize: number;
  /** Equals ngramSize. */
  convDilation: number;
  /** (convKernelSize - 1) * convDilation. */
  convStateLength: number;
  normEps: number;
  tableDtype: NgramEmbeddingTableDtype;
  layers: NgramEmbeddingLayerSchema[];
}

export interface ResolvedNgramEmbeddingLayer {
  readonly sourceLayerId: number;
  readonly decoderLayerIndex: number;
  readonly pleLayerIndex: number;
  readonly multipliers: readonly bigint[];
  readonly headVocabSizes: readonly number[];
  readonly headOffsets: readonly number[];
  readonly unpaddedRows: number;
  readonly rows: number;
}

export interface ResolvedNgramEmbedding extends Readonly<Omit<NgramEmbeddingSchema, 'layers'>> {
  readonly layers: readonly ResolvedNgramEmbeddingLayer[];
}

/** Session policy `inference.session.ngramEmbedding`. */
export interface NgramEmbeddingSessionSchema {
  rowCache: { maxRows: number; maxBytes: number; eviction: NgramRowCacheEviction };
  /** Host staging rows per table read call; at least one token's headCount rows. */
  staging: { maxRows: number };
  reads: { maxConcurrent: number };
}

export interface ResolvedNgramEmbeddingSession {
  readonly rowCache: Readonly<NgramEmbeddingSessionSchema['rowCache']>;
  readonly staging: Readonly<NgramEmbeddingSessionSchema['staging']>;
  readonly reads: Readonly<NgramEmbeddingSessionSchema['reads']>;
}

export interface NgramEmbeddingGeometry {
  hiddenSize: number;
  numLayers: number;
  vocabSize: number;
  layerTypes?: readonly string[] | null;
}

export declare function deriveNgramMultipliers(
  unigramVocabSize: number,
  ngramSize: number,
  pleLayerIndex: number,
  seed: number
): bigint[];

export declare function deriveNgramLayerTable(options: {
  vocabSizeBase: number;
  vocabPadMultiple: number;
  headCount: number;
  pleLayerIndex: number;
}): { headVocabSizes: number[]; headOffsets: number[]; unpaddedRows: number; rows: number };

export declare function resolveNgramEmbeddingContract(
  contract: NgramEmbeddingSchema | null | undefined,
  geometry: NgramEmbeddingGeometry,
  label?: string
): ResolvedNgramEmbedding | null;

export declare function serializeNgramEmbeddingContract(resolved: ResolvedNgramEmbedding): NgramEmbeddingSchema;

export declare function resolveNgramEmbeddingSession(
  session: NgramEmbeddingSessionSchema | null | undefined,
  contract: ResolvedNgramEmbedding | null,
  label?: string
): ResolvedNgramEmbeddingSession | null;
