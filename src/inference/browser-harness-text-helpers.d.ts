export declare function resolvePrompt(runtimeConfig: Record<string, unknown>): string;
export declare function getDefaultEmbeddingSemanticFixtures(): {
  retrievalCases: Array<Record<string, unknown>>;
  pairCases: Array<Record<string, unknown>>;
  minRetrievalTop1Acc: number;
  minPairAcc: number;
  pairMargin: number;
};
export declare function getDefaultRerankSemanticFixtures(): {
  cases: Array<Record<string, unknown>>;
  minPairAcc: number;
  minScoreMargin: number;
};
export declare function resolveBenchmarkRunSettings(
  runtimeConfig: Record<string, unknown>,
  source?: Record<string, unknown> | null
): {
  warmupRuns: number;
  timedRuns: number;
  prompt: string | Record<string, unknown>;
  promptLabel: string;
  maxTokens: number;
  sampling: Record<string, unknown>;
  seed?: number;
};
export declare function normalizeDecodeRecordOpLabels(
  value: unknown
): Record<string, number> | null;
export declare function buildDecodeRecordTopOps(
  labelCounts: unknown,
  totalOps?: number | null,
  limit?: number
): Array<{
  label: string;
  count: number;
  shareOfOps: number | null;
}>;
export declare function groupDecodeRecordOpLabels(
  labelCounts: unknown
): Record<string, number> | null;
export declare function buildDecodeRecordTopOpGroups(
  labelCounts: unknown,
  totalOps?: number | null,
  limit?: number
): Array<{
  label: string;
  count: number;
  shareOfOps: number | null;
}>;
export declare function normalizeUniformCacheStats(value: unknown): {
  hits?: number;
  misses?: number;
  totalLookups?: number;
  hitRateRatio?: number;
  hitRate?: string;
  evictions?: number;
  currentSize?: number;
  pendingDestruction?: number;
} | null;
export declare function runEmbeddingSemanticChecks(
  pipeline: Record<string, unknown>,
  options?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;
export declare function resolveRerankScoringConfig(
  pipeline: Record<string, unknown>
): Record<string, unknown>;
export declare function formatRerankPrompt(
  query: string,
  document: string,
  scoringConfig: Record<string, unknown>
): string;
export declare function scoreRerankDocument(
  pipeline: Record<string, unknown>,
  query: string,
  document: string,
  scoringConfig?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;
export declare function runRerank(
  pipeline: Record<string, unknown>,
  runtimeConfig: Record<string, unknown>,
  runOverrides?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;
export declare function runRerankSemanticChecks(
  pipeline: Record<string, unknown>,
  options?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;
export declare function isCoherentOutput(tokens: Array<unknown>, output: unknown): boolean;

export interface ReferenceLogitsDigest {
  index: number | null;
  tokenId: number | null;
  inputTokenCount: number | null;
  dtype: 'f32';
  elementCount: number;
  digest: string;
  top?: Array<{
    tokenId: number;
    logit: number;
    text: string | null;
  }>;
}

export interface KvCacheLayerByteProof {
  layer: number;
  seqLen: number;
  keyBytes: number;
  valueBytes: number;
  keyDigest: string;
  valueDigest: string;
}

export interface KvCacheByteProof {
  mode: 'sha256-layer-kv-bytes';
  layout: string;
  kvDtype: string | null;
  layerCount: number;
  digest: string;
  layers: KvCacheLayerByteProof[];
}

export declare function digestLogitsForTranscript(
  logits: Float32Array,
  context?: Record<string, unknown> | null
): ReferenceLogitsDigest;

export declare function captureKvCacheByteProof(
  pipeline: Record<string, unknown>,
  enabled: boolean
): Promise<KvCacheByteProof | null>;

export declare function runGeneration(
  pipeline: Record<string, unknown>,
  runtimeConfig: Record<string, unknown>,
  runOverrides?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;
export declare function runEmbedding(
  pipeline: Record<string, unknown>,
  runtimeConfig: Record<string, unknown>,
  runOverrides?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;

export declare function runImageTranscription(
  pipeline: Record<string, unknown>,
  runtimeConfig: Record<string, unknown>,
  runOverrides?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;

export declare function runTextInference(
  pipeline: Record<string, unknown>,
  runtimeConfig: Record<string, unknown>,
  runOverrides?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;
