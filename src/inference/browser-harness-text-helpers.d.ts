export declare function resolvePrompt(runtimeConfig: Record<string, unknown>): string;
export declare function getDefaultEmbeddingSemanticFixtures(): {
  retrievalCases: Array<Record<string, unknown>>;
  pairCases: Array<Record<string, unknown>>;
  minRetrievalTop1Acc: number;
  minPairAcc: number;
  pairMargin: number;
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
export declare function runEmbeddingSemanticChecks(
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
