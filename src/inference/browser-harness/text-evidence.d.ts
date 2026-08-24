export declare function normalizeDecodeRecordOpLabels(value: unknown): Record<string, number> | null;
export declare function buildDecodeRecordTopOps(
  labelCounts: unknown,
  totalOps?: number | null,
  limit?: number
): Array<{ label: string; count: number; shareOfOps: number | null }>;
export declare function groupDecodeRecordOpLabels(value: unknown): Record<string, number> | null;
export declare function buildDecodeRecordTopOpGroups(
  labelCounts: unknown,
  totalOps?: number | null,
  limit?: number
): Array<{ label: string; count: number; shareOfOps: number | null }>;
export declare function normalizeUniformCacheStats(value: unknown): Record<string, number | string> | null;
export declare function normalizeTokenIdArray(value: unknown, label: string): number[];
export declare function resolveGenerationUseChatTemplate(
  pipeline: Record<string, unknown>,
  runtimeConfig: Record<string, unknown>,
  runOverrides: Record<string, unknown> | null,
  promptInput: unknown
): boolean;
export declare function resolvePromptTokenIdsForTranscript(
  pipeline: Record<string, unknown>,
  promptInput: unknown,
  useChatTemplate: boolean
): number[] | null;
export declare function shouldCaptureReferenceLogits(
  runOverrides: Record<string, unknown> | null,
  runtimeConfig: Record<string, unknown>
): boolean;
export declare function shouldCaptureReferenceKvBytes(
  runOverrides: Record<string, unknown> | null,
  runtimeConfig: Record<string, unknown>
): boolean;
export declare function shouldEnableReferenceTranscriptDiagnostics(
  runOverrides: Record<string, unknown> | null,
  runtimeConfig: Record<string, unknown>
): boolean;
export interface ReferenceLogitsDigest {
  index: number | null;
  tokenId: number | null;
  inputTokenCount: number | null;
  dtype: 'f32';
  elementCount: number;
  digest: string;
  top?: Array<{ tokenId: number; logit: number; text: string | null }>;
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
export declare function summarizeRerankScores(scores: unknown[]): Record<string, unknown>;
export declare function summarizeEmbeddingValues(embedding: unknown): Record<string, unknown>;
export declare function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number;
export declare function top1Index(values: ArrayLike<number>): number;
export declare function summarizeGenerationTokens(tokenRecords: unknown[]): Record<string, unknown>;
export declare function buildGenerationPhaseFromStats(
  pipeline: Record<string, unknown>,
  durationMs: number,
  tokenCount: number
): Record<string, unknown>;
export declare function isCoherentOutput(tokens: Array<unknown>, output: unknown): boolean;
