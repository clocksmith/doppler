export declare function sumProfileTimings(timings: Record<string, number> | null): number | null;

/**
 * Error thrown when decode readback produces non-finite values.
 * Callers can branch on the .name === 'FinitenessError' or use
 * instanceof.
 */
export declare class FinitenessError extends Error {
  constructor(message: string);
}

/**
 * Advance one decode step using a pre-resolved token id and its
 * embedding. Mutates `state` to reflect the new sequence length and
 * decode-step counter.
 */
export declare function advanceWithTokenAndEmbedding(
  state: Record<string, unknown>,
  tokenId: number,
  opts: Record<string, unknown>,
  helpers: Record<string, unknown>,
  embeddingMode: string
): Promise<void>;

export interface BatchDecodeSelectionConfig {
  batchSize: number;
  useGPU: boolean;
  gpuSamplingAvailable: boolean;
  disableMultiTokenDecode: boolean;
  disableCommandBatching: boolean;
  isBdpaPagedLayout?: boolean;
  finitenessFallbackWindowOpen?: boolean;
}

export declare function shouldUseBatchDecode(config: BatchDecodeSelectionConfig): boolean;

export interface FusedDecodeSamplingConfig {
  recorderEnabled: boolean;
  gpuSamplingEnabled: boolean;
  fusedDecodeDisabled: boolean;
  layerTypes?: string[] | null;
}

export declare function shouldUseFusedDecodeSampling(config: FusedDecodeSamplingConfig): boolean;

export interface StopTokenLookup {
  firstTokenId: number | null;
  secondTokenId: number | null;
  tokenSet: Set<number> | null;
}

export declare function createStopTokenLookup(
  stopTokenIds: number[],
  eosTokenId?: number | null
): StopTokenLookup;

export declare function resolveBatchStop(
  tokens: ArrayLike<number>,
  stopFlags: ArrayLike<number> | null,
  stopTokenLookup: StopTokenLookup
): number;

export declare function findInvalidGeneratedToken(
  tokens: ArrayLike<number>,
  vocabSize: number,
  padTokenId?: number | null
): { index: number; tokenId: number } | null;

export interface SampledTokenStagingBuffer {
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBufferLike;
  unmap(): void;
  destroy(): void;
}

export interface DeferredCleanupRecorder {
  completeDeferredCleanup(options?: { discardPooled?: boolean }): Promise<void> | void;
}

export declare function readSampledTokenFromStagingBuffer(
  stagingBuffer: SampledTokenStagingBuffer,
  options?: {
    ownsStagingBuffer?: boolean;
    hasFinitenessBuffer?: boolean;
    ring?: { advance(): void } | null;
    cleanupRecorder?: DeferredCleanupRecorder | null;
  }
): Promise<{
  nextToken: number;
  finitenessStatus: {
    triggered: boolean;
    metadata: string;
  };
  timing: {
    mapWaitMs: number;
    cleanupMs: number;
    copyMs: number;
  };
}>;

export declare function readMappedBufferCopy(
  stagingBuffer: SampledTokenStagingBuffer,
  options?: {
    ownsStagingBuffer?: boolean;
  }
): Promise<ArrayBuffer>;

export declare function readBatchTokensFromStagingBuffers(options: {
  tokensStagingBuffer: SampledTokenStagingBuffer;
  stopStagingBuffer?: SampledTokenStagingBuffer | null;
  finitenessStagingBuffer?: SampledTokenStagingBuffer | null;
  finitenessOffsetBytes?: number | null;
  tokenCount: number;
  ownsTokensStaging?: boolean;
  ownsStopStaging?: boolean;
  ownsFinitenessStaging?: boolean;
  ring?: { advance(): void } | null;
  cleanupRecorder?: DeferredCleanupRecorder | null;
}): Promise<{
  tokens: Uint32Array;
  stopFlags: Uint32Array | null;
  finitenessStatus: {
    triggered: boolean;
    metadata: string;
  };
  timing: {
    mapWaitMs: number;
    cleanupMs: number;
    copyMs: number;
  };
}>;

export declare function decodeStep(
  state: unknown,
  currentIds: number[],
  opts: Record<string, unknown>,
  helpers: {
    buildLayerContext: (recorder: unknown, isDecode: boolean, debugLayers: unknown, executionPlan?: unknown) => unknown;
    getLogitsWeights: () => unknown;
    getLogitsConfig: () => unknown;
    debugCheckBuffer?: (buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>;
  }
): Promise<number>;

export declare function decodeStepLogits(
  state: unknown,
  currentIds: number[],
  opts: Record<string, unknown>,
  helpers: {
    buildLayerContext: (recorder: unknown, isDecode: boolean, debugLayers: unknown, executionPlan?: unknown) => unknown;
    getLogitsWeights: () => unknown;
    getLogitsConfig: () => unknown;
    debugCheckBuffer?: (buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>;
  }
): Promise<{
  logits: Float32Array;
  logitsBuffer: GPUBuffer | null;
  logitsDtype: string | null;
  rawVocabSize: number;
  vocabSize: number;
}>;

export declare function advanceWithToken(
  state: unknown,
  tokenId: number,
  opts: Record<string, unknown>,
  helpers: {
    buildLayerContext: (recorder: unknown, isDecode: boolean, debugLayers: unknown, executionPlan?: unknown) => unknown;
    getLogitsWeights: () => unknown;
    getLogitsConfig: () => unknown;
    debugCheckBuffer?: (buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>;
  }
): Promise<void>;

export declare function generateNTokensGPU(
  state: unknown,
  startToken: number,
  N: number,
  currentIds: number[],
  opts: Record<string, unknown>,
  helpers: {
    buildLayerContext: (recorder: unknown, isDecode: boolean, debugLayers: unknown, executionPlan?: unknown) => unknown;
    getLogitsWeights: () => unknown;
    getLogitsConfig: () => unknown;
  }
): Promise<{ tokens: number[] | Uint32Array; actualCount: number }>;
