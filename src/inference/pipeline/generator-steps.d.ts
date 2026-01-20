export declare function sumProfileTimings(timings: Record<string, number> | null): number | null;

export interface BatchDecodeSelectionConfig {
  batchSize: number;
  useGPU: boolean;
  gpuSamplingAvailable: boolean;
  disableMultiTokenDecode: boolean;
  disableCommandBatching: boolean;
}

export declare function shouldUseBatchDecode(config: BatchDecodeSelectionConfig): boolean;

export declare function resolveBatchStop(
  tokens: number[],
  stopFlags: Uint32Array | null,
  stopTokenIds: number[],
  eosTokenId: number | undefined | null
): number;

export declare function decodeStep(
  state: unknown,
  currentIds: number[],
  opts: Record<string, unknown>,
  helpers: {
    buildLayerContext: (recorder: unknown, isDecode: boolean, debugLayers: unknown) => unknown;
    getLogitsWeights: () => unknown;
    getLogitsConfig: () => unknown;
    debugCheckBuffer: (buffer: GPUBuffer, label: string, numTokens: number, expectedDim?: number) => Promise<void>;
  }
): Promise<number>;

export declare function generateNTokensGPU(
  state: unknown,
  startToken: number,
  N: number,
  currentIds: number[],
  opts: Record<string, unknown>,
  helpers: {
    buildLayerContext: (recorder: unknown, isDecode: boolean, debugLayers: unknown) => unknown;
    getLogitsWeights: () => unknown;
    getLogitsConfig: () => unknown;
  }
): Promise<{ tokens: number[]; actualCount: number }>;
