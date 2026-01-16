export declare function sumProfileTimings(timings: Record<string, number> | null): number | null;

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
