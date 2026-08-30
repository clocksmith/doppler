export interface ImageTranscriptionInput {
  imageBytes: Uint8Array | Uint8ClampedArray | Float32Array;
  width: number;
  height: number;
  prompt?: string;
  maxTokens?: number;
  softTokenBudget?: number;
  signal?: AbortSignal;
}

export declare function createImageTranscriptionResourceScope(
  pipeline: Record<string, unknown>,
  encodeResult: { features?: GPUBuffer | null },
  releaseFeature?: (buffer: GPUBuffer) => void
): {
  setGlmOcrRopeOverride(override: {
    cos: GPUBuffer;
    sin: GPUBuffer;
    release: () => void;
  }): void;
  runGeneration<T>(task: () => Promise<T> | T): Promise<T>;
  run<T>(task: () => Promise<T> | T): Promise<T>;
};

export declare function transcribeImage(
  this: Record<string, unknown>,
  input: ImageTranscriptionInput
): Promise<{ text: string; tokens: number[] }>;
