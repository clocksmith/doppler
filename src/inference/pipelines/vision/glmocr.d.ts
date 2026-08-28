export interface GlmOcrVisionConfig {
  patchSize: number;
  spatialMergeSize: number;
  temporalPatchSize: number;
  minPixels: number;
  maxPixels: number;
  inChannels: number;
  normalization: {
    mean: [number, number, number];
    std: [number, number, number];
  };
  depth: number;
  hiddenSize: number;
  intermediateSize: number;
  numHeads: number;
  headDim: number;
  outHiddenSize: number;
  eps: number;
  ropeTheta: number;
  mergerIntermediateSize: number;
  downsampleKernelSize: number;
}

export interface GlmOcrPreprocessResult {
  patches: Float32Array;
  patchDim: number;
  numPatches: number;
  gridHeight: number;
  gridWidth: number;
  targetWidth: number;
  targetHeight: number;
}

export interface GlmOcrEncodeResult {
  features: GPUBuffer;
  numTokens: number;
  gridThw: [number, number, number];
  imageWidth: number;
  imageHeight: number;
}

export type GlmOcrVisionProbe = (
  stage: string,
  buffer: GPUBuffer | Float32Array,
  options: {
    layerIdx?: number;
    numTokens: number;
    hiddenSize: number;
    dtype: 'f32';
  }
) => Promise<void>;

export declare function resolveGlmOcrImageSize(
  width: number,
  height: number,
  visionConfig: GlmOcrVisionConfig
): { targetWidth: number; targetHeight: number };

export declare function resizeGlmOcrImageBicubic(
  pixels: Uint8Array | Uint8ClampedArray | Float32Array,
  width: number,
  height: number,
  targetWidth: number,
  targetHeight: number
): Uint8Array;

export declare function preprocessGlmOcrImage(
  pixels: Uint8Array | Uint8ClampedArray | Float32Array,
  width: number,
  height: number,
  visionConfig: GlmOcrVisionConfig
): GlmOcrPreprocessResult;

export declare function encodeGlmOcrImage(params: {
  pixels: Uint8Array | Uint8ClampedArray | Float32Array;
  width: number;
  height: number;
  visionConfig: GlmOcrVisionConfig;
  weights: Record<string, unknown>;
  softTokenBudget?: number;
  probeTensor?: GlmOcrVisionProbe;
}): Promise<GlmOcrEncodeResult>;
