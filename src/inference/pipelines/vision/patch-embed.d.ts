import type { Tensor } from '../../../gpu/tensor.js';

export interface VisionPatchEmbedConfig {
  patchSize: number;
  hiddenSize: number;
  temporalPatchSize: number;
}

export interface VisionPatchEmbedWeights {
  patchProjWeight?: Tensor;
  patchProjBias?: Tensor | null;
  'visual.patch_embed.proj.weight'?: Tensor;
  'visual.patch_embed.proj.bias'?: Tensor | null;
}

export interface VisionPatchEmbedResult {
  patchBuffer: GPUBuffer;
  numPatches: number;
}

export declare function patchEmbed(params: {
  imageData: Float32Array;
  height: number;
  width: number;
  channels: number;
  gridHeight: number;
  gridWidth: number;
  visionConfig: VisionPatchEmbedConfig;
  weights: VisionPatchEmbedWeights;
}): Promise<VisionPatchEmbedResult>;
