import type { Tensor } from '../../../gpu/tensor.js';
import type { VisionMergerWeights } from './spatial-merge.js';

export interface VisionEncoderConfig {
  depth: number;
  hiddenSize: number;
  intermediateSize: number;
  numHeads: number;
  headDim: number;
  outHiddenSize: number;
  spatialMergeSize: number;
  eps: number;
}

export interface VisionEncoderLayerWeights {
  norm1Weight: Tensor;
  norm1Bias?: Tensor | null;
  norm2Weight: Tensor;
  norm2Bias?: Tensor | null;
  qkvWeight: Tensor;
  qkvBias?: Tensor | null;
  projWeight: Tensor;
  projBias?: Tensor | null;
  fc1Weight: Tensor;
  fc1Bias?: Tensor | null;
  fc2Weight: Tensor;
  fc2Bias?: Tensor | null;
}

export interface VisionEncoderWeights extends VisionMergerWeights {
  layers?: VisionEncoderLayerWeights[];
  [tensorName: string]: Tensor | Tensor[] | null | undefined;
}

export declare function runVisionEncoder(params: {
  patchBuffer: GPUBuffer;
  numPatches: number;
  gridHeight: number;
  gridWidth: number;
  visionConfig: VisionEncoderConfig;
  weights: VisionEncoderWeights;
}): Promise<{ features: GPUBuffer; numTokens: number }>;
