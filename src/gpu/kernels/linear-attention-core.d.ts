import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';

export interface LinearAttentionCoreLayerState {
  convDim: number;
  convKernelSize: number;
  valueDim: number;
  numVHeads: number;
  numKHeads: number;
  headKDim: number;
  headVDim: number;
  qSize: number;
  kSize: number;
  qRep: number;
  normMode: 'shared' | 'per_head';
  rmsNormEps: number;
  convWeightGPU: GPUBuffer;
  dtBiasGPU: GPUBuffer;
  aNegExpGPU: GPUBuffer;
  normWeightGPU: GPUBuffer;
  convStateGPU: GPUBuffer;
  recurrentStateGPU: GPUBuffer;
}

export interface RunLinearAttentionCoreGPUOptions {
  numTokens: number;
  layerIdx?: number;
  qkL2NormEps?: number;
  recorder?: CommandRecorder | null;
  outputDtype?: 'f16' | 'f32';
}

export declare function runLinearAttentionCoreGPU(
  qkvTensor: Tensor,
  zTensor: Tensor,
  aTensor: Tensor,
  bTensor: Tensor,
  layerState: LinearAttentionCoreLayerState,
  options: RunLinearAttentionCoreGPUOptions
): Promise<Tensor>;
