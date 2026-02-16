import type { Tensor } from '../../../gpu/tensor.js';
import type { WeightBuffer } from '../../../gpu/weight-buffer.js';
import type { DiffusionRuntimeConfig } from './types.js';

export interface DiffusionGpuScaffoldConfig {
  tokens: number;
  hiddenSize: number;
  numHeads: number;
  seed?: number;
  layerNormEps: number;
}

export interface DiffusionGpuScaffoldState {
  device: GPUDevice;
  input: Tensor;
  tokens: number;
  hiddenSize: number;
  numHeads: number;
  eps: number;
  weights: {
    q: WeightBuffer;
    k: WeightBuffer;
    v: WeightBuffer;
    out: WeightBuffer;
    ffnUp: WeightBuffer;
    ffnDown: WeightBuffer;
    lnWeight: GPUBuffer;
    lnBias: GPUBuffer;
  };
}

export interface DiffusionAttentionWeights {
  q: WeightBuffer;
  k: WeightBuffer;
  v: WeightBuffer;
  out: WeightBuffer;
}

export interface DiffusionFFNWeights {
  ffnUp: WeightBuffer;
  ffnDown: WeightBuffer;
}

export interface DiffusionGpuScaffoldStepOptions {
  stepIndex?: number;
}

export interface DiffusionLinearOptions {
  tokens: number;
  outDim: number;
  inDim: number;
  outputDtype?: 'f16' | 'f32';
}

export interface DiffusionAttentionOptions {
  tokens: number;
  hiddenSize: number;
}

export interface DiffusionFFNOptions {
  tokens: number;
  hiddenSize: number;
}

export declare function initializeDiffusionGpuScaffold(
  runtime: DiffusionRuntimeConfig
): DiffusionGpuScaffoldState;

export declare function runLinear(
  input: Tensor,
  weight: WeightBuffer,
  bias: GPUBuffer | null,
  options: DiffusionLinearOptions
): Promise<Tensor>;

export declare function runAttentionScaffold(
  input: Tensor,
  weights: DiffusionAttentionWeights,
  options: DiffusionAttentionOptions
): Promise<Tensor>;

export declare function runFeedForwardScaffold(
  input: Tensor,
  weights: DiffusionFFNWeights,
  options: DiffusionFFNOptions
): Promise<Tensor>;

export declare function runDiffusionGpuScaffold(
  scaffold: DiffusionGpuScaffoldState,
  options?: DiffusionGpuScaffoldStepOptions
): Promise<Tensor | null>;

export declare function logDiffusionGpuScaffold(
  scaffold: DiffusionGpuScaffoldState | null
): void;
