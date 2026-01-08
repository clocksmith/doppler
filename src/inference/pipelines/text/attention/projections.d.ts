import type { CommandRecorder } from '../../../../gpu/kernel-selector.js';
import type { Tensor } from '../../../../gpu/tensor.js';
import type { WeightBuffer, CpuWeightBuffer } from '../../../../gpu/weight-buffer.js';
import type { LayerWeights } from '../types.js';
import type { LoRAAdapter } from '../lora.js';

export interface AttentionInputInfo {
  phase: 'prefill' | 'decode';
  layerIdx: number;
  numTokens?: number;
  kvLen?: number;
  numHeads?: number;
  numKVHeads?: number;
  headDim?: number;
  activationDtype?: string | null;
  inputDtype?: string | null;
  normedDtype?: string | null;
  kvDtype?: string | null;
  kvCacheDtype?: string | null;
  cachedKDtype?: string | null;
  cachedVDtype?: string | null;
  qDtype?: string | null;
  kDtype?: string | null;
  vDtype?: string | null;
  qWeightDtype?: string | null;
  kWeightDtype?: string | null;
  vWeightDtype?: string | null;
  oWeightDtype?: string | null;
  useF16Attention?: boolean;
  useF16Activations?: boolean;
  hasF16Weights?: boolean;
  matmulOutputDtype?: string | null;
  useFusedQKV?: boolean;
  kvStart?: number;
  kvLayout?: string;
  kvPageSize?: number | null;
  hotLen?: number | null;
  coldLen?: number | null;
  hotWindow?: number | null;
  hotStart?: number | null;
  coldQuantMode?: string | null;
}

export function recordAttentionInputs(
  state: { stats?: { attentionInputs?: AttentionInputInfo[] } } | null | undefined,
  info: AttentionInputInfo | null | undefined
): void;

export function resolveAttentionProjectionOutputDtype(attentionInputDtype: string): 'f16' | 'f32' | string;

export interface ProjectAttentionQKVOptions {
  recorder?: CommandRecorder | null;
  normed: Tensor;
  layerWeights: LayerWeights;
  numTokens: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  hiddenSize: number;
  layerIdx: number;
  matmulOutputDtype: string;
  getWeightBuffer?: (weight: GPUBuffer | WeightBuffer | Float32Array | ArrayBuffer | CpuWeightBuffer, label: string) => GPUBuffer | WeightBuffer;
  lora?: LoRAAdapter | null;
  releaseTemporary: (buffer: GPUBuffer) => void;
  onFusedQKV?: ((info: { qSize: number; kSize: number; vSize: number; totalSize: number }) => void) | null;
}

export interface ProjectAttentionQKVResult {
  qTensor: Tensor;
  kTensor: Tensor;
  vTensor: Tensor;
  usedFusedQKV: boolean;
}

export function projectAttentionQKV(options: ProjectAttentionQKVOptions): Promise<ProjectAttentionQKVResult>;

export interface ApplyAttentionQKNormOptions {
  recorder?: CommandRecorder | null;
  qTensor: Tensor;
  kTensor: Tensor;
  layerWeights: LayerWeights;
  getNormWeightBuffer?: (weight: GPUBuffer | Float32Array | ArrayBuffer | CpuWeightBuffer, label: string) => GPUBuffer;
  rmsNormEps: number;
  numTokens: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  rmsNormWeightOffset?: boolean;
  releaseTemporary: (buffer: GPUBuffer) => void;
  onQNormApplied?: ((tensor: Tensor) => Promise<void> | void) | null;
  onKNormApplied?: ((tensor: Tensor) => Promise<void> | void) | null;
}

export function applyAttentionQKNorm(
  options: ApplyAttentionQKNormOptions
): Promise<{ qTensor: Tensor; kTensor: Tensor }>;
