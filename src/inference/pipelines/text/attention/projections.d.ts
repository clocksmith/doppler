import type { CommandRecorder } from '../../../../gpu/kernel-selector.js';
import type { Tensor } from '../../../../gpu/tensor.js';
import type { WeightBuffer, CpuWeightBuffer } from '../../../../gpu/weight-buffer.js';
import type { LayerWeights } from '../types.js';
import type { LoRAAdapter } from '../lora.js';
import type { MatmulDebugConfigSchema } from '../../../../config/schema/debug.schema.js';

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

export function shouldForceF32AttentionProjectionForRoPE(options: {
  attentionInputDtype: string;
  headDim: number;
  rotaryDim?: number;
  interleaved?: boolean;
  kernelPathIsF16?: boolean;
}): boolean;
export function resolveAttentionProjectionOutputDtype(
  attentionInputDtype: string,
  options?: { forceF32?: boolean }
): 'f16' | 'f32' | string;
export function resolveProjectionMatmulDtype(options: {
  useFusedQKV: boolean;
  phase: 'prefill' | 'decode';
  layerIdx: number;
  kernelPath: Record<string, unknown> | null | undefined;
  precisionField: 'inputDtype' | 'outputDtype';
  fallbackDtype: 'f16' | 'f32' | string | null | undefined;
}): 'f16' | 'f32' | string | null | undefined;
export function resolveProjectionSliceOffsetBytes(
  weightBuffer: WeightBuffer | Tensor | GPUBuffer | null | undefined,
  outputRows: number,
  inputCols: number
): number;

export interface AttentionQKNormState {
  wantsQKNorm: boolean;
  hasQNorm: boolean;
  hasKNorm: boolean;
  allowUnitQKNorm: boolean;
  skipKNorm: boolean;
}

export function hasAttentionProjectionDiagnostics(state: {
  operatorDiagnostics?: {
    enabled?: boolean;
    tsirFixture?: { dir?: string | null } | null;
  } | null;
  debugProbes?: Array<{ stage?: string | null }> | null;
} | null | undefined): boolean;

export function hasAttentionStageDiagnostics(
  state: {
    operatorDiagnostics?: {
      enabled?: boolean;
      tsirFixture?: { dir?: string | null } | null;
    } | null;
    debugProbes?: Array<{ stage?: string | null }> | null;
  } | null | undefined,
  stages: string[]
): boolean;

export function resolveAttentionQKNormState(options: {
  config: {
    queryKeyNorm?: boolean;
    queryKeyNormWeightLayers?: number[] | null;
  };
  layerWeights: LayerWeights;
  layerIdx: number;
  reusesSharedKV: boolean;
}): AttentionQKNormState;

export interface ProjectAttentionQKNormFusionOptions {
  enabled: boolean;
  getNormWeightBuffer?: (weight: GPUBuffer | Float32Array | ArrayBuffer | CpuWeightBuffer, label: string) => GPUBuffer;
  rmsNormEps: number;
  rmsNormWeightOffset?: boolean;
  skipKNorm?: boolean;
  allowUnitQKNorm?: boolean;
  projectionDiagnosticsEnabled?: boolean;
}

export interface ProjectAttentionQKNormRoPEFusionOptions extends ProjectAttentionQKNormFusionOptions {
  freqsCos?: GPUBuffer | Tensor | null;
  freqsSin?: GPUBuffer | Tensor | null;
  headDim?: number;
  startPos?: number;
  rotaryDim?: number;
  pairSpanDim?: number;
  interleaved?: boolean;
  reusesSharedKV?: boolean;
  f16KVCacheWrite?: {
    keysBuffer: GPUBuffer;
    valuesBuffer: GPUBuffer;
    dstOffset: number;
  } | null;
}

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
  matmulDebug?: MatmulDebugConfigSchema | null;
  onFusedQKV?: ((info: { qSize: number; kSize: number; vSize: number; totalSize: number }) => void) | null;
  qkNormFusion?: ProjectAttentionQKNormFusionOptions | null;
  qkNormRoPEFusion?: ProjectAttentionQKNormRoPEFusionOptions | null;
}

export interface ProjectAttentionQKVResult {
  qTensor: Tensor;
  qGateTensor: Tensor | null;
  kTensor: Tensor | null;
  vTensor: Tensor | null;
  usedFusedQKV: boolean;
  valueAliasesKey: boolean;
  qkNormApplied: boolean;
  ropeApplied: boolean;
  kvCacheWriteFused: boolean;
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
  retainKInput?: boolean;
  allowUnitQKNorm?: boolean;
}

export function applyAttentionQKNorm(
  options: ApplyAttentionQKNormOptions
): Promise<{ qTensor: Tensor; kTensor: Tensor }>;

export interface ApplyAttentionValueNormOptions {
  recorder?: CommandRecorder | null;
  vTensor: Tensor;
  rmsNormEps: number;
  numTokens: number;
  numKVHeads: number;
  headDim: number;
  releaseTemporary: (buffer: GPUBuffer) => void;
  onVNormApplied?: ((tensor: Tensor) => Promise<void> | void) | null;
}

export function applyAttentionValueNorm(
  options: ApplyAttentionValueNormOptions
): Promise<Tensor>;
