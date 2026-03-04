import type { LayerWeights } from './types.js';
import type { Tensor } from '../../../gpu/tensor.js';
import type { WeightBuffer } from '../../../gpu/weight-buffer.js';
import type { CommandRecorder } from '../../../gpu/command-recorder.js';
import type { LinearNormMode } from '../../../config/schema/index.js';

export interface LinearLayerRuntimeState {
  layerIdx: number;
  seqLen: number;
  warnedSeqMismatch: boolean;
  convKernelSize: number;
  convDim: number;
  keyDim: number;
  valueDim: number;
  numKHeads: number;
  numVHeads: number;
  headKDim: number;
  headVDim: number;
  qSize: number;
  kSize: number;
  vSize: number;
  qRep: number;
  normMode: LinearNormMode;
  rmsNormEps: number;
  convWeight: Float32Array;
  dtBias: Float32Array;
  aNegExp: Float32Array;
  normWeight: Float32Array;
  convState: Float32Array;
  recurrentState: Float32Array;
  convWeightGPU?: GPUBuffer | null;
  dtBiasGPU?: GPUBuffer | null;
  aNegExpGPU?: GPUBuffer | null;
  normWeightGPU?: GPUBuffer | null;
  convStateGPU?: GPUBuffer | null;
  recurrentStateGPU?: GPUBuffer | null;
}

export interface LinearAttentionRuntime {
  schemaVersion: number;
  layers: Map<number, LinearLayerRuntimeState>;
}

export interface RunLinearAttentionLayerOptions {
  layerIdx: number;
  numTokens: number;
  hiddenSize: number;
  config: {
    linearNumKeyHeads: number | null;
    linearNumValueHeads: number | null;
    linearKeyHeadDim: number | null;
    linearValueHeadDim: number | null;
    linearConvKernelDim?: number | null;
    linearNormMode?: LinearNormMode | null;
    rmsNormEps: number;
    rmsNormWeightOffset: boolean;
  };
  currentSeqLen: number;
  activationDtype?: 'f16' | 'f32';
  kernelPath?: string | null;
  linearRuntime?: LinearAttentionRuntime | null;
  getWeightBuffer: (
    weight: GPUBuffer | WeightBuffer | Float32Array | ArrayBuffer,
    label: string
  ) => GPUBuffer | WeightBuffer;
  getNormWeightBuffer: (
    weight: GPUBuffer | Float32Array | ArrayBuffer,
    label: string
  ) => GPUBuffer;
  recorder?: CommandRecorder | null;
}

export declare function hasLinearAttentionLayers(layerTypes: unknown): boolean;

export declare function createLinearAttentionRuntime(): LinearAttentionRuntime;

export declare function resetLinearAttentionRuntime(
  runtime: LinearAttentionRuntime | null | undefined
): LinearAttentionRuntime;

export declare function cloneLinearAttentionRuntime(
  runtime: LinearAttentionRuntime | null | undefined
): Promise<LinearAttentionRuntime>;

export declare function restoreLinearAttentionRuntime(
  runtime: LinearAttentionRuntime | null | undefined,
  snapshot: LinearAttentionRuntime | null | undefined
): LinearAttentionRuntime;

export declare function runLinearAttentionLayer(
  inputTensor: Tensor,
  layerWeights: LayerWeights | null,
  options: RunLinearAttentionLayerOptions
): Promise<Tensor>;
