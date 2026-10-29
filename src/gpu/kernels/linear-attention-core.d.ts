import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { ExecutionV1PoliciesSchema } from '../../config/schema/execution-v1.schema.js';
import type { LinearNormGateActivation } from '../../config/schema/inference.schema.js';

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
  aLogGPU: GPUBuffer;
  normWeightGPU: GPUBuffer;
  convStateGPU: GPUBuffer;
  convStatePrecision?: 'f32' | 'two-f32';
  convStateComponents?: 1 | 2;
  normWeightOffset?: boolean;
  recurrentStateGPU: GPUBuffer;
  recurrentStatePrecision: 'f32' | 'two-f32';
  recurrentStateComponents: 1 | 2;
}

export interface RunLinearAttentionCoreGPUOptions {
  numTokens: number;
  /** Resolved session; standalone calls use the current runtime session. */
  runtimeSession?: Pick<import('../../config/schema/execution-v1.schema.js').ExecutionV1SessionSchema, 'useLinearAttentionRegisterPrefill' | 'useLinearAttentionRegisterDecode' | 'useLinearAttentionFusedDecodeCore'>;
  /** Resolved model-config gated-RMSNorm activation; selects and guards the gate kernel. */
  normGateActivation: LinearNormGateActivation;
  /**
   * Paired execution: model-config `pairedMechanismReplacements`, the declared replacement
   * programs by `operation/variant`. Required for two-F32 inputs; ignored otherwise.
   */
  mechanismReplacements?: Readonly<Record<string, string>>;
  layerIdx?: number;
  qkL2NormEps?: number;
  recorder?: CommandRecorder | null;
  outputDtype?: 'f16' | 'f32';
  executionPolicies?: ExecutionV1PoliciesSchema | null;
  abPacked?: boolean;
  qkvzPacked?: boolean;
  bProjOffsetElements?: number;
}

export declare function runLinearAttentionCoreGPU(
  qkvTensor: Tensor,
  zTensor: Tensor,
  aTensor: Tensor,
  bTensor: Tensor,
  layerState: LinearAttentionCoreLayerState,
  options: RunLinearAttentionCoreGPUOptions
): Promise<Tensor>;
