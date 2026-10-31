import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { Tensor } from '../../../../gpu/tensor.js';
import type { WeightBuffer } from '../../../../gpu/weight-buffer.js';
import type { KernelPathSchema } from '../../../../config/schema/index.js';

/**
 * The registered variant of a layer's declared linear_norm_ab step (decode input norm
 * fused with in_proj_a and in_proj_b), or null when the layer declares none. Throws
 * for a step that is not exactly one registered decode step.
 */
export declare function resolveLinearNormAB(
  kernelPath: KernelPathSchema | null | undefined,
  layerIdx: number,
  phase: 'prefill' | 'decode'
): 'f32' | 'f32_single_pass' | null;

export interface DeclaredLinearNormABOptions {
  kernelPath: KernelPathSchema | null | undefined;
  phase: 'prefill' | 'decode';
  layerIdx: number;
  numTokens: number;
  /** Layer input [1, hiddenSize] F32. */
  inputTensor: Tensor;
  /** A norm computed elsewhere; the fused step refuses it (and releases it). */
  precomputedInputNorm: Tensor | null;
  layerWeights: {
    inputNorm?: unknown;
    linearInProjA: GPUBuffer | WeightBuffer;
    linearInProjB: GPUBuffer | WeightBuffer;
  };
  getNormWeightBuffer: (weight: unknown, label: string) => GPUBuffer | WeightBuffer;
  projectionDtype: string;
  rmsNormWeightOffset: boolean;
  eps: number;
  hasABLoRA: boolean;
  hiddenSize: number;
  /** Value heads: the row count of in_proj_a and in_proj_b. */
  rows: number;
}

/**
 * Record a declared linear_norm_ab step into decode-workspace outputs: the normalized
 * row, a and b. Null when the layer declares none; throws when the declared step cannot
 * cover this call.
 */
export declare function recordDeclaredLinearNormAB(
  recorder: CommandRecorder | null | undefined,
  options: DeclaredLinearNormABOptions
): Promise<{ normed: Tensor; a: Tensor; b: Tensor } | null>;
