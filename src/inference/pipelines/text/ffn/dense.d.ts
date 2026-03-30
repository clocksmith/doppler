/**
 * Dense FFN Operations
 *
 * Handles standard dense (non-MoE) FFN computations including:
 * - Gate/Up -> Activation -> Down projections
 * - Fused FFN variants
 * - Fused Down+Norm optimization
 *
 * @module inference/pipelines/text/ffn/dense
 */

import type { Tensor } from '../../../../gpu/tensor.js';
import type { WeightBuffer } from '../../../../gpu/weight-buffer.js';
import type { LayerContext, LayerWeights } from '../types.js';

export declare function resolveGateUpPathMode(options?: {
  kernelPath?: Record<string, unknown> | null;
  phase?: 'prefill' | 'decode' | null;
  layerIdx?: number;
}): 'fused' | 'split' | 'implicit';

export declare function resolveFusedGateUpWeights(
  layerWeights: LayerWeights | undefined,
  options?: {
    activationDtype?: 'f16' | 'f32' | null;
    hiddenSize?: number;
    kernelPath?: Record<string, unknown> | null;
    phase?: 'prefill' | 'decode' | null;
    layerIdx?: number;
  }
): {
  gate: GPUBuffer | WeightBuffer | null;
  up: GPUBuffer | WeightBuffer | null;
  gateDtype: string | null;
  upDtype: string | null;
  hasQ4KMaterialization: boolean;
};

/**
 * Run dense (non-MoE) FFN on GPU.
 */
export declare function runDenseFFNGPU(
  layerIdx: number,
  inputTensor: Tensor,
  numTokens: number,
  context: LayerContext,
  layerWeights: LayerWeights | undefined
): Promise<Tensor>;

/**
 * Run dense FFN with fused down projection + post-FFN norm.
 * Used for sandwich norm architectures when conditions allow fusion.
 */
export declare function runDenseFFNWithFusedPostNormGPU(
  layerIdx: number,
  inputTensor: Tensor,
  numTokens: number,
  context: LayerContext,
  layerWeights: LayerWeights,
  residualTensor: Tensor,
  eps: number,
  transposeB: boolean,
  outputBuffer?: GPUBuffer | null
): Promise<Tensor>;
