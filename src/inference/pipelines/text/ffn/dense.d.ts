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
import type { LayerContext, LayerWeights } from '../types.js';

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
