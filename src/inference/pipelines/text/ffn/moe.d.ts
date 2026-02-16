/**
 * Mixture of Experts (MoE) FFN Operations
 *
 * Handles MoE FFN computations with expert routing and selection.
 *
 * @module inference/pipelines/text/ffn/moe
 */

import type { Tensor } from '../../../../gpu/tensor.js';
import type { LayerContext } from '../types.js';

/**
 * Run MoE FFN on GPU.
 * Routes tokens to experts and combines expert outputs.
 */
export declare function runMoEFFNGPU(
  layerIdx: number,
  inputTensor: Tensor,
  numTokens: number,
  context: LayerContext
): Promise<Tensor>;
