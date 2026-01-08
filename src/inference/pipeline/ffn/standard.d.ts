/**
 * Standard FFN Processing
 *
 * Handles FFN with standard architecture (LLaMA-style) where
 * post-attention norm precedes the FFN block.
 *
 * @module inference/pipeline/ffn/standard
 */

import type { Tensor } from '../../../gpu/tensor.js';
import type { LayerContext, LayerWeights } from '../types.js';

/**
 * Process FFN with standard architecture (LLaMA-style).
 */
export declare function processFFNStandard(
  layerIdx: number,
  postAttn: Tensor,
  numTokens: number,
  size: number,
  context: LayerContext,
  layerWeights: LayerWeights | undefined
): Promise<Tensor>;
