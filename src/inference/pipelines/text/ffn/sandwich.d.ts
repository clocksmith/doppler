/**
 * Sandwich Norm FFN Processing
 *
 * Handles FFN with sandwich norm architecture (Gemma 3 style) where
 * pre/post FFN norms wrap the FFN block.
 *
 * @module inference/pipelines/text/ffn/sandwich
 */

import type { Tensor } from '../../../../gpu/tensor.js';
import type { LayerContext, LayerWeights, SandwichNormInfo } from '../types.js';

/**
 * Process FFN with sandwich norm architecture (Gemma 3).
 * Input and output are Tensor for dtype-aware processing.
 */
export declare function processFFNWithSandwichNorm(
  layerIdx: number,
  postAttn: Tensor,
  numTokens: number,
  size: number,
  context: LayerContext,
  layerWeights: LayerWeights | undefined,
  sandwichNorm: SandwichNormInfo
): Promise<Tensor>;
