import type { Tensor } from '../../../../gpu/tensor.js';
import type { LayerContext, LayerWeights } from '../types.js';

export declare function validatePairedFFN(
  layerIdx: number, input: Tensor, numTokens: number, context: LayerContext, weights: LayerWeights
): { hiddenSize: number; intermediateSize: number };

/** Execute the declared split paired gate/up -> SiLU(gate)*up -> down program. */
export declare function runPairedFFNGPU(
  layerIdx: number, input: Tensor, numTokens: number, context: LayerContext, weights: LayerWeights
): Promise<Tensor>;
