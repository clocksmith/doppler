/**
 * Mixture of Experts (MoE) FFN Operations
 *
 * Handles MoE FFN computations with expert routing and selection.
 *
 * @module inference/pipeline/ffn/moe
 */

import { Tensor, createTensor } from '../../../gpu/tensor.js';
import type { LayerContext } from '../types.js';

/**
 * Run MoE FFN on GPU.
 * Routes tokens to experts and combines expert outputs.
 */
export async function runMoEFFNGPU(
  layerIdx: number,
  inputTensor: Tensor,
  numTokens: number,
  context: LayerContext
): Promise<Tensor> {
  const { config, moeRouter, expertWeights, expertLoader, layerRouterWeights } = context;

  if (!moeRouter || !expertWeights || !expertLoader) {
    throw new Error('MoE components not initialized');
  }

  const { moeFeedForwardGPU } = await import('../moe-impl.js');

  const outputBuffer = await moeFeedForwardGPU(
    inputTensor.buffer,
    numTokens,
    {
      hiddenSize: config.hiddenSize,
      intermediateSize: config.intermediateSize,
      numExperts: config.numExperts || 8,
      moeTopK: config.moeTopK || 2,
      hiddenActivation: config.hiddenActivation,
    },
    moeRouter,
    expertWeights,
    expertLoader,
    layerIdx,
    layerRouterWeights as Map<number, import('../moe-impl.js').LayerRouterWeights> | undefined
  );

  return createTensor(outputBuffer, inputTensor.dtype, [...inputTensor.shape], 'moe_ffn_output');
}
