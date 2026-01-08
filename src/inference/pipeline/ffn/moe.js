/**
 * Mixture of Experts (MoE) FFN Operations
 *
 * Handles MoE FFN computations with expert routing and selection.
 *
 * @module inference/pipeline/ffn/moe
 */

import { createTensor } from '../../../gpu/tensor.js';

/**
 * Run MoE FFN on GPU.
 * Routes tokens to experts and combines expert outputs.
 * @param {number} layerIdx
 * @param {import('../../../gpu/tensor.js').Tensor} inputTensor
 * @param {number} numTokens
 * @param {import('../types.js').LayerContext} context
 * @returns {Promise<import('../../../gpu/tensor.js').Tensor>}
 */
export async function runMoEFFNGPU(
  layerIdx,
  inputTensor,
  numTokens,
  context
) {
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
    /** @type {Map<number, import('../moe-impl.js').LayerRouterWeights> | undefined} */ (layerRouterWeights)
  );

  return createTensor(outputBuffer, inputTensor.dtype, [...inputTensor.shape], 'moe_ffn_output');
}
