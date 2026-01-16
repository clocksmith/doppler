

import { createTensor } from '../../../gpu/tensor.js';


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
      swigluLimit: config.swigluLimit,
      activationDtype: inputTensor.dtype,
    },
    moeRouter,
    expertWeights,
    expertLoader,
    layerIdx,
     (layerRouterWeights)
  );

  return createTensor(outputBuffer, inputTensor.dtype, [...inputTensor.shape], 'moe_ffn_output');
}
