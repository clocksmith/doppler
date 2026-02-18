

import { createTensor } from '../../../../gpu/tensor.js';


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
  if (!Number.isFinite(config.numExperts) || config.numExperts <= 0) {
    throw new Error('MoE config is missing numExperts.');
  }
  if (!Number.isFinite(config.moeTopK) || config.moeTopK <= 0) {
    throw new Error('MoE config is missing moeTopK.');
  }

  const { moeFeedForwardGPU } = await import('../moe-impl.js');

  const outputBuffer = await moeFeedForwardGPU(
    inputTensor.buffer,
    numTokens,
    {
      modelType: config.modelType ?? (config.expertFormat === 'gpt-oss' ? 'gpt-oss' : null),
      hiddenSize: config.hiddenSize,
      intermediateSize: config.intermediateSize,
      numExperts: config.numExperts,
      moeTopK: config.moeTopK,
      expertFormat: config.expertFormat,
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
