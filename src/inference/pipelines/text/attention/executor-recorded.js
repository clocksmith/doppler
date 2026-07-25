import { interpretAttentionWithRecorder } from './interpreter.js';

export async function recordLayerAttentionGPU(
  recorder,
  input,
  layerWeights,
  config,
  state,
  debug,
  debugFlags,
  getWeightBuffer,
  getNormWeightBuffer,
  debugCheckBuffer,
  lora
) {
  return interpretAttentionWithRecorder(
    recorder,
    input,
    layerWeights,
    config,
    state,
    debug,
    debugFlags,
    getWeightBuffer,
    getNormWeightBuffer,
    debugCheckBuffer,
    lora
  );
}
