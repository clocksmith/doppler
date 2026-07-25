import { createCommandRecorder } from '../../../../gpu/command-recorder.js';
import { interpretAttentionWithRecorder } from './interpreter.js';

export async function runLayerAttentionGPU(
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
  const phase = config?.isPrefill ? 'prefill' : 'decode';
  const layerIdx = Number.isInteger(config?.layerIdx) ? config.layerIdx : 'unknown';
  const recorder = createCommandRecorder(
    `attention_${phase}_layer_${layerIdx}`,
    { recordLabels: debug === true }
  );
  try {
    const result = await interpretAttentionWithRecorder(
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
    recorder.submit();
    return result;
  } catch (error) {
    recorder.abort();
    throw error;
  }
}
