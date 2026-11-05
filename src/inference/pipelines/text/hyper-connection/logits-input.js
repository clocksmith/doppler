import { createTensor } from '../../../../gpu/tensor.js';
import { isGpuBufferInstance } from '../../../../gpu/weight-buffer.js';
import { runRootMixer } from './decoder-layer.js';
import { createHyperConnectionProjection } from './projection.js';

/** @type {typeof import('./logits-input.js').mixHyperConnectionLogitsInput} */
export async function mixHyperConnectionLogitsInput(recorder, hidden, numTokens, weights, config) {
  const hyper = config.hyperConnections;
  const mixer = weights.hyperConnectionMixer;
  if (!hyper || !mixer || weights.finalNorm || !isGpuBufferInstance(hidden)
    || config.activationDtype !== 'f32' || config.activationPrecision) {
    throw new Error('[HyperConnection] Logits require four-stream f32 input and a root mixer in place of the final norm.');
  }
  const width = hyper.streamCount * hyper.streamWidth;
  if (hidden.size < numTokens * width * 4) throw new Error('[HyperConnection] Logits input does not contain all residual streams.');
  const project = createHyperConnectionProjection({ recorder, kernelPath: config.kernelPath,
    phase: config.executionPhase, layerIdx: 0 });
  return runRootMixer(recorder, { streams: createTensor(hidden, 'f32', [numTokens, width], 'hc_logits_input'),
    weights: mixer, hyper, numTokens, project });
}
