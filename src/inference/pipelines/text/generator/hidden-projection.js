import { computeLogits } from '../logits/index.js';
import { getLogitsConfig, getLogitsWeights } from './logits-config.js';

// Projects caller-supplied hidden states through this model's own final
// normalization and output head: the computation every decode step ends with,
// applied to vectors that did not come from the last layer. The result is an
// observation aid (a "logit lens"), not something the model produced at that
// layer. Nothing here touches sequence state or the KV cache.
/** @type {typeof import('./hidden-projection.js').projectHiddenStates} */
export const projectHiddenStates = async function projectHiddenStates(state, hiddenStates, rowCount) {
  if (!(hiddenStates instanceof Float32Array)) {
    throw new Error('projectHiddenStates requires hidden states as a Float32Array.');
  }
  if (!Number.isInteger(rowCount) || rowCount < 1) {
    throw new Error('projectHiddenStates requires a positive whole number of rows.');
  }
  const config = getLogitsConfig(state);
  if (config.hyperConnections) {
    // Their trace rows hold several streams and a mixer replaces the final norm.
    throw new Error('projectHiddenStates does not support hyper-connection models: a recorded layer output there is not one hidden state.');
  }
  if (config.activationPrecision !== undefined && config.activationPrecision !== null) {
    throw new Error(`projectHiddenStates requires single-component activations; the active plan uses "${config.activationPrecision}".`);
  }
  if (hiddenStates.length !== rowCount * config.hiddenSize) {
    throw new Error(`projectHiddenStates expected ${rowCount} rows of ${config.hiddenSize} values (${rowCount * config.hiddenSize}), got ${hiddenStates.length}.`);
  }
  for (let index = 0; index < hiddenStates.length; index += 1) {
    if (!Number.isFinite(hiddenStates[index])) {
      throw new Error(`projectHiddenStates received a non-finite value at index ${index}.`);
    }
  }
  const logits = await computeLogits(
    hiddenStates,
    rowCount,
    getLogitsWeights(state),
    config,
    state.useGPU,
    state.debugFlags,
    undefined,
    undefined,
    state.runtimeConfig.shared.debug.probes,
    { lastPositionOnly: false }
  );
  if (!(logits instanceof Float32Array) || logits.length % rowCount !== 0) {
    throw new Error('projectHiddenStates did not receive one logit row per hidden state.');
  }
  return { logits, rowCount, vocabSize: logits.length / rowCount };
};
