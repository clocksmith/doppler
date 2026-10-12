import { getRuntimeConfig } from '../../../../config/runtime.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';

/** @param {unknown} precision @returns {1 | 2} */
export function recurrentStateComponents(precision) {
  if (precision !== 'f32' && precision !== 'two-f32') {
    throw new Error(`Unsupported linear recurrent state precision: ${String(precision)}.`);
  }
  return /** @type {1 | 2} */ (selectRuleValue('kernels', 'linearAttention', 'stateComponents', { precision }));
}

/** @param {{linearRecurrentStatePrecision?: 'f32' | 'two-f32' | null} | null} [sessionSettings] */
export function resolveRecurrentStateLayout(sessionSettings = null) {
  const precision = sessionSettings?.linearRecurrentStatePrecision
    ?? getRuntimeConfig().inference.session.linearRecurrentStatePrecision;
  return { precision, components: recurrentStateComponents(precision) };
}

/** @param {import('../linear-attention.js').LinearLayerRuntimeState} state @returns {1 | 2} */
export function assertRecurrentStateLayout(state) {
  const components = recurrentStateComponents(state.recurrentStatePrecision);
  const scalarCount = state.numVHeads * state.headKDim * state.headVDim;
  if (![state.numVHeads, state.headKDim, state.headVDim].every(value => Number.isSafeInteger(value) && value > 0)
      || !Number.isSafeInteger(scalarCount) || scalarCount <= 0
      || state.recurrentStateComponents !== components
      || !(state.recurrentState instanceof Float32Array)
      || state.recurrentState.length !== scalarCount * components) {
    throw new Error('Linear recurrent state storage does not match its declared precision and geometry.');
  }
  return components;
}

/** @param {import('../linear-attention.js').LinearLayerRuntimeState} state */
export function assertConvolutionStateLayout(state) {
  const components = recurrentStateComponents(state.convStatePrecision);
  const count = state.convDim * state.convKernelSize;
  if (![state.convDim, state.convKernelSize].every(v => Number.isSafeInteger(v) && v > 0)
    || !Number.isSafeInteger(count) || state.convStateComponents !== components
    || !(state.convState instanceof Float32Array) || state.convState.length !== count * components
    || typeof state.normWeightOffset !== 'boolean') {
    throw new Error('Linear convolution state storage does not match its declared precision and geometry.');
  }
  return components;
}
