import { recordScale, runScale } from '../../../gpu/kernel-selector.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { normalizeLayerType } from './layer-input-execution.js';

export function resolveActivationDtype(dtype) {
  return selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', { dtype });
}

export function getConvLayerState(convLayerStates, layerIdx) {
  if (!convLayerStates) return {};
  return convLayerStates.get(layerIdx) ?? {};
}

export function isSlidingLayerType(layerType) {
  const normalized = normalizeLayerType(layerType);
  return normalized === 'sliding_attention'
    || normalized === 'local_attention'
    || normalized === 'local'
    || normalized === 'sliding';
}

export function resolveAttentionRotaryDim(config, layerType) {
  if (isSlidingLayerType(layerType)) {
    return config.ropeLocalRotaryDim ?? config.ropeRotaryDim;
  }
  return config.ropeRotaryDim;
}

export function resolveAttentionFrequencyBaseDim(config, layerType) {
  if (isSlidingLayerType(layerType)) {
    return config.ropeLocalFrequencyBaseDim ?? resolveAttentionRotaryDim(config, layerType);
  }
  return config.ropeFrequencyBaseDim ?? config.ropeRotaryDim;
}

export function resolveAttentionHeadDim(config, layerType) {
  if (isSlidingLayerType(layerType)) {
    return config.headDim;
  }
  return config.globalHeadDim ?? config.headDim;
}

function resolveProjectionOutputRows(layerWeight, hiddenSize) {
  if (!layerWeight || !Array.isArray(layerWeight.shape) || layerWeight.shape.length < 2) {
    return null;
  }
  const dim0 = Number(layerWeight.shape[0]);
  const dim1 = Number(layerWeight.shape[1]);
  if (!Number.isFinite(dim0) || !Number.isFinite(dim1)) {
    return null;
  }
  if (dim1 === hiddenSize) {
    return Math.trunc(dim0);
  }
  if (dim0 === hiddenSize) {
    return Math.trunc(dim1);
  }
  return null;
}

export function resolveAttentionNumKVHeads(config, layerType, layerWeights, headDim) {
  const kRows = resolveProjectionOutputRows(layerWeights?.kProj, config.hiddenSize);
  if (kRows != null && Number.isFinite(headDim) && headDim > 0 && kRows % headDim === 0) {
    return kRows / headDim;
  }
  if (!isSlidingLayerType(layerType) && Number.isFinite(config.numGlobalKVHeads) && config.numGlobalKVHeads > 0) {
    return Math.trunc(config.numGlobalKVHeads);
  }
  return config.numKVHeads;
}

export function resolveLayerScalarValue(layerScalar) {
  if (layerScalar == null) {
    return 1;
  }
  if (!(layerScalar instanceof Float32Array) || layerScalar.length === 0) {
    throw new Error(
      'Gemma 4 per-layer input layer_scalar must be CPU-resident Float32Array data. ' +
      'Re-convert or reload the model with the updated loader.'
    );
  }
  const value = Number(layerScalar[0]);
  if (!Number.isFinite(value)) {
    throw new Error(`Gemma 4 layer_scalar must be finite; got "${String(layerScalar[0])}".`);
  }
  return value;
}

export async function applyLayerScalar(layerIdx, tensor, size, context, layerWeights) {
  const layerScalar = resolveLayerScalarValue(layerWeights?.layerScalar ?? null);
  if (layerScalar === 1) {
    return tensor;
  }
  return context.recorder
    ? recordScale(context.recorder, tensor, layerScalar, { count: size })
    : runScale(tensor, layerScalar, { count: size });
}
