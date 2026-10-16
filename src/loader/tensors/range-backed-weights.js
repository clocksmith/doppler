import { createCpuWeightBuffer } from '../../gpu/weight-buffer.js';
import { selectRuleValue } from '../../rules/rule-registry.js';
import { loadTensorRange } from './tensor-reader.js';

/** @type {import('./range-backed-weights.js').RangeBackedTensorSourceFactory} */
function createRangeBackedTensorSource(ctx, name, location) {
  const loadShardRange = ctx.loadShardRange;
  if (typeof loadShardRange !== 'function') {
    return null;
  }
  const normalizedLocationDtype = typeof location?.dtype === 'string'
    ? location.dtype.toLowerCase()
    : 'f32';
  return {
    kind: 'tensor_range_source',
    sourceDtype: normalizedLocationDtype,
    async loadRange(byteOffset, byteLength) {
      return loadTensorRange(location, name, byteOffset, byteLength, loadShardRange);
    },
  };
}

/** @type {import('./range-backed-weights.js').createRangeBackedWeightBuffer} */
export function createRangeBackedWeightBuffer(ctx, name, location) {
  const source = createRangeBackedTensorSource(ctx, name, location);
  if (!source || !location?.shape || location.shape.length !== 2) {
    return null;
  }
  const layout = ctx.resolveWeightLayout(location);
  const dtype = selectRuleValue('loader', 'weights', 'floatLocationDtype', {
    locationDtype: location.dtype,
  });
  return createCpuWeightBuffer(source, dtype, layout, location.shape, name);
}
