import { log } from '../debug/index.js';
import { isWeightBuffer } from '../gpu/weight-buffer.js';

export function toPositiveInt(value, label) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) {
    if (value != null && label) {
      log.debug('LayerLoader', `toPositiveInt: invalid value for ${label}: ${String(value)}`);
    }
    return null;
  }
  return Math.trunc(num);
}

export function getWeightShape(value) {
  if (!isWeightBuffer(value)) return null;
  if (!Array.isArray(value.shape)) {
    log.warn('LayerLoader', `getWeightShape: expected value.shape to be an array, got ${typeof value.shape}`);
    return null;
  }
  if (value.shape.length < 2) return null;
  const dim0 = toPositiveInt(value.shape[0], 'shape[0]');
  const dim1 = toPositiveInt(value.shape[1], 'shape[1]');
  if (dim0 === null || dim1 === null) {
    return null;
  }
  return [dim0, dim1];
}

