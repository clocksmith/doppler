

import { log } from '../debug/index.js';

const bufferDtypes = new WeakMap();

function canTrackBuffer(buffer) {
  return typeof GPUBuffer !== 'undefined' && buffer instanceof GPUBuffer;
}

function normalizeDtype(dtype) {
  if (typeof dtype !== 'string') {
    log.debug('WeightBuffer', 'normalizeDtype received non-string dtype: ' + typeof dtype + ' (' + String(dtype) + ')');
    return null;
  }
  const value = dtype.toLowerCase();
  return value.length > 0 ? value : null;
}

export function tagBufferDtype(buffer, dtype) {
  if (!canTrackBuffer(buffer)) return;
  const normalized = normalizeDtype(dtype);
  if (!normalized) return;
  bufferDtypes.set(buffer, normalized);
}

export function getBufferDtype(buffer) {
  if (!canTrackBuffer(buffer)) return null;
  return bufferDtypes.get(buffer) ?? null;
}


export function createWeightBuffer(
  buffer,
  dtype,
  layout,
  shape,
  label
) {
  tagBufferDtype(buffer, dtype);
  return {
    buffer,
    dtype,
    layout,
    shape: Object.freeze([...shape]),
    label,
  };
}


export function createCpuWeightBuffer(
  data,
  dtype,
  layout,
  shape,
  label
) {
  return {
    data,
    dtype,
    layout,
    shape: Object.freeze([...shape]),
    label,
  };
}


export function isColumnMajor(weight) {
  return weight.layout === 'column';
}


export function isWeightBuffer(value) {
  return (
    typeof value === 'object' &&
    value !== null &&
    'buffer' in value &&
    'dtype' in value &&
    'layout' in value &&
    'shape' in value
  );
}


export function isCpuWeightBuffer(value) {
  return (
    typeof value === 'object' &&
    value !== null &&
    'data' in value &&
    'dtype' in value &&
    'layout' in value &&
    'shape' in value
  );
}

// Intentionally lenient: does not call isValidGPUBuffer on value.buffer.
// This function is used as a structural duck-type check to distinguish tensor-like
// wrappers from raw GPUBuffer objects. The buffer property may hold a CPU typed array,
// a GPU buffer, or a test double. Callers that need a valid GPU buffer should validate
// the extracted buffer separately via isValidGPUBuffer in device.js.
function isTensorLike(value) {
  return (
    typeof value === 'object' &&
    value !== null &&
    'buffer' in value &&
    'dtype' in value &&
    'shape' in value
  );
}


export function getBuffer(weight) {
  if (isWeightBuffer(weight)) return weight.buffer;
  if (isTensorLike(weight)) return weight.buffer;
  return weight;
}


export function getLayout(weight) {
  return isWeightBuffer(weight) ? weight.layout : null;
}


export function getWeightDtype(weight) {
  if (isWeightBuffer(weight)) return weight.dtype;
  if (isTensorLike(weight)) return weight.dtype;
  return getBufferDtype(weight);
}
