

import { log } from '../debug/index.js';

const bufferDtypes = new WeakMap();

function isProviderGpuBufferLike(value) {
  return (
    typeof value === 'object'
    && value !== null
    && typeof value.size === 'number'
    && typeof value.usage === 'number'
    && typeof value.destroy === 'function'
    && typeof value.mapAsync === 'function'
    && typeof value.getMappedRange === 'function'
    && typeof value.unmap === 'function'
  );
}

export function isGpuBufferInstance(value) {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  if (value.__dopplerFakeGPUBuffer === true) {
    return true;
  }
  if (typeof GPUBuffer !== 'undefined' && value instanceof GPUBuffer) {
    return true;
  }
  if (value.constructor?.name === 'FakeBuffer' && typeof value.size === 'number' && typeof value.usage === 'number' && typeof value.destroy === 'function') {
    return true;
  }
  return isProviderGpuBufferLike(value);
}

function canTrackBuffer(buffer) {
  return isGpuBufferInstance(buffer);
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
  label,
  materializations = null
) {
  tagBufferDtype(buffer, dtype);
  const normalizedMaterializations = {};
  if (materializations && typeof materializations === 'object') {
    for (const [materializationDtype, descriptor] of Object.entries(materializations)) {
      if (!descriptor?.buffer) {
        continue;
      }
      tagBufferDtype(descriptor.buffer, materializationDtype);
      normalizedMaterializations[materializationDtype] = Object.freeze({
        buffer: descriptor.buffer,
        layout: descriptor.layout ?? layout,
      });
    }
  }
  normalizedMaterializations[dtype] = Object.freeze({
    buffer,
    layout,
  });
  return {
    buffer,
    dtype,
    layout,
    shape: Object.freeze([...shape]),
    label,
    materializations: Object.freeze(normalizedMaterializations),
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

export function resolveWeightBufferMaterialization(weight, preferredDtype = null) {
  if (!isWeightBuffer(weight) || preferredDtype == null || preferredDtype === weight.dtype) {
    return weight;
  }
  const materialization = weight.materializations?.[preferredDtype];
  if (!materialization?.buffer) {
    return weight;
  }
  return {
    ...weight,
    buffer: materialization.buffer,
    dtype: preferredDtype,
    layout: materialization.layout ?? weight.layout,
  };
}
