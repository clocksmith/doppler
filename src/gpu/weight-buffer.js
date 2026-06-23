

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

function normalizeWeightMetadata(metadata) {
  if (!metadata || typeof metadata !== 'object') {
    return null;
  }
  const normalized = {};
  for (const [key, value] of Object.entries(metadata)) {
    if (value === undefined) {
      continue;
    }
    normalized[key] = key === 'storageEncoding' && typeof value === 'string'
      ? value.toLowerCase()
      : value;
  }
  return Object.keys(normalized).length > 0 ? Object.freeze(normalized) : null;
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
  materializations = null,
  metadata = null
) {
  tagBufferDtype(buffer, dtype);
  const normalizedMetadata = normalizeWeightMetadata(metadata);
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
    ...(normalizedMetadata ? { metadata: normalizedMetadata } : {}),
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

export function createSplitWeightBuffer(
  sections,
  dtype,
  layout,
  shape,
  label,
  metadata = null
) {
  if (!Array.isArray(sections) || sections.length === 0) {
    throw new Error('[WeightBuffer] split weight buffer requires at least one section.');
  }
  const normalizedSections = sections.map((section, index) => {
    if (!section || !canTrackBuffer(section.buffer)) {
      throw new Error(`[WeightBuffer] split section ${index} is missing a GPU buffer.`);
    }
    if (!Number.isInteger(section.rowStart) || section.rowStart < 0) {
      throw new Error(`[WeightBuffer] split section ${index} has invalid rowStart.`);
    }
    if (!Number.isInteger(section.rowCount) || section.rowCount <= 0) {
      throw new Error(`[WeightBuffer] split section ${index} has invalid rowCount.`);
    }
    tagBufferDtype(section.buffer, dtype);
    return Object.freeze({
      buffer: section.buffer,
      rowStart: section.rowStart,
      rowCount: section.rowCount,
    });
  });
  const normalizedMetadata = normalizeWeightMetadata(metadata);
  return Object.freeze({
    kind: 'split_weight_buffer',
    sections: Object.freeze(normalizedSections),
    dtype,
    layout,
    shape: Object.freeze([...shape]),
    label,
    ...(normalizedMetadata ? { metadata: normalizedMetadata } : {}),
  });
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


export function isSplitWeightBuffer(value) {
  return (
    typeof value === 'object' &&
    value !== null &&
    value.kind === 'split_weight_buffer' &&
    Array.isArray(value.sections) &&
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
  if (isSplitWeightBuffer(weight)) return weight.layout;
  return isWeightBuffer(weight) ? weight.layout : null;
}


export function getWeightDtype(weight) {
  if (isSplitWeightBuffer(weight)) return weight.dtype;
  if (isWeightBuffer(weight)) return weight.dtype;
  if (isTensorLike(weight)) return weight.dtype;
  return getBufferDtype(weight);
}

export function getWeightMetadata(weight) {
  if (isWeightBuffer(weight) || isCpuWeightBuffer(weight) || isSplitWeightBuffer(weight)) {
    return weight.metadata ?? null;
  }
  return null;
}

export function resolveWeightBufferMaterialization(weight, preferredDtype = null) {
  if (isSplitWeightBuffer(weight)) {
    return weight;
  }
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
