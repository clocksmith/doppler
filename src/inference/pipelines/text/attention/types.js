

import { getDevice } from '../../../../gpu/device.js';
import { releaseBuffer } from '../../../../memory/buffer-pool.js';

// ============================================================================
// Debug Helpers
// ============================================================================


export function shouldDebugLayer(layerIdx, debugLayers) {
  if (debugLayers === null) return false;
  if (debugLayers === undefined || debugLayers.length === 0) {
    // Backward compat: default to layer 0 only
    return layerIdx === 0;
  }
  return debugLayers.includes(layerIdx);
}


export function markStageLogged(layerIdx, stage, flags) {
  if (!flags.loggedStages) {
    flags.loggedStages = new Set();
  }
  const key = `L${layerIdx}_${stage}`;
  if (flags.loggedStages.has(key)) {
    return true; // Already logged
  }
  flags.loggedStages.add(key);
  return false; // First time
}


export function releaseOrTrack(recorder, buffer) {
  if (recorder) {
    recorder.trackTemporaryBuffer(buffer);
  } else {
    releaseBuffer(buffer);
  }
}

// ============================================================================
// Q/K Norm Cache
// ============================================================================


const qkNormOnesCache = new WeakMap();
const qkNormZerosCache = new WeakMap();

function getCachedQKNormConstantBuffer(cache, size, value, label) {
  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for Q/K norm buffer');
  }
  let perDeviceCache = cache.get(device);
  if (!perDeviceCache) {
    perDeviceCache = new Map();
    cache.set(device, perDeviceCache);
  }
  const cached = perDeviceCache.get(size);
  if (cached) return cached;
  const data = new Float32Array(size);
  data.fill(value);
  const buffer = device.createBuffer({
    label: `${label}_${size}`,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  perDeviceCache.set(size, buffer);
  return buffer;
}


export function getQKNormOnesBuffer(headDim) {
  return getCachedQKNormConstantBuffer(qkNormOnesCache, headDim, 1, 'qk_norm_ones');
}

export function getQKNormZerosBuffer(size) {
  return getCachedQKNormConstantBuffer(qkNormZerosCache, size, 0, 'qk_norm_zeros');
}
