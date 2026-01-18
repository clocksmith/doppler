

import { getDevice } from '../../../gpu/device.js';
import { releaseBuffer } from '../../../memory/buffer-pool.js';

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


const qkNormOnesCache = new Map();


export function getQKNormOnesBuffer(headDim) {
  const cached = qkNormOnesCache.get(headDim);
  if (cached) return cached;
  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for Q/K norm buffer');
  }
  const data = new Float32Array(headDim);
  data.fill(1);
  const buffer = device.createBuffer({
    label: `qk_norm_ones_${headDim}`,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  qkNormOnesCache.set(headDim, buffer);
  return buffer;
}
