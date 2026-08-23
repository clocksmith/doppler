

import { releaseBuffer } from '../../../../memory/buffer-pool.js';
import { getConstantVectorBuffer } from '../../../../gpu/constant-buffer.js';

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


export function getQKNormOnesBuffer(headDim) {
  return getConstantVectorBuffer(headDim, 1, 'qk_norm_ones');
}

export function getQKNormZerosBuffer(size) {
  return getConstantVectorBuffer(size, 0, 'qk_norm_zeros');
}
