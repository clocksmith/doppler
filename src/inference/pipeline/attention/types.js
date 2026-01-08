/**
 * Attention Types and Utilities
 *
 * Shared interfaces, debug helpers, and utility functions for attention operations.
 *
 * @module inference/pipeline/attention/types
 */

import { getDevice } from '../../../gpu/device.js';
import { releaseBuffer } from '../../../gpu/buffer-pool.js';

// ============================================================================
// Debug Helpers
// ============================================================================

/**
 * Check if a layer should be debugged
 * @param {number} layerIdx
 * @param {number[] | null | undefined} debugLayers
 * @returns {boolean}
 */
export function shouldDebugLayer(layerIdx, debugLayers) {
  if (debugLayers === null) return false;
  if (debugLayers === undefined || debugLayers.length === 0) {
    // Backward compat: default to layer 0 only
    return layerIdx === 0;
  }
  return debugLayers.includes(layerIdx);
}

/**
 * Check if a stage has been logged for a layer, and mark it as logged
 * @param {number} layerIdx
 * @param {string} stage
 * @param {import('./types.js').AttentionDebugFlags} flags
 * @returns {boolean}
 */
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

/**
 * Release buffer or track for later cleanup (recording mode).
 * @param {import('../../../gpu/kernel-selector.js').CommandRecorder | undefined} recorder
 * @param {GPUBuffer} buffer
 * @returns {void}
 */
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

/** @type {Map<number, GPUBuffer>} */
const qkNormOnesCache = new Map();

/**
 * Get or create a buffer of ones for Q/K norm when per-head weights are absent.
 * @param {number} headDim
 * @returns {GPUBuffer}
 */
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
