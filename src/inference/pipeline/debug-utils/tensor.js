/**
 * Tensor inspection utilities for kernel step debugging.
 *
 * Provides GPU buffer readback and statistics computation for debugging.
 * These operations are expensive (require GPU sync) - use sparingly.
 *
 * Enable with setDebugCategories({ kernel: true }, { bufferStats: true })
 * or use DEBUG_PRESETS.kernelStep
 *
 * @example
 * // Enable kernel step debugging for layer 0 only
 * setDebugCategories({ kernel: true }, { layers: [0], bufferStats: true });
 *
 * // In pipeline code:
 * if (isKernelDebugEnabled(layerIdx)) {
 *   await dumpTensor(outputBuffer, 'matmul_output', { layerIdx });
 * }
 *
 * @module inference/pipeline/debug-utils/tensor
 */

import { readBuffer } from '../../../gpu/buffer-pool.js';
import { log } from '../../../debug/index.js';
import { isEnabled } from './config.js';
import { decodeReadback } from './utils.js';

// ============================================================================
// Tensor Inspection Functions
// ============================================================================

/**
 * Dump a GPU tensor's contents for debugging.
 * This is expensive (requires GPU sync + readback) - use sparingly.
 *
 * @param {GPUBuffer} buffer - GPU buffer to inspect
 * @param {string} label - Descriptive label for logging
 * @param {{ layerIdx?: number; shape?: [number, number] | [number]; dtype?: 'f32' | 'f16' | 'bf16'; sampleCount?: number; warnThreshold?: number }} [options] - Additional options
 * @returns {Promise<import('./tensor.js').TensorStats | null>}
 */
export async function dumpTensor(buffer, label, options = {}) {
  if (!isEnabled('kernel', options.layerIdx)) return null;

  const { shape, dtype = 'f32', sampleCount = 8, warnThreshold = 10000 } = options;

  try {
    const data = await readBuffer(buffer);
    const arr = decodeReadback(data, dtype);

    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let nanCount = 0;
    let infCount = 0;
    let nonZero = 0;

    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (Number.isNaN(v)) {
        nanCount++;
      } else if (!Number.isFinite(v)) {
        infCount++;
      } else {
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        if (v !== 0) nonZero++;
      }
    }

    const maxAbs = Math.max(Math.abs(min), Math.abs(max));
    const mean = sum / (arr.length - nanCount - infCount);
    const sample = Array.from(arr.slice(0, sampleCount));

    const shapeStr = shape ? `[${shape.join('x')}]` : `[${arr.length}]`;

    /** @type {import('./tensor.js').TensorStats} */
    const stats = {
      shape: shapeStr,
      dtype,
      min,
      max,
      maxAbs,
      mean,
      nonZero,
      total: arr.length,
      nanCount,
      infCount,
      sample,
    };

    // Format log message
    const tag = options.layerIdx !== undefined
      ? `[KERNEL][L${options.layerIdx}]`
      : '[KERNEL]';

    let msg = `${tag} ${label} ${shapeStr}`;
    msg += ` min=${min.toFixed(3)} max=${max.toFixed(3)} maxAbs=${maxAbs.toFixed(3)}`;
    msg += ` mean=${mean.toFixed(3)} nonZero=${nonZero}/${arr.length}`;

    if (nanCount > 0) msg += ` NaN=${nanCount}`;
    if (infCount > 0) msg += ` Inf=${infCount}`;

    msg += `\n  sample=[${sample.map(v => v.toFixed(4)).join(', ')}]`;

    // Warnings
    if (maxAbs > warnThreshold) {
      msg += `\n  ☡ VALUE EXPLOSION: maxAbs=${maxAbs.toFixed(1)} > ${warnThreshold}`;
    }
    if (nanCount > 0 || infCount > 0) {
      msg += `\n  ☡ NUMERICAL INSTABILITY: ${nanCount} NaN, ${infCount} Inf`;
    }
    if (nonZero === 0 && arr.length > 0) {
      msg += `\n  ☡ ALL ZEROS`;
    }

    log.debug('Debug', msg);
    return stats;
  } catch (e) {
    log.error('Kernel', `${label} ERROR: ${e}`);
    return null;
  }
}

/**
 * Dump stats for a single token row within a 2D [numTokens, rowSize] buffer.
 * Use this when matching per-token reference implementations (e.g., HuggingFace hooks).
 * @param {GPUBuffer} buffer
 * @param {string} label
 * @param {{ layerIdx?: number; tokenIdx: number; rowSize: number; dtype?: 'f32' | 'f16' | 'bf16'; sampleCount?: number; warnThreshold?: number }} options
 * @returns {Promise<import('./tensor.js').TensorStats | null>}
 */
export async function dumpTokenVector(buffer, label, options) {
  if (!isEnabled('kernel', options.layerIdx)) return null;

  const {
    tokenIdx,
    rowSize,
    dtype = 'f32',
    sampleCount = 8,
    warnThreshold = 10000,
  } = options;

  try {
    const data = await readBuffer(buffer);
    const arr = decodeReadback(data, dtype);

    const offset = tokenIdx * rowSize;
    const end = offset + rowSize;
    if (offset < 0 || end > arr.length) {
      log.error('Kernel', `${label} ERROR: token slice out of bounds (tokenIdx=${tokenIdx}, rowSize=${rowSize}, len=${arr.length})`);
      return null;
    }

    const row = arr.subarray(offset, end);

    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let nanCount = 0;
    let infCount = 0;
    let nonZero = 0;

    for (let i = 0; i < row.length; i++) {
      const v = row[i];
      if (Number.isNaN(v)) {
        nanCount++;
      } else if (!Number.isFinite(v)) {
        infCount++;
      } else {
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        if (v !== 0) nonZero++;
      }
    }

    const maxAbs = Math.max(Math.abs(min), Math.abs(max));
    const denom = row.length - nanCount - infCount;
    const mean = denom > 0 ? sum / denom : 0;
    const sample = Array.from(row.slice(0, sampleCount));

    const shapeStr = `[t${tokenIdx}x${rowSize}]`;

    /** @type {import('./tensor.js').TensorStats} */
    const stats = {
      shape: shapeStr,
      dtype,
      min,
      max,
      maxAbs,
      mean,
      nonZero,
      total: row.length,
      nanCount,
      infCount,
      sample,
    };

    const tag = options.layerIdx !== undefined
      ? `[KERNEL][L${options.layerIdx}]`
      : '[KERNEL]';

    let msg = `${tag} ${label} ${shapeStr}`;
    msg += ` min=${min.toFixed(3)} max=${max.toFixed(3)} maxAbs=${maxAbs.toFixed(3)}`;
    msg += ` mean=${mean.toFixed(3)} nonZero=${nonZero}/${row.length}`;

    if (nanCount > 0) msg += ` NaN=${nanCount}`;
    if (infCount > 0) msg += ` Inf=${infCount}`;

    msg += `\n  sample=[${sample.map(v => v.toFixed(4)).join(', ')}]`;

    if (maxAbs > warnThreshold) {
      msg += `\n  ☡ VALUE EXPLOSION: maxAbs=${maxAbs.toFixed(1)} > ${warnThreshold}`;
    }
    if (nanCount > 0 || infCount > 0) {
      msg += `\n  ☡ NUMERICAL INSTABILITY: ${nanCount} NaN, ${infCount} Inf`;
    }
    if (nonZero === 0 && row.length > 0) {
      msg += `\n  ☡ ALL ZEROS`;
    }

    log.debug('Debug', msg);
    return stats;
  } catch (e) {
    log.error('Kernel', `${label} ERROR: ${e}`);
    return null;
  }
}

/**
 * Log a kernel step with optional tensor dump.
 * Use this after kernel invocations to trace execution.
 *
 * @param {string} kernelName - Name of the kernel (e.g., 'matmul', 'rmsnorm')
 * @param {{ layerIdx?: number; M?: number; N?: number; K?: number; size?: number; label?: string }} info - Additional info to log
 * @returns {void}
 */
export function logKernelStep(kernelName, info) {
  if (!isEnabled('kernel', info.layerIdx)) return;

  const tag = info.layerIdx !== undefined
    ? `[KERNEL][L${info.layerIdx}]`
    : '[KERNEL]';

  let msg = `${tag} ${kernelName}`;
  if (info.label) msg += ` (${info.label})`;
  if (info.M !== undefined) msg += ` M=${info.M}`;
  if (info.N !== undefined) msg += ` N=${info.N}`;
  if (info.K !== undefined) msg += ` K=${info.K}`;
  if (info.size !== undefined) msg += ` size=${info.size}`;

  log.debug('Debug', msg);
}

/**
 * Dump KV cache state for a specific layer.
 * Reads both keys and values buffers and reports statistics.
 *
 * @param {import('../../kv-cache.js').KVCache | import('../../kv-cache.js').SlidingWindowKVCache} kvCache - KV cache instance
 * @param {number} layerIdx - Layer index to inspect
 * @returns {Promise<{ keys: import('./tensor.js').TensorStats | null; values: import('./tensor.js').TensorStats | null } | null>}
 */
export async function dumpKVCache(kvCache, layerIdx) {
  if (!isEnabled('kernel', layerIdx) && !isEnabled('kv', layerIdx)) return null;

  const tag = `[KV][L${layerIdx}]`;

  try {
    if (!kvCache?.hasGPUCache?.()) {
      log.debug('Debug', `${tag} No GPU cache available`);
      return null;
    }

    const gpuBuffers = kvCache.getGPUBuffers(layerIdx);
    if (!gpuBuffers) {
      log.debug('Debug', `${tag} No buffers for layer ${layerIdx}`);
      return null;
    }

    const { keysGPU, valuesGPU, seqLen } = gpuBuffers;
    const numHeads = kvCache.numHeads || 0;
    const headDim = kvCache.headDim || 0;

    log.debug('Debug', `${tag} seqLen=${seqLen} numHeads=${numHeads} headDim=${headDim}`);

    const kvDtype = kvCache.kvDtype === 'f16' ? 'f16' : 'f32';
    const keysStats = keysGPU
      ? await dumpTensor(keysGPU, 'K_cache', {
          layerIdx,
          shape: [seqLen, numHeads * headDim],
          dtype: kvDtype,
        })
      : null;

    const valuesStats = valuesGPU
      ? await dumpTensor(valuesGPU, 'V_cache', {
          layerIdx,
          shape: [seqLen, numHeads * headDim],
          dtype: kvDtype,
        })
      : null;

    return { keys: keysStats, values: valuesStats };
  } catch (e) {
    log.error('Debug', `${tag} ERROR: ${e}`);
    return null;
  }
}

/**
 * Check if kernel step debugging is enabled.
 * Use this to gate expensive debug operations.
 * @param {number} [layerIdx]
 * @returns {boolean}
 */
export function isKernelDebugEnabled(layerIdx) {
  return isEnabled('kernel', layerIdx);
}
