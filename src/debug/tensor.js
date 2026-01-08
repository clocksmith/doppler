/**
 * DOPPLER Debug Module - Tensor Inspection Utilities
 *
 * Tools for inspecting GPU and CPU tensors, checking health, and comparing values.
 *
 * @module debug/tensor
 */

import { gpuDevice } from './config.js';
import { log } from './log.js';

// ============================================================================
// Internal Helpers
// ============================================================================

/**
 * F16 to F32 conversion helper.
 */
function f16ToF32(h) {
  const sign = (h >> 15) & 0x1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;

  if (exp === 0) {
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024);
  } else if (exp === 31) {
    return mant === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }

  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
}

// ============================================================================
// Tensor Inspection Interface
// ============================================================================

/**
 * Tensor inspection utilities.
 */
export const tensor = {
  /**
   * Inspect a GPU or CPU tensor and log statistics.
   */
  async inspect(
    buffer,
    label,
    options = {}
  ) {
    const { shape = [], maxPrint = 8, checkNaN = true } = options;

    let data;
    let isGPU = false;

    // Handle GPU buffers
    if (buffer && typeof buffer.mapAsync === 'function') {
      const gpuBuffer = buffer;
      await gpuBuffer.mapAsync(GPUMapMode.READ);
      data = new Float32Array(gpuBuffer.getMappedRange().slice(0));
      gpuBuffer.unmap();
    } else if (buffer && buffer.size !== undefined && gpuDevice) {
      isGPU = true;
      const gpuBuffer = buffer;
      const readSize = Math.min(gpuBuffer.size, 4096);
      const staging = gpuDevice.createBuffer({
        label: `debug_staging_${label}`,
        size: readSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      const encoder = gpuDevice.createCommandEncoder();
      encoder.copyBufferToBuffer(gpuBuffer, 0, staging, 0, readSize);
      gpuDevice.queue.submit([encoder.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      data = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
    } else if (buffer instanceof Float32Array || buffer instanceof Float64Array) {
      data = buffer instanceof Float32Array ? buffer : new Float32Array(buffer);
    } else if (buffer instanceof Uint16Array) {
      data = new Float32Array(buffer.length);
      for (let i = 0; i < buffer.length; i++) {
        data[i] = f16ToF32(buffer[i]);
      }
    } else {
      log.warn('Debug', `Cannot inspect tensor "${label}": unknown type`);
      return null;
    }

    // Compute statistics
    let min = Infinity,
      max = -Infinity,
      sum = 0,
      sumSq = 0;
    let nanCount = 0,
      infCount = 0,
      zeroCount = 0;

    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (Number.isNaN(v)) {
        nanCount++;
        continue;
      }
      if (!Number.isFinite(v)) {
        infCount++;
        continue;
      }
      if (v === 0) zeroCount++;
      min = Math.min(min, v);
      max = Math.max(max, v);
      sum += v;
      sumSq += v * v;
    }

    const validCount = data.length - nanCount - infCount;
    const mean = validCount > 0 ? sum / validCount : 0;
    const variance = validCount > 0 ? sumSq / validCount - mean * mean : 0;
    const std = Math.sqrt(Math.max(0, variance));

    const stats = {
      label,
      shape,
      size: data.length,
      isGPU,
      min,
      max,
      mean,
      std,
      nanCount,
      infCount,
      zeroCount,
      zeroPercent: ((zeroCount / data.length) * 100).toFixed(1),
      first: Array.from(data.slice(0, maxPrint)).map((v) => v.toFixed(4)),
      last: Array.from(data.slice(-maxPrint)).map((v) => v.toFixed(4)),
    };

    const shapeStr = shape.length > 0 ? `[${shape.join('x')}]` : `[${data.length}]`;
    log.debug(
      'Tensor',
      `${label} ${shapeStr}: min=${min.toFixed(4)}, max=${max.toFixed(4)}, mean=${mean.toFixed(4)}, std=${std.toFixed(4)}`
    );

    if (checkNaN && (nanCount > 0 || infCount > 0)) {
      log.warn('Tensor', `${label} has ${nanCount} NaN and ${infCount} Inf values!`);
    }

    return stats;
  },

  /**
   * Compare two tensors element-wise.
   */
  compare(
    a,
    b,
    label,
    tolerance = 1e-5
  ) {
    if (a.length !== b.length) {
      log.error('Tensor', `${label}: size mismatch ${a.length} vs ${b.length}`);
      return { label, match: false, error: 'size_mismatch', maxDiff: 0, maxDiffIdx: 0, avgDiff: 0, mismatchCount: 0, mismatchPercent: '0' };
    }

    let maxDiff = 0,
      maxDiffIdx = 0;
    let sumDiff = 0;
    let mismatchCount = 0;

    for (let i = 0; i < a.length; i++) {
      const diff = Math.abs(a[i] - b[i]);
      sumDiff += diff;
      if (diff > maxDiff) {
        maxDiff = diff;
        maxDiffIdx = i;
      }
      if (diff > tolerance) {
        mismatchCount++;
      }
    }

    const avgDiff = sumDiff / a.length;
    const match = mismatchCount === 0;

    const result = {
      label,
      match,
      maxDiff,
      maxDiffIdx,
      avgDiff,
      mismatchCount,
      mismatchPercent: ((mismatchCount / a.length) * 100).toFixed(2),
    };

    if (match) {
      log.debug('Tensor', `${label}: MATCH (maxDiff=${maxDiff.toExponential(2)})`);
    } else {
      log.warn(
        'Tensor',
        `${label}: MISMATCH ${mismatchCount}/${a.length} (${result.mismatchPercent}%) maxDiff=${maxDiff.toFixed(6)} at idx=${maxDiffIdx}`
      );
    }

    return result;
  },

  /**
   * Check tensor for common issues.
   */
  healthCheck(data, label) {
    const issues = [];

    const allZero = data.every((v) => v === 0);
    if (allZero) {
      issues.push('ALL_ZEROS');
    }

    const hasNaN = data.some((v) => Number.isNaN(v));
    const hasInf = data.some((v) => !Number.isFinite(v) && !Number.isNaN(v));
    if (hasNaN) issues.push('HAS_NAN');
    if (hasInf) issues.push('HAS_INF');

    const maxAbs = Math.max(...Array.from(data).map(Math.abs).filter(Number.isFinite));
    if (maxAbs > 1e6) issues.push(`EXTREME_VALUES (max=${maxAbs.toExponential(2)})`);

    const tinyCount = data.filter((v) => Math.abs(v) > 0 && Math.abs(v) < 1e-30).length;
    if (tinyCount > data.length * 0.1) {
      issues.push(`POTENTIAL_UNDERFLOW (${tinyCount} tiny values)`);
    }

    const healthy = issues.length === 0;

    if (healthy) {
      log.debug('Tensor', `${label}: healthy`);
    } else {
      log.warn('Tensor', `${label}: issues found - ${issues.join(', ')}`);
    }

    return { label, healthy, issues };
  },
};
