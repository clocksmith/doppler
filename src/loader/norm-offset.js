/**
 * Norm Offset - Apply +1 offset to RMSNorm weights.
 *
 * Gemma 3+ models use (1 + weight) formula for RMSNorm instead of
 * just weight. This module handles the transformation.
 *
 * @module loader/norm-offset
 */

import { getDevice } from '../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../gpu/buffer-pool.js';
import { f16ToF32 } from './dtype-utils.js';
import { log, trace as debugTrace } from '../debug/index.js';

// ============================================================================
// Main Function
// ============================================================================

/**
 * Apply +1 offset to norm weights for Gemma 3+ models.
 *
 * Transforms weight values from `w` to `1 + w`.
 *
 * IMPORTANT: actualNumElements must be provided to avoid reading garbage
 * padding from the buffer pool's power-of-2 bucketing.
 *
 * @param {GPUBuffer | Float32Array} tensor - Input tensor (GPUBuffer or Float32Array)
 * @param {import('./norm-offset.js').NormOffsetOptions} [options={}] - Offset options
 * @returns {Promise<import('./norm-offset.js').NormOffsetResult>} Transformed tensor result
 */
export async function applyNormWeightOffset(tensor, options = {}) {
  const {
    actualNumElements,
    bufferDtype = 'f32',
    enableDebugLog = false,
  } = options;

  const device = getDevice();
  if (!device) {
    log.warn('Loader', 'No GPU device for norm offset');
    return { tensor, debugLogged: false };
  }

  if (tensor instanceof GPUBuffer) {
    return applyOffsetToGPUBuffer(tensor, {
      actualNumElements,
      bufferDtype,
      enableDebugLog,
      device,
    });
  }

  if (tensor instanceof Float32Array) {
    return applyOffsetToFloat32Array(tensor, {
      actualNumElements,
      enableDebugLog,
      device,
    });
  }

  log.warn('Loader', 'Unknown tensor type for norm offset');
  return { tensor, debugLogged: false };
}

// ============================================================================
// GPU Buffer Path
// ============================================================================

/**
 * @param {GPUBuffer} tensor
 * @param {{ actualNumElements?: number; bufferDtype: 'f16' | 'f32' | 'bf16'; enableDebugLog: boolean; device: GPUDevice }} options
 * @returns {Promise<import('./norm-offset.js').NormOffsetResult>}
 */
async function applyOffsetToGPUBuffer(tensor, options) {
  const { actualNumElements, bufferDtype, enableDebugLog, device } = options;

  // Use provided dtype to determine element size (norm weights default to f32)
  const isF16 = bufferDtype === 'f16' || bufferDtype === 'bf16';
  const bytesPerElement = isF16 ? 2 : 4;

  // Use actual element count if provided, otherwise infer from buffer size
  const numElements = actualNumElements ?? Math.floor(tensor.size / bytesPerElement);
  const dataSize = numElements * bytesPerElement;

  // Ensure we don't read past the buffer
  const readSize = Math.min(dataSize, tensor.size);

  // Create staging buffer for readback
  const stagingBuffer = device.createBuffer({
    size: readSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(tensor, 0, stagingBuffer, 0, readSize);
  device.queue.submit([encoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const rawData = stagingBuffer.getMappedRange().slice(0);
  stagingBuffer.unmap();
  stagingBuffer.destroy();

  // Convert to F32 for offset calculation
  /** @type {Float32Array} */
  let data;
  if (isF16) {
    const u16Data = new Uint16Array(rawData);
    data = new Float32Array(u16Data.length);
    if (bufferDtype === 'bf16') {
      // BF16 to F32 conversion
      const tmp = new ArrayBuffer(4);
      const u32View = new Uint32Array(tmp);
      const f32View = new Float32Array(tmp);
      for (let i = 0; i < u16Data.length; i++) {
        u32View[0] = u16Data[i] << 16;
        data[i] = f32View[0];
      }
    } else {
      // F16 to F32 conversion
      for (let i = 0; i < u16Data.length; i++) {
        data[i] = f16ToF32(u16Data[i]);
      }
    }
  } else {
    data = new Float32Array(rawData);
  }

  // Apply +1 offset
  const offsetData = new Float32Array(numElements);
  for (let i = 0; i < numElements; i++) {
    offsetData[i] = 1.0 + data[i];
  }

  // Debug logging
  let debugLogged = false;
  if (enableDebugLog) {
    const sampleSize = Math.min(256, numElements);
    const beforeMin = Math.min(...Array.from(data.slice(0, sampleSize)));
    const beforeMax = Math.max(...Array.from(data.slice(0, sampleSize)));
    const afterMin = Math.min(...Array.from(offsetData.slice(0, sampleSize)));
    const afterMax = Math.max(...Array.from(offsetData.slice(0, sampleSize)));
    debugTrace.loader(
      `Norm +1 offset: before=[${beforeMin.toFixed(3)}, ${beforeMax.toFixed(3)}] ` +
      `after=[${afterMin.toFixed(3)}, ${afterMax.toFixed(3)}]`
    );
    debugLogged = true;
  }

  // Release old buffer and create new one
  releaseBuffer(tensor);
  const newBuffer = acquireBuffer(offsetData.byteLength, undefined, 'norm_offset');
  device.queue.writeBuffer(newBuffer, 0, offsetData);

  return { tensor: newBuffer, debugLogged };
}

// ============================================================================
// Float32Array Path
// ============================================================================

/**
 * @param {Float32Array} tensor
 * @param {{ actualNumElements?: number; enableDebugLog: boolean; device: GPUDevice }} options
 * @returns {Promise<import('./norm-offset.js').NormOffsetResult>}
 */
async function applyOffsetToFloat32Array(tensor, options) {
  const { actualNumElements, enableDebugLog, device } = options;

  const numElements = actualNumElements ?? tensor.length;
  const offsetData = new Float32Array(numElements);
  for (let i = 0; i < numElements; i++) {
    offsetData[i] = 1.0 + tensor[i];
  }

  // Debug logging
  let debugLogged = false;
  if (enableDebugLog) {
    const sampleSize = Math.min(256, numElements);
    const beforeMin = Math.min(...Array.from(tensor.slice(0, sampleSize)));
    const beforeMax = Math.max(...Array.from(tensor.slice(0, sampleSize)));
    const afterMin = Math.min(...Array.from(offsetData.slice(0, sampleSize)));
    const afterMax = Math.max(...Array.from(offsetData.slice(0, sampleSize)));
    debugTrace.loader(
      `Norm +1 offset (CPU): before=[${beforeMin.toFixed(3)}, ${beforeMax.toFixed(3)}] ` +
      `after=[${afterMin.toFixed(3)}, ${afterMax.toFixed(3)}]`
    );
    debugLogged = true;
  }

  // Always upload to GPU to prevent double-offset in pipeline
  // Pipeline's getNormWeightBuffer returns GPUBuffer as-is, skipping offset
  const newBuffer = acquireBuffer(offsetData.byteLength, undefined, 'norm_offset');
  device.queue.writeBuffer(newBuffer, 0, offsetData);

  return { tensor: newBuffer, debugLogged };
}
