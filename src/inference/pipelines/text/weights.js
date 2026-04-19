

import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { log } from '../../../debug/index.js';
import { isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, tagBufferDtype } from '../../../gpu/weight-buffer.js';

// ============================================================================
// Type Guards
// ============================================================================


function isLayerWeights(value) {
  return value !== null && typeof value === 'object' && !ArrayBuffer.isView(value) && !('getMappedRange' in  (value)) && !isWeightBuffer(value) && !isCpuWeightBuffer(value);
}


export function getLayerWeights(weights, key) {
  const value = weights.get(key);
  if (value && isLayerWeights(value)) return value;
  return null;
}

// ============================================================================
// Weight Buffer Creation
// ============================================================================


export function getWeightBuffer(weight, label) {
  // Preserve WeightBuffer to maintain dtype/layout for matmul
  if (isWeightBuffer(weight)) {
    return weight;
  }
  if (isGpuBufferInstance(weight)) {
    return weight;
  }

  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for weight buffer creation');
  }

  
  let data;
  let bufferDtype = 'f32';
  if (isCpuWeightBuffer(weight)) {
    data = weight.data;
    bufferDtype = weight.dtype ?? 'f32';
  } else if (weight instanceof Float32Array) {
    data = weight;
  } else {
    data = new Float32Array( (weight));
  }

  const buf = acquireBuffer(data.byteLength, undefined, label);
  try {
    device.queue.writeBuffer(buf, 0,  ( (data)));
    tagBufferDtype(buf, bufferDtype);
    return buf;
  } catch (error) {
    releaseBuffer(buf);
    throw error;
  }
}


export function getNormWeightBuffer(weight, label, config, debugFlags) {
  // Debug: Log whether weight is GPUBuffer (first time only)
  if (debugFlags && !debugFlags.normBufferTypeLogged) {
    debugFlags.normBufferTypeLogged = true;
    log.debug('Weights', `getNormWeightBuffer: weight is GPUBuffer=${isGpuBufferInstance(weight)}, label=${label}`);
  }

  if (isWeightBuffer(weight)) {
    return weight.buffer;
  }

  if (isGpuBufferInstance(weight)) {
    // If already a GPUBuffer, we can't modify it - assume it was preprocessed
    return weight;
  }

  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for norm weight buffer creation');
  }

  // RMSNorm weight offset is handled in the kernel, so upload raw weights as-is.

  // Standard path: just copy to GPU
  
  let data;
  if (isCpuWeightBuffer(weight)) {
    data = weight.data;
  } else if (weight instanceof Float32Array) {
    data = weight;
  } else if ('buffer' in weight && 'byteOffset' in weight && 'byteLength' in weight) {
    data = new Float32Array(weight.buffer, weight.byteOffset, weight.byteLength / 4);
  } else {
    data = new Float32Array( (weight));
  }

  const buf = acquireBuffer(data.byteLength, undefined, label);
  try {
    device.queue.writeBuffer(buf, 0,  ( (data)));
    tagBufferDtype(buf, 'f32');
    return buf;
  } catch (error) {
    releaseBuffer(buf);
    throw error;
  }
}


function getGPUWeightBuffer(weight, label) {
  // Handle WeightBuffer by extracting underlying GPUBuffer
  if (isWeightBuffer(weight)) {
    return weight.buffer;
  }
  if (isGpuBufferInstance(weight)) {
    return weight;
  }
  // Weight not on GPU - this shouldn't happen if loader is working correctly
  log.warn('Weights', `Weight ${label} not on GPU, uploading`);
  // At this point weight is Float32Array or ArrayBuffer, so getWeightBuffer returns GPUBuffer
  return  (getWeightBuffer(weight, label));
}

