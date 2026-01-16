

import { getKernelCapabilities } from '../gpu/device.js';
import { createTensor } from '../gpu/tensor.js';
import { castF32ToF16 } from '../gpu/kernel-selector.js';
import { releaseBuffer } from '../gpu/buffer-pool.js';
import {
  createWeightBuffer,
  isWeightBuffer,
  getWeightDtype,
  getLayout,
} from '../gpu/weight-buffer.js';
import { trace as debugTrace } from '../debug/index.js';
import { selectRuleValue } from '../rules/rule-registry.js';

// ============================================================================
// Main Downcast Function
// ============================================================================


export async function maybeDowncastToF16(buf, options) {
  if (!buf) return null;

  const caps = getKernelCapabilities();
  if (!caps.hasF16) {
    // No F16 support - return as-is
    return {
      buffer: buf,
      wasDowncast: false,
      newBuffer: null,
    };
  }

  if (options.keepF32) {
    const layerStr = options.layerIdx !== undefined ? `Layer ${options.layerIdx}` : '';
    debugTrace.loader(`${layerStr} keeping ${options.label} in f32 (keepF32Weights=true)`);
    return {
      buffer: buf,
      wasDowncast: false,
      newBuffer: null,
    };
  }

  // Handle WeightBuffer
  if (isWeightBuffer(buf)) {
    return downcastWeightBuffer(buf, options);
  }

  // Handle raw GPUBuffer
  if (buf instanceof GPUBuffer) {
    return downcastGPUBuffer(buf, options);
  }

  // Unsupported type
  return {
    buffer: buf,
    wasDowncast: false,
    newBuffer: null,
  };
}

// ============================================================================
// Internal Helpers
// ============================================================================


async function downcastWeightBuffer(buf, options) {
  const dtype = getWeightDtype(buf);
  if (dtype !== 'f32') {
    // Already F16 or other dtype
    return {
      buffer: buf,
      wasDowncast: false,
      newBuffer: null,
    };
  }

  const elems = buf.buffer.size / 4;
  const wasColumnMajor = getLayout(buf) === 'column';
  const layerStr = options.layerIdx !== undefined ? `Layer ${options.layerIdx}` : '';

  debugTrace.loader(
    `${layerStr} downcasting WeightBuffer ${options.label}: ` +
    `bufSize=${buf.buffer.size}, elems=${elems}, columnMajor=${wasColumnMajor}`
  );

  try {
    const inputTensor = createTensor(buf.buffer, 'f32', [elems], `${options.label}_f32`);
    const f16Tensor = await castF32ToF16(inputTensor);

    // Create new WeightBuffer with f16 dtype, preserving layout
    const layout = selectRuleValue('loader', 'weights', 'weightLayout', {
      layout: options.layout ?? null,
      useColumnWise: wasColumnMajor,
    });
    const shape = options.shape ??  (buf.shape);
    const newWeightBuffer = createWeightBuffer(
      f16Tensor.buffer,
      'f16',
      layout,
      shape,
      buf.label ?? options.label
    );

    debugTrace.loader(`${layerStr} ${options.label} downcast result: f16Size=${f16Tensor.buffer.size}`);

    // Release old buffer
    releaseBuffer(buf.buffer);

    return {
      buffer: newWeightBuffer,
      wasDowncast: true,
      newBuffer: f16Tensor.buffer,
    };
  } catch (e) {
    debugTrace.loader(`Failed to downcast ${options.label} to f16: ${ (e).message}`);
    return {
      buffer: buf,
      wasDowncast: false,
      newBuffer: null,
    };
  }
}


async function downcastGPUBuffer(buf, options) {
  const dtype = getWeightDtype(buf) || 'f32';
  if (dtype !== 'f32') {
    // Already F16 or other dtype
    return {
      buffer: buf,
      wasDowncast: false,
      newBuffer: null,
    };
  }

  const elems = buf.size / 4;
  const wasColumnMajor = getLayout(buf) === 'column';
  const layerStr = options.layerIdx !== undefined ? `Layer ${options.layerIdx}` : '';

  debugTrace.loader(
    `${layerStr} downcasting ${options.label}: ` +
    `bufSize=${buf.size}, elems=${elems}, expectedF16=${elems * 2}, columnMajor=${wasColumnMajor}`
  );

  try {
    const inputTensor = createTensor(buf, 'f32', [elems], `${options.label}_f32`);
    const f16Tensor = await castF32ToF16(inputTensor);

    // Create WeightBuffer with f16 dtype, preserving layout
    const layout = selectRuleValue('loader', 'weights', 'weightLayout', {
      layout: options.layout ?? null,
      useColumnWise: wasColumnMajor,
    });
    const shape = options.shape ?? [elems];
    const newWeightBuffer = createWeightBuffer(
      f16Tensor.buffer,
      'f16',
      layout,
      shape,
      options.label
    );

    debugTrace.loader(`${layerStr} ${options.label} downcast result: f16Size=${f16Tensor.buffer.size}`);

    // Release old buffer
    releaseBuffer(buf);

    return {
      buffer: newWeightBuffer,
      wasDowncast: true,
      newBuffer: f16Tensor.buffer,
    };
  } catch (e) {
    debugTrace.loader(`Failed to downcast ${options.label} to f16: ${ (e).message}`);
    return {
      buffer: buf,
      wasDowncast: false,
      newBuffer: null,
    };
  }
}

// ============================================================================
// Batch Downcast Helper
// ============================================================================


export async function batchDowncastWeights(weights, keys, options, gpuBuffers) {
  for (const key of keys) {
    const buf = weights[key];
    if (!buf) continue;

    const result = await maybeDowncastToF16( (buf), {
      ...options,
      label: String(key),
    });

    if (result?.wasDowncast) {
       (weights)[ (key)] = result.buffer;
      if (result.newBuffer) {
        gpuBuffers.add(result.newBuffer);
      }
    }
  }
}
