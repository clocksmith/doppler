import { getDevice, getKernelCapabilities } from '../device.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { GPU_LIMITS, WORKGROUP_SIZES } from './constants.js';
import {
  createPipeline,
  createUniformBufferWithView,
  getCachedPipeline,
  getKernelConfig,
  hasRequiredFeatures,
} from './utils.js';

function calculate2DDispatch(workgroups) {
  const maxWorkgroupsPerDim = GPU_LIMITS.MAX_WORKGROUPS;
  return workgroups <= maxWorkgroupsPerDim
    ? [workgroups, 1, 1]
    : [maxWorkgroupsPerDim, Math.ceil(workgroups / maxWorkgroupsPerDim), 1];
}

function assertNonNegativeInteger(value, label) {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`KV cache write requires ${label} to be a non-negative integer.`);
  }
}

function assertBufferCapacity(buffer, offset, elementCount, bytesPerElement, label) {
  const requiredBytes = (offset + elementCount) * bytesPerElement;
  if (!Number.isFinite(buffer?.size) || requiredBytes > buffer.size) {
    throw new Error(
      `KV cache write ${label} buffer is smaller than requested range ` +
      `(${requiredBytes} > ${buffer?.size ?? 'unknown'} bytes).`
    );
  }
}

async function executeKVCacheWriteF32ToF16(
  recorder,
  keys,
  values,
  outputKeys,
  outputValues,
  options = {}
) {
  const device = recorder?.device || getDevice();
  const srcOffset = options.srcOffset ?? 0;
  const dstOffset = options.dstOffset ?? 0;
  const elementCount = options.elementCount;

  assertNonNegativeInteger(srcOffset, 'srcOffset');
  assertNonNegativeInteger(dstOffset, 'dstOffset');
  assertNonNegativeInteger(elementCount, 'elementCount');
  if (elementCount === 0) {
    return;
  }

  assertBufferCapacity(keys, srcOffset, elementCount, 4, 'keys input');
  assertBufferCapacity(values, srcOffset, elementCount, 4, 'values input');
  assertBufferCapacity(outputKeys, dstOffset, elementCount, 2, 'keys output');
  assertBufferCapacity(outputValues, dstOffset, elementCount, 2, 'values output');

  const variant = 'f32_to_f16';
  const config = getKernelConfig('kv_cache_write', variant);
  const caps = getKernelCapabilities();
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`KV cache write kernel "${variant}" requires unsupported GPU features.`);
  }

  const pipeline = getCachedPipeline('kv_cache_write', variant)
    ?? await createPipeline('kv_cache_write', variant);
  const uniformBuffer = createUniformBufferWithView(
    'kv_cache_write_uniforms',
    16,
    (view) => {
      view.setUint32(0, srcOffset, true);
      view.setUint32(4, dstOffset, true);
      view.setUint32(8, elementCount, true);
      view.setUint32(12, 0, true);
    },
    recorder,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'kv_cache_write_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: keys } },
      { binding: 2, resource: { buffer: values } },
      { binding: 3, resource: { buffer: outputKeys } },
      { binding: 4, resource: { buffer: outputValues } },
    ],
  });

  const workgroups = Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT);
  const dispatchSize = calculate2DDispatch(workgroups);
  if (recorder) {
    recordDispatch(recorder, pipeline, bindGroup, dispatchSize, 'kv_cache_write:f32_to_f16');
  } else {
    try {
      dispatch(device, pipeline, bindGroup, dispatchSize, 'kv_cache_write:f32_to_f16');
    } finally {
      uniformBuffer.destroy();
    }
  }
}

export async function runKVCacheWriteF32ToF16(
  keys,
  values,
  outputKeys,
  outputValues,
  options = {}
) {
  return executeKVCacheWriteF32ToF16(null, keys, values, outputKeys, outputValues, options);
}

export async function recordKVCacheWriteF32ToF16(
  recorder,
  keys,
  values,
  outputKeys,
  outputValues,
  options = {}
) {
  return executeKVCacheWriteF32ToF16(recorder, keys, values, outputKeys, outputValues, options);
}
