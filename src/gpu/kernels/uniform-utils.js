

import { UNIFORM_WRITERS } from './generated/uniform-writers.js';
import { getKernelConfig } from '../../config/kernel-registry-contract.js';
import { getDevice } from '../device.js';
import { getUniformCache, toUniformArrayBuffer } from '../uniform-cache.js';

// ============================================================================
// Uniform Buffer Creation
// ============================================================================

function getUniformWriter(config) {
  const kernel = `${config?.operation}/${config?.variant}`;
  const writer = UNIFORM_WRITERS[kernel];
  if (!writer) throw new Error(`Kernel "${kernel}" has no generated uniform layout.`);
  if (config.uniforms !== getKernelConfig(config.operation, config.variant).uniforms) {
    throw new Error(`Kernel "${kernel}" must use its immutable registry uniform layout.`);
  }
  return { kernel, writer };
}

export function writeUniformsFromObject(view, config, values) {
  const { kernel, writer } = getUniformWriter(config);
  if (!(view instanceof DataView) || view.byteLength < writer.size) {
    throw new Error(`Kernel "${kernel}" requires a ${writer.size}-byte uniform view.`);
  }
  if (!values || typeof values !== 'object') throw new Error(`Kernel "${kernel}" requires uniform values.`);
  writer.write(view, values, kernel);
}

export function getUniformByteLength(config) {
  if (config?.uniforms === null) return 0;
  return getUniformWriter(config).writer.size;
}

export function createKernelUniformBuffer(label, config, values, recorder, deviceOverride) {
  return createUniformBufferWithView(
    label,
    getUniformByteLength(config),
    (view) => writeUniformsFromObject(view, config, values),
    recorder,
    deviceOverride
  );
}


export function createUniformBufferFromData(
  label,
  data,
  recorder,
  deviceOverride,
  options
) {
  if (recorder) {
    return recorder.createUniformBuffer(data, label);
  }

  const arrayBuffer = toUniformArrayBuffer(data);

  // Use cache by default for non-recorder paths
  const useCache = options?.useCache ?? true;
  if (useCache && !deviceOverride) {
    return getUniformCache().getOrCreate(arrayBuffer, label);
  }

  // Fallback to direct creation (for custom device or explicit no-cache)
  const device = deviceOverride ?? getDevice();
  if (!device) {
    throw new Error('GPU device not initialized');
  }

  const byteLength = arrayBuffer.byteLength;
  const buffer = device.createBuffer({
    label,
    size: byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  try {
    device.queue.writeBuffer(buffer, 0, arrayBuffer);
    return buffer;
  } catch (error) {
    buffer.destroy();
    throw error;
  }
}


export function createUniformBufferWithView(
  label,
  byteLength,
  writer,
  recorder,
  deviceOverride
) {
  const data = new ArrayBuffer(byteLength);
  const view = new DataView(data);
  writer(view);
  return createUniformBufferFromData(label, data, recorder, deviceOverride);
}
