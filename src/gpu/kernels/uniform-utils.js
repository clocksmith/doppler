

import { getDevice } from '../device.js';
import { getUniformCache } from '../uniform-cache.js';

// ============================================================================
// Uniform Buffer Creation
// ============================================================================

export function writeUniformsFromObject(view, config, values) {
  const uniforms = config?.uniforms;
  if (!uniforms) {
    const op = config?.operation ?? 'unknown';
    const variant = config?.variant ?? 'unknown';
    throw new Error(`Kernel "${op}/${variant}" has no uniforms defined in registry.`);
  }

  for (const field of uniforms.fields) {
    const value = values[field.name];
    if (value === undefined) {
      // Optional fields or internal padding can be 0
      continue;
    }

    switch (field.type) {
      case 'u32':
        view.setUint32(field.offset, value, true);
        break;
      case 'i32':
        view.setInt32(field.offset, value, true);
        break;
      case 'f32':
        view.setFloat32(field.offset, value, true);
        break;
      default:
        throw new Error(`Unsupported uniform type "${field.type}" for field "${field.name}" in op "${opName}"`);
    }
  }
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

  // Convert ArrayBufferView to ArrayBuffer for caching
  const arrayBuffer = data instanceof ArrayBuffer
    ? data
    : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);

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
  device.queue.writeBuffer(buffer, 0, arrayBuffer);
  return buffer;
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
