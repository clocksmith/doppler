import { getSharedDeviceState } from './device-state.js';

const sharedDeviceState = getSharedDeviceState();

/** @param {unknown} value */
function isValidGPUBuffer(value) {
  if (!value || typeof value !== 'object') {
    return false;
  }
  if ('__dopplerFakeGPUBuffer' in value && value.__dopplerFakeGPUBuffer === true) {
    return true;
  }
  if (
    typeof value === 'object'
    && value.constructor?.name === 'FakeBuffer'
    && 'size' in value && typeof value.size === 'number'
    && 'usage' in value && typeof value.usage === 'number'
    && 'destroy' in value && typeof value.destroy === 'function'
  ) {
    return true;
  }
  if (typeof GPUBuffer === 'undefined') {
    return true;
  }
  return value instanceof GPUBuffer;
}

/** @param {unknown} value */
function describeBindGroupBufferValue(value) {
  if (value === null) return 'null (explicitly set to null)';
  if (value === undefined) return 'undefined (missing or never assigned)';
  if (typeof GPUBuffer !== 'undefined' && value instanceof GPUBuffer) return 'GPUBuffer';
  if (typeof value === 'object') {
    return value.constructor?.name || 'object';
  }
  return typeof value;
}

/** @param {GPUBindGroupDescriptor} descriptor @param {GPUDevice} device */
function validateBindGroupDescriptor(descriptor, device) {
  const label = descriptor?.label || 'unlabeled_bind_group';
  const entries = Array.isArray(descriptor?.entries) ? descriptor.entries : [];
  for (const entry of entries) {
    const resource = entry?.resource;
    if (!resource || typeof resource !== 'object' || !('buffer' in resource)) {
      continue;
    }
    if (!isValidGPUBuffer(resource.buffer)) {
      throw new Error(
        `[${label}] binding ${entry.binding} requires a GPUBuffer; ` +
        `got ${describeBindGroupBufferValue(resource.buffer)}.`
      );
    }
    const owner = sharedDeviceState.bufferOwners.get(resource.buffer);
    if (owner && owner !== device) {
      throw new Error(
        `[${label}] binding ${entry.binding} uses a GPUBuffer created by a different GPUDevice.`
      );
    }
  }
}

function shouldValidateBindGroupDescriptors() {
  const env = typeof process !== 'undefined' ? process?.env?.DOPPLER_VALIDATE_BIND_GROUPS : null;
  if (env === '1' || env === 'true') {
    return true;
  }
  return false;
}

/** @type {import('./device-bindings.js').wrapDeviceCreateBindGroup} */
export function wrapDeviceCreateBindGroup(device) {
  if (!device || device.__dopplerBindGroupValidationWrapped) {
    return device;
  }
  const originalCreateBindGroup = device.createBindGroup.bind(device);
  device.createBindGroup = (descriptor) => {
    if (shouldValidateBindGroupDescriptors()) {
      validateBindGroupDescriptor(descriptor, device);
    } else {
      const entries = Array.isArray(descriptor?.entries) ? descriptor.entries : [];
      for (const entry of entries) {
        const resource = entry?.resource;
        const buffer = resource && typeof resource === 'object' && 'buffer' in resource ? resource.buffer : null;
        if (buffer && sharedDeviceState.bufferOwners.get(buffer) !== undefined) {
          validateBindGroupDescriptor(descriptor, device);
          break;
        }
      }
    }
    return originalCreateBindGroup(descriptor);
  };
  Object.defineProperty(device, '__dopplerBindGroupValidationWrapped', {
    value: true,
    configurable: true,
    enumerable: false,
    writable: false,
  });
  return device;
}
