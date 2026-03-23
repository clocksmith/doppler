

import { wrapQueueForTracking, setTrackSubmits } from './submit-tracker.js';
import { log } from '../debug/index.js';
import { createDopplerError, ERROR_CODES } from '../errors/index.js';
import { GB } from '../config/schema/index.js';

// Re-export submit tracker for convenience
export { setTrackSubmits };

// Cached device and capabilities

let gpuDevice = null;

let kernelCapabilities = null;

// Cached platform config (set during initDevice)

let resolvedPlatformConfig = null;

// Track whether platform/registry initialization has been attempted
let platformInitialized = false;

// Monotonic device epoch. Increments whenever device-scoped GPU objects become invalid.
let deviceEpoch = 0;

function advanceDeviceEpoch() {
  deviceEpoch += 1;
}

function clearActiveDeviceState() {
  gpuDevice = null;
  kernelCapabilities = null;
  resolvedPlatformConfig = null;
  platformInitialized = false;
}

function isValidGPUBuffer(value) {
  if (!value) {
    return false;
  }
  if (value.__dopplerFakeGPUBuffer === true) {
    return true;
  }
  if (
    typeof value === 'object'
    && value.constructor?.name === 'FakeBuffer'
    && typeof value.size === 'number'
    && typeof value.usage === 'number'
    && typeof value.destroy === 'function'
  ) {
    return true;
  }
  if (typeof GPUBuffer === 'undefined') {
    return true;
  }
  return value instanceof GPUBuffer;
}

function isUsableGPUDevice(device) {
  return !!(
    device
    && typeof device.createBuffer === 'function'
    && typeof device.createBindGroup === 'function'
    && typeof device.createCommandEncoder === 'function'
    && typeof device.createShaderModule === 'function'
    && device.queue
    && typeof device.queue.submit === 'function'
  );
}

function describeBindGroupBufferValue(value) {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';
  if (typeof GPUBuffer !== 'undefined' && value instanceof GPUBuffer) return 'GPUBuffer';
  if (typeof value === 'object') {
    return value.constructor?.name || 'object';
  }
  return typeof value;
}

function validateBindGroupDescriptor(descriptor) {
  const label = descriptor?.label || 'unlabeled_bind_group';
  const entries = Array.isArray(descriptor?.entries) ? descriptor.entries : [];
  for (const entry of entries) {
    const resource = entry?.resource;
    if (!resource || typeof resource !== 'object' || !('buffer' in resource)) {
      continue;
    }
    if (isValidGPUBuffer(resource.buffer)) {
      continue;
    }
    throw new Error(
      `[${label}] binding ${entry.binding} requires a GPUBuffer; ` +
      `got ${describeBindGroupBufferValue(resource.buffer)}.`
    );
  }
}

function wrapDeviceCreateBindGroup(device) {
  if (!device || device.__dopplerBindGroupValidationWrapped) {
    return device;
  }
  const originalCreateBindGroup = device.createBindGroup.bind(device);
  device.createBindGroup = (descriptor) => {
    validateBindGroupDescriptor(descriptor);
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

function registerDeviceLostHandler(device) {
  if (!device || device.__dopplerLossHandlerRegistered) {
    return device;
  }

  if (device.lost && typeof device.lost.then === 'function') {
    const trackedDevice = device;
    device.lost.then((info) => {
      if (gpuDevice !== trackedDevice) {
        return;
      }
      log.error('GPU', 'Device lost: ' + info.message + ', Reason: ' + info.reason);
      clearActiveDeviceState();
      advanceDeviceEpoch();
    }).catch((error) => {
      if (gpuDevice !== trackedDevice) {
        return;
      }
      log.warn('GPU', 'Device lost handler failed: ' + (error?.message ?? error));
      clearActiveDeviceState();
      advanceDeviceEpoch();
    });
  }

  Object.defineProperty(device, '__dopplerLossHandlerRegistered', {
    value: true,
    configurable: true,
    enumerable: false,
    writable: false,
  });
  return device;
}


export const FEATURES =  ({
  SHADER_F16: 'shader-f16',
  SUBGROUPS: 'subgroups',
  SUBGROUPS_F16: 'subgroups-f16',
  TIMESTAMP_QUERY: 'timestamp-query',
});


export function isWebGPUAvailable() {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}


async function requestAdapter(options = {}) {
  if (!isWebGPUAvailable()) {
    return null;
  }

  // Try high-performance first, then fallback
  
  const adapterOptions = [
    { powerPreference: 'high-performance', ...options },
    { powerPreference: 'low-power', ...options },
    { ...options }, // Default
  ];

  for (const opts of adapterOptions) {
    try {
      const adapter = await navigator.gpu.requestAdapter(opts);
      if (adapter) {
        return adapter;
      }
    } catch (e) {
      // Continue to next option
    }
  }

  return null;
}


function detectFeatures(adapter) {
  const available = new Set();

  for (const feature of adapter.features) {
    available.add(feature);
  }

  return available;
}


function buildFeatureRequests(available) {
  
  const requested = [];

  // Request shader-f16 for FP16 matmul kernels
  if (available.has(FEATURES.SHADER_F16)) {
    requested.push( (FEATURES.SHADER_F16));
  }

  // Request subgroups for efficient dequantization
  if (available.has(FEATURES.SUBGROUPS)) {
    requested.push( (FEATURES.SUBGROUPS));
  }

  // Request subgroups-f16 if available (for combined f16 + subgroup ops)
  if (available.has(FEATURES.SUBGROUPS_F16)) {
    requested.push( (FEATURES.SUBGROUPS_F16));
  }

  // Request timestamp query for profiling (optional)
  if (available.has(FEATURES.TIMESTAMP_QUERY)) {
    requested.push( (FEATURES.TIMESTAMP_QUERY));
  }

  return requested;
}


function buildLimits(adapter) {
  const adapterLimits = adapter.limits;

  return {
    // Request maximum available storage buffer size (critical for large weight tensors)
    maxStorageBufferBindingSize: adapterLimits.maxStorageBufferBindingSize,
    // Request maximum buffer size
    maxBufferSize: adapterLimits.maxBufferSize,
    // Request maximum compute workgroup sizes
    maxComputeWorkgroupSizeX: adapterLimits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: adapterLimits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: adapterLimits.maxComputeWorkgroupSizeZ,
    maxComputeInvocationsPerWorkgroup: adapterLimits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupStorageSize: adapterLimits.maxComputeWorkgroupStorageSize,
    // Binding limits
    maxStorageBuffersPerShaderStage: adapterLimits.maxStorageBuffersPerShaderStage,
    maxUniformBufferBindingSize: adapterLimits.maxUniformBufferBindingSize,
  };
}


async function initializePlatformAndRegistry(adapter) {
  if (platformInitialized) {
    return;
  }

  platformInitialized = true;

  try {
    // Dynamic import to avoid circular dependencies and enable hotswap
    const [platformLoader, kernelRegistry] = await Promise.all([
      import('../config/platforms/loader.js'),
      import('../config/kernels/registry.js'),
    ]);

    // Initialize platform detection with the adapter
    resolvedPlatformConfig = await platformLoader.initializePlatform(adapter);

    // Preload kernel registry (cached for subsequent calls)
    await kernelRegistry.getRegistry();

    log.debug('GPU', 'Platform: ' + resolvedPlatformConfig.platform.name + ' (' + resolvedPlatformConfig.platform.id + ')');
    log.debug('GPU', 'Capabilities: f16=' + resolvedPlatformConfig.capabilities.hasF16 + ', subgroups=' + resolvedPlatformConfig.capabilities.hasSubgroups);
  } catch (e) {
    // Platform/registry init is optional - kernel selection will use fallbacks
    log.warn('GPU', 'Platform/registry init failed (non-fatal): ' +  (e).message);
    resolvedPlatformConfig = null;
  }
}


export async function initDevice() {
  // Return cached device if available
  if (gpuDevice) {
    if (isUsableGPUDevice(gpuDevice)) {
      return gpuDevice;
    }
    clearActiveDeviceState();
    advanceDeviceEpoch();
  }

  if (!isWebGPUAvailable()) {
    throw createDopplerError(ERROR_CODES.GPU_UNAVAILABLE, 'WebGPU is not available in this browser');
  }

  const adapter = await requestAdapter();
  if (!adapter) {
    throw new Error('Failed to get WebGPU adapter');
  }

  // Initialize platform loader and kernel registry early (before device creation)
  // This allows platform-specific config to be available for kernel selection
  await initializePlatformAndRegistry(adapter);

  // Detect available features
  const availableFeatures = detectFeatures(adapter);
  const requestedFeatures = buildFeatureRequests(availableFeatures);
  const limits = buildLimits(adapter);

  // Get adapter info (adapter.info is synchronous in modern WebGPU)
  const adapterInfo = adapter.info || { vendor: 'unknown', architecture: 'unknown', device: 'unknown', description: '' };

  try {
    gpuDevice = await adapter.requestDevice({
      requiredFeatures: requestedFeatures,
      requiredLimits: limits,
    });
  } catch (e) {
    // Fallback: request device without optional features
    log.warn('GPU', 'Failed to request device with features, trying minimal config: ' +  (e).message);
    gpuDevice = await adapter.requestDevice();
  }

  if (!gpuDevice) {
    throw createDopplerError(ERROR_CODES.GPU_DEVICE_FAILED, 'Failed to create WebGPU device');
  }
  wrapDeviceCreateBindGroup(gpuDevice);
  registerDeviceLostHandler(gpuDevice);
  advanceDeviceEpoch();

  // Wrap queue for submit tracking (when enabled)
  wrapQueueForTracking(gpuDevice.queue);

  // Cache kernel capabilities
  kernelCapabilities = {
    hasSubgroups: gpuDevice.features.has(FEATURES.SUBGROUPS),
    hasSubgroupsF16: gpuDevice.features.has(FEATURES.SUBGROUPS_F16),
    hasF16: gpuDevice.features.has(FEATURES.SHADER_F16),
    hasTimestampQuery: gpuDevice.features.has(FEATURES.TIMESTAMP_QUERY),
    maxBufferSize: gpuDevice.limits.maxStorageBufferBindingSize,
    maxWorkgroupSize: gpuDevice.limits.maxComputeInvocationsPerWorkgroup,
    maxWorkgroupStorageSize: gpuDevice.limits.maxComputeWorkgroupStorageSize,
    adapterInfo: {
      vendor: adapterInfo.vendor || 'unknown',
      architecture: adapterInfo.architecture || 'unknown',
      device: adapterInfo.device || 'unknown',
      description: adapterInfo.description || '',
    },
  };

  const features = [
    kernelCapabilities.hasF16 && 'f16',
    kernelCapabilities.hasSubgroups && 'subgroups',
  ].filter(Boolean).join('/') || 'basic';
  console.log('[GPU] ' + (adapterInfo.vendor || 'unknown') + ' ' + (adapterInfo.architecture || adapterInfo.device || '') + ', ' + features + ', ' + (kernelCapabilities.maxBufferSize / GB).toFixed(1) + 'GB');

  return gpuDevice;
}

export function setDevice(device, options = {}) {
  if (!device) {
    clearActiveDeviceState();
    advanceDeviceEpoch();
    return;
  }

  gpuDevice = device;
  wrapDeviceCreateBindGroup(gpuDevice);
  registerDeviceLostHandler(gpuDevice);
  advanceDeviceEpoch();
  wrapQueueForTracking(gpuDevice.queue);

  const adapterInfo = options.adapterInfo ?? {
    vendor: 'unknown',
    architecture: 'unknown',
    device: 'unknown',
    description: '',
  };

  kernelCapabilities = {
    hasSubgroups: gpuDevice.features.has(FEATURES.SUBGROUPS),
    hasSubgroupsF16: gpuDevice.features.has(FEATURES.SUBGROUPS_F16),
    hasF16: gpuDevice.features.has(FEATURES.SHADER_F16),
    hasTimestampQuery: gpuDevice.features.has(FEATURES.TIMESTAMP_QUERY),
    maxBufferSize: gpuDevice.limits.maxStorageBufferBindingSize,
    maxWorkgroupSize: gpuDevice.limits.maxComputeInvocationsPerWorkgroup,
    maxWorkgroupStorageSize: gpuDevice.limits.maxComputeWorkgroupStorageSize,
    adapterInfo,
  };

  if (options.platformConfig !== undefined) {
    resolvedPlatformConfig = options.platformConfig;
    platformInitialized = options.platformConfig !== null;
  } else {
    resolvedPlatformConfig = null;
    platformInitialized = false;
  }
}


export function getKernelCapabilities() {
  if (!kernelCapabilities) {
    throw new Error('Device not initialized. Call initDevice() first.');
  }
  return { ...kernelCapabilities };
}


export function getDevice() {
  return gpuDevice;
}

export function getDeviceEpoch() {
  return deviceEpoch;
}


export function getPlatformConfig() {
  return resolvedPlatformConfig;
}


export function isPlatformInitialized() {
  return platformInitialized && resolvedPlatformConfig !== null;
}


export function resetDeviceState() {
  clearActiveDeviceState();
  advanceDeviceEpoch();
}


export function destroyDevice() {
  if (gpuDevice) {
    gpuDevice.destroy();
    clearActiveDeviceState();
    advanceDeviceEpoch();
  }
}


export function hasFeature(feature) {
  if (!gpuDevice) {
    return false;
  }
  return gpuDevice.features.has( (feature));
}


export function getDeviceLimits() {
  if (!gpuDevice) {
    return null;
  }
  return {
    maxStorageBufferBindingSize: gpuDevice.limits.maxStorageBufferBindingSize,
    maxBufferSize: gpuDevice.limits.maxBufferSize,
    maxComputeWorkgroupSizeX: gpuDevice.limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: gpuDevice.limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: gpuDevice.limits.maxComputeWorkgroupSizeZ,
    maxComputeInvocationsPerWorkgroup: gpuDevice.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupStorageSize: gpuDevice.limits.maxComputeWorkgroupStorageSize,
    maxStorageBuffersPerShaderStage: gpuDevice.limits.maxStorageBuffersPerShaderStage,
    maxUniformBufferBindingSize: gpuDevice.limits.maxUniformBufferBindingSize,
    maxComputeWorkgroupsPerDimension: gpuDevice.limits.maxComputeWorkgroupsPerDimension,
  };
}
