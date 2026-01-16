/**
 * WebGPU Device Initialization and Feature Probing
 *
 * Handles WebGPU adapter/device setup with comprehensive feature detection
 * for optimal kernel selection.
 *
 * Also initializes the platform loader and kernel registry for config-as-code
 * kernel selection based on detected GPU hardware.
 */

import { wrapQueueForTracking, setTrackSubmits } from './submit-tracker.js';
import { log } from '../debug/index.js';
import { createDopplerError, ERROR_CODES } from '../errors/index.js';

// Re-export submit tracker for convenience
export { setTrackSubmits };

// Cached device and capabilities
/** @type {GPUDevice | null} */
let gpuDevice = null;
/** @type {import('./device.js').KernelCapabilities | null} */
let kernelCapabilities = null;

// Cached platform config (set during initDevice)
/** @type {import('../config/schema/platform.schema.js').ResolvedPlatformConfig | null} */
let resolvedPlatformConfig = null;

// Track whether platform/registry initialization has been attempted
let platformInitialized = false;

/**
 * Feature flags detected during initialization
 */
export const FEATURES = /** @type {const} */ ({
  SHADER_F16: 'shader-f16',
  SUBGROUPS: 'subgroups',
  SUBGROUPS_F16: 'subgroups-f16',
  TIMESTAMP_QUERY: 'timestamp-query',
});

/**
 * Probe for WebGPU availability
 * @returns {boolean}
 */
export function isWebGPUAvailable() {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

/**
 * Request WebGPU adapter with fallback options
 * @param {{powerPreference?: 'low-power' | 'high-performance', forceFallbackAdapter?: boolean}} [options]
 * @returns {Promise<GPUAdapter | null>}
 */
async function requestAdapter(options = {}) {
  if (!isWebGPUAvailable()) {
    return null;
  }

  // Try high-performance first, then fallback
  /** @type {GPURequestAdapterOptions[]} */
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

/**
 * Detect available features from adapter
 * @param {GPUAdapter} adapter
 * @returns {Set<string>}
 */
function detectFeatures(adapter) {
  const available = new Set();

  for (const feature of adapter.features) {
    available.add(feature);
  }

  return available;
}

/**
 * Build list of features to request based on availability
 * @param {Set<string>} available
 * @returns {GPUFeatureName[]}
 */
function buildFeatureRequests(available) {
  /** @type {GPUFeatureName[]} */
  const requested = [];

  // Request shader-f16 for FP16 matmul kernels
  if (available.has(FEATURES.SHADER_F16)) {
    requested.push(/** @type {GPUFeatureName} */ (FEATURES.SHADER_F16));
  }

  // Request subgroups for efficient dequantization
  if (available.has(FEATURES.SUBGROUPS)) {
    requested.push(/** @type {GPUFeatureName} */ (FEATURES.SUBGROUPS));
  }

  // Request subgroups-f16 if available (for combined f16 + subgroup ops)
  if (available.has(FEATURES.SUBGROUPS_F16)) {
    requested.push(/** @type {GPUFeatureName} */ (FEATURES.SUBGROUPS_F16));
  }

  // Request timestamp query for profiling (optional)
  if (available.has(FEATURES.TIMESTAMP_QUERY)) {
    requested.push(/** @type {GPUFeatureName} */ (FEATURES.TIMESTAMP_QUERY));
  }

  return requested;
}

/**
 * Build device limits based on adapter capabilities
 * @param {GPUAdapter} adapter
 * @returns {Record<string, GPUSize64>}
 */
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

/**
 * Initialize platform loader and kernel registry.
 * Called automatically during initDevice() after adapter is obtained.
 * @param {GPUAdapter} adapter
 * @returns {Promise<void>}
 */
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
    log.warn('GPU', 'Platform/registry init failed (non-fatal): ' + /** @type {Error} */ (e).message);
    resolvedPlatformConfig = null;
  }
}

/**
 * Initialize WebGPU device with optimal features
 * @returns {Promise<GPUDevice>}
 * @throws Error if WebGPU is unavailable or device creation fails
 */
export async function initDevice() {
  // Return cached device if available
  if (gpuDevice) {
    return gpuDevice;
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
    log.warn('GPU', 'Failed to request device with features, trying minimal config: ' + /** @type {Error} */ (e).message);
    gpuDevice = await adapter.requestDevice();
  }

  if (!gpuDevice) {
    throw createDopplerError(ERROR_CODES.GPU_DEVICE_FAILED, 'Failed to create WebGPU device');
  }

  // Set up device lost handler
  gpuDevice.lost.then((info) => {
    log.error('GPU', 'Device lost: ' + info.message + ', Reason: ' + info.reason);
    gpuDevice = null;
    kernelCapabilities = null;
    resolvedPlatformConfig = null;
    platformInitialized = false;
  });

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
  console.log('[GPU] ' + (adapterInfo.vendor || 'unknown') + ' ' + (adapterInfo.architecture || adapterInfo.device || '') + ', ' + features + ', ' + (kernelCapabilities.maxBufferSize / (1024 * 1024 * 1024)).toFixed(1) + 'GB');

  return gpuDevice;
}

/**
 * Get kernel capabilities (call after initDevice)
 * @returns {import('./device.js').KernelCapabilities}
 * @throws Error if device not initialized
 */
export function getKernelCapabilities() {
  if (!kernelCapabilities) {
    throw new Error('Device not initialized. Call initDevice() first.');
  }
  return { ...kernelCapabilities };
}

/**
 * Get the current GPU device (call after initDevice)
 * @returns {GPUDevice | null}
 */
export function getDevice() {
  return gpuDevice;
}

/**
 * Get the resolved platform configuration (call after initDevice)
 * @returns {import('../config/schema/platform.schema.js').ResolvedPlatformConfig | null}
 */
export function getPlatformConfig() {
  return resolvedPlatformConfig;
}

/**
 * Check if platform and registry are initialized
 * @returns {boolean}
 */
export function isPlatformInitialized() {
  return platformInitialized && resolvedPlatformConfig !== null;
}

/**
 * Destroy the device and clear cache
 * @returns {void}
 */
export function destroyDevice() {
  if (gpuDevice) {
    gpuDevice.destroy();
    gpuDevice = null;
    kernelCapabilities = null;
    resolvedPlatformConfig = null;
    platformInitialized = false;
  }
}

/**
 * Check if a specific feature is available
 * @param {string} feature
 * @returns {boolean}
 */
export function hasFeature(feature) {
  if (!gpuDevice) {
    return false;
  }
  return gpuDevice.features.has(/** @type {GPUFeatureName} */ (feature));
}

/**
 * Get device limits
 * @returns {import('./device.js').DeviceLimits | null}
 */
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
