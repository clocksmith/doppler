/**
 * Platform Loader
 *
 * Detects the current GPU platform and loads appropriate configs.
 * Provides platform-specific kernel overrides and memory hints.
 *
 * @module config/platforms/loader
 */

/** @type {import('../schema/platform.schema.js').PlatformSchema | null} */
let currentPlatform = null;

/** @type {import('../schema/platform.schema.js').RuntimeCapabilities | null} */
let currentCapabilities = null;

/** @type {Map<string, import('../schema/platform.schema.js').PlatformSchema>} */
const platformCache = new Map();

/** @type {string | null} */
let platformsBaseUrl = null;

/**
 * Known platform IDs and their config file names.
 * Order matters for detection priority.
 */
const PLATFORM_FILES = [
  'apple-m3',
  'apple-m2',
  'apple-m1',
  'nvidia-rtx40',
  'nvidia-rtx30',
  'amd-rdna3',
  'generic', // Fallback
];

/**
 * Set the base URL for loading platform configs.
 * @param {string} baseUrl
 */
export function setPlatformsBaseUrl(baseUrl) {
  platformsBaseUrl = baseUrl;
  platformCache.clear();
  currentPlatform = null;
}

/**
 * Load a platform config by ID.
 * @param {string} platformId
 * @returns {Promise<import('../schema/platform.schema.js').PlatformSchema | null>}
 */
async function loadPlatformConfig(platformId) {
  if (platformCache.has(platformId)) {
    return platformCache.get(platformId) || null;
  }

  const baseUrl = platformsBaseUrl || new URL('./', import.meta.url).href;
  const url = `${baseUrl}${platformId}.json`;

  try {
    const response = await fetch(url);
    if (!response.ok) {
      return null;
    }
    const config = await response.json();
    platformCache.set(platformId, config);
    return config;
  } catch {
    return null;
  }
}

/**
 * Detect platform from WebGPU adapter info.
 * @param {GPUAdapterInfo} adapterInfo
 * @returns {Promise<import('../schema/platform.schema.js').PlatformSchema>}
 */
export async function detectPlatform(adapterInfo) {
  const vendor = adapterInfo.vendor?.toLowerCase() || '';
  const architecture = adapterInfo.architecture?.toLowerCase() || '';
  const device = adapterInfo.device?.toLowerCase() || '';
  const description = adapterInfo.description?.toLowerCase() || '';

  // Try each platform in priority order
  for (const platformId of PLATFORM_FILES) {
    const config = await loadPlatformConfig(platformId);
    if (!config) continue;

    const detection = config.detection;
    let matches = true;

    if (detection.vendor && !vendor.includes(detection.vendor.toLowerCase())) {
      matches = false;
    }
    if (detection.architecture && !architecture.includes(detection.architecture.toLowerCase())) {
      matches = false;
    }
    if (detection.device && !device.includes(detection.device.toLowerCase())) {
      matches = false;
    }
    if (detection.description && !description.includes(detection.description.toLowerCase())) {
      matches = false;
    }

    if (matches && !config.isGeneric) {
      currentPlatform = config;
      return config;
    }
  }

  // Fall back to generic
  const genericConfig = await loadPlatformConfig('generic');
  if (genericConfig) {
    currentPlatform = genericConfig;
    return genericConfig;
  }

  // Absolute fallback if no config files available
  const fallback = {
    id: 'unknown',
    name: 'Unknown Platform',
    detection: {},
    isGeneric: true,
  };
  currentPlatform = fallback;
  return fallback;
}

/**
 * Initialize platform detection with a WebGPU adapter.
 * @param {GPUAdapter} adapter
 * @returns {Promise<import('../schema/platform.schema.js').ResolvedPlatformConfig>}
 */
export async function initializePlatform(adapter) {
  const adapterInfo = adapter.info;
  const platform = await detectPlatform(adapterInfo);

  // Detect runtime capabilities
  const features = adapter.features;
  currentCapabilities = {
    hasF16: features.has('shader-f16'),
    hasSubgroups: features.has('subgroups'),
    subgroupSize: features.has('subgroups') ? 32 : undefined, // TODO: detect actual size
    maxWorkgroupSize: adapter.limits.maxComputeWorkgroupSizeX,
    maxSharedMemory: adapter.limits.maxComputeWorkgroupStorageSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
  };

  return {
    platform,
    capabilities: currentCapabilities,
  };
}

/**
 * Get the current platform (throws if not initialized).
 * @returns {import('../schema/platform.schema.js').PlatformSchema}
 */
export function getPlatform() {
  if (!currentPlatform) {
    throw new Error('Platform not initialized. Call initializePlatform() first.');
  }
  return currentPlatform;
}

/**
 * Get the current runtime capabilities (throws if not initialized).
 * @returns {import('../schema/platform.schema.js').RuntimeCapabilities}
 */
export function getCapabilities() {
  if (!currentCapabilities) {
    throw new Error('Platform not initialized. Call initializePlatform() first.');
  }
  return currentCapabilities;
}

/**
 * Get kernel override for an operation on current platform.
 * @param {string} operation
 * @returns {import('../schema/platform.schema.js').KernelOperationOverrideSchema | undefined}
 */
export function getKernelOverride(operation) {
  const platform = getPlatform();
  return platform.kernelOverrides?.[operation];
}

/**
 * Get preferred variant for an operation, if platform specifies one.
 * @param {string} operation
 * @returns {string | undefined}
 */
export function getPreferredVariant(operation) {
  return getKernelOverride(operation)?.preferred;
}

/**
 * Check if a variant should be avoided on current platform.
 * @param {string} operation
 * @param {string} variant
 * @returns {boolean}
 */
export function shouldAvoidVariant(operation, variant) {
  const override = getKernelOverride(operation);
  return override?.avoid?.includes(variant) ?? false;
}

/**
 * Get workgroup size override for a variant, if platform specifies one.
 * @param {string} operation
 * @param {string} variant
 * @returns {[number, number, number] | undefined}
 */
export function getWorkgroupOverride(operation, variant) {
  const override = getKernelOverride(operation);
  return override?.workgroupOverrides?.[variant];
}

/**
 * Get WGSL override constants for a variant, if platform specifies any.
 * @param {string} operation
 * @param {string} variant
 * @returns {Record<string, number> | undefined}
 */
export function getWgslOverrides(operation, variant) {
  const override = getKernelOverride(operation);
  return override?.wgslOverrides?.[variant];
}

/**
 * Get memory hints for current platform.
 * @returns {import('../schema/platform.schema.js').MemoryHintsSchema | undefined}
 */
export function getMemoryHints() {
  return getPlatform().memoryHints;
}

/**
 * Check if current platform prefers unified memory strategies.
 * @returns {boolean}
 */
export function prefersUnifiedMemory() {
  return getMemoryHints()?.preferUnifiedMemory ?? false;
}

/**
 * Get optimal buffer alignment for current platform.
 * @returns {number}
 */
export function getBufferAlignment() {
  return getMemoryHints()?.bufferAlignment ?? 256;
}

/**
 * Clear all cached platform data. Useful for hot-reloading.
 */
export function clearPlatformCache() {
  platformCache.clear();
  currentPlatform = null;
  currentCapabilities = null;
}

/**
 * Get resolved platform config with capabilities.
 * @returns {import('../schema/platform.schema.js').ResolvedPlatformConfig}
 */
export function getResolvedPlatformConfig() {
  return {
    platform: getPlatform(),
    capabilities: getCapabilities(),
  };
}
