import { getRuntimeConfig } from '../runtime.js';

let currentPlatform = null;

let currentCapabilities = null;

const platformCache = new Map();

let platformsBaseUrl = null;

const PLATFORM_FILES = [
  'apple-m3',
  'apple-m2',
  'apple-m1',
  'nvidia-rtx40',
  'nvidia-rtx30',
  'amd-rdna3',
  'generic',
];

export function setPlatformsBaseUrl(baseUrl) {
  platformsBaseUrl = baseUrl;
  platformCache.clear();
  currentPlatform = null;
}

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

export async function detectPlatform(adapterInfo) {
  const vendor = adapterInfo.vendor?.toLowerCase() || '';
  const architecture = adapterInfo.architecture?.toLowerCase() || '';
  const device = adapterInfo.device?.toLowerCase() || '';
  const description = adapterInfo.description?.toLowerCase() || '';

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

  const genericConfig = await loadPlatformConfig('generic');
  if (genericConfig) {
    currentPlatform = genericConfig;
    return genericConfig;
  }

  const fallback = {
    id: 'unknown',
    name: 'Unknown Platform',
    detection: {},
    isGeneric: true,
  };
  currentPlatform = fallback;
  return fallback;
}

export async function initializePlatform(adapter) {
  const adapterInfo = adapter.info;
  const platform = await detectPlatform(adapterInfo);

  const features = adapter.features;
  currentCapabilities = {
    hasF16: features.has('shader-f16'),
    hasSubgroups: features.has('subgroups'),
    subgroupSize: features.has('subgroups') ? 32 : undefined,
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

export function getPlatform() {
  if (!currentPlatform) {
    throw new Error('Platform not initialized. Call initializePlatform() first.');
  }
  return currentPlatform;
}

export function getCapabilities() {
  if (!currentCapabilities) {
    throw new Error('Platform not initialized. Call initializePlatform() first.');
  }
  return currentCapabilities;
}

export function getKernelOverride(operation) {
  const platform = getPlatform();
  return platform.kernelOverrides?.[operation];
}

export function getPreferredVariant(operation) {
  return getKernelOverride(operation)?.preferred;
}

export function shouldAvoidVariant(operation, variant) {
  const override = getKernelOverride(operation);
  return override?.avoid?.includes(variant) ?? false;
}

export function getWorkgroupOverride(operation, variant) {
  const override = getKernelOverride(operation);
  return override?.workgroupOverrides?.[variant];
}

export function getWgslOverrides(operation, variant) {
  const override = getKernelOverride(operation);
  return override?.wgslOverrides?.[variant];
}

export function getMemoryHints() {
  return getPlatform().memoryHints;
}

export function prefersUnifiedMemory() {
  return getMemoryHints()?.preferUnifiedMemory ?? false;
}

export function getBufferAlignment() {
  const fallback = getRuntimeConfig().loading.storage.alignment.bufferAlignmentBytes;
  return getMemoryHints()?.bufferAlignment ?? fallback;
}

export function clearPlatformCache() {
  platformCache.clear();
  currentPlatform = null;
  currentCapabilities = null;
}

export function getResolvedPlatformConfig() {
  return {
    platform: getPlatform(),
    capabilities: getCapabilities(),
  };
}
