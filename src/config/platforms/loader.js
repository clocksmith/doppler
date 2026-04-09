import { getRuntimeConfig } from '../runtime.js';
import { loadJson } from '../../utils/load-json.js';
import { log } from '../../debug/index.js';

let currentPlatform = null;

let currentCapabilities = null;

const platformCache = new Map();

let platformsBaseUrl = null;

const DEFAULT_PREFER_UNIFIED_MEMORY = false;

const PLATFORM_FILES = [
  'apple-m3',
  'apple-m2',
  'apple-m1',
  'nvidia-rtx40',
  'nvidia-rtx30',
  'amd-rdna3',
  'generic',
];

function normalizeDetectionValue(value) {
  if (value == null) {
    return '';
  }
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, '');
}

export function setPlatformsBaseUrl(baseUrl) {
  platformsBaseUrl = baseUrl;
  platformCache.clear();
  currentPlatform = null;
}

async function loadPlatformConfig(platformId) {
  if (platformCache.has(platformId)) {
    return platformCache.get(platformId) || null;
  }

  try {
    const baseUrl = platformsBaseUrl || new URL('./', import.meta.url).href;
    const config = await loadJson(`${platformId}.json`, baseUrl, 'Failed to load platform config');
    if (!config) {
      return null;
    }
    platformCache.set(platformId, config);
    return config;
  } catch {
    return null;
  }
}

export async function detectPlatform(adapterInfo) {
  const vendor = normalizeDetectionValue(adapterInfo.vendor);
  const architecture = normalizeDetectionValue(adapterInfo.architecture);
  const device = normalizeDetectionValue(adapterInfo.device);
  const description = normalizeDetectionValue(adapterInfo.description);

  for (const platformId of PLATFORM_FILES) {
    const config = await loadPlatformConfig(platformId);
    if (!config) continue;

    const detection = config.detection;
    let matches = true;

    if (detection.vendor && !vendor.includes(normalizeDetectionValue(detection.vendor))) {
      matches = false;
    }
    if (detection.architecture && !architecture.includes(normalizeDetectionValue(detection.architecture))) {
      matches = false;
    }
    if (detection.device && !device.includes(normalizeDetectionValue(detection.device))) {
      matches = false;
    }
    if (detection.description && !description.includes(normalizeDetectionValue(detection.description))) {
      matches = false;
    }

    if (matches && !config.isGeneric) {
      currentPlatform = config;
      return config;
    }
  }

  log.info('Platform', 'No specific platform matched for adapter (vendor=' + (adapterInfo.vendor || 'unknown') + ', arch=' + (adapterInfo.architecture || 'unknown') + '), falling back to generic');
  const genericConfig = await loadPlatformConfig('generic');
  if (genericConfig) {
    currentPlatform = genericConfig;
    return genericConfig;
  }

  log.info('Platform', 'Generic platform config not available, using built-in fallback');
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

export function getMemoryHints() {
  return getPlatform().memoryHints;
}

export function prefersUnifiedMemory() {
  return getMemoryHints()?.preferUnifiedMemory ?? DEFAULT_PREFER_UNIFIED_MEMORY;
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
