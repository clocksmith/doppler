

import { log } from '../../debug/index.js';
import { getRuntimeConfig } from '../../config/runtime.js';


export function getTunerConfig() {
  return getRuntimeConfig().shared.tuner;
}

function normalizeSignaturePart(value) {
  return String(value ?? '')
    .trim()
    .replace(/[^a-zA-Z0-9]/g, '_');
}

function hasAdapterIdentity(info) {
  if (!info || typeof info !== 'object') {
    return false;
  }
  return ['vendor', 'architecture', 'device'].some((key) => {
    const value = String(info[key] ?? '').trim().toLowerCase();
    return value.length > 0 && value !== 'unknown';
  });
}

function getFallbackSignature(capabilities) {
  return [
    capabilities?.hasF16 ? 'f16' : 'nof16',
    capabilities?.hasSubgroups ? 'subgroups' : 'nosubgroups',
    capabilities?.hasSubgroupsF16 ? 'subgroupsf16' : 'nosubgroupsf16',
    capabilities?.hasTimestampQuery ? 'timestamp' : 'notimestamp',
    `buf${Number.isFinite(capabilities?.maxBufferSize) ? capabilities.maxBufferSize : 'na'}`,
    `wg${Number.isFinite(capabilities?.maxWorkgroupSize) ? capabilities.maxWorkgroupSize : 'na'}`,
    `wgs${Number.isFinite(capabilities?.maxWorkgroupStorageSize) ? capabilities.maxWorkgroupStorageSize : 'na'}`,
  ].join('_');
}

function isValidDeviceInfo(value) {
  if (value == null) {
    return true;
  }
  return typeof value === 'object';
}

function isValidTuneRecord(value) {
  return !!value
    && typeof value === 'object'
    && Array.isArray(value.optimalWorkgroupSize)
    && value.optimalWorkgroupSize.length === 3
    && value.optimalWorkgroupSize.every((entry) => Number.isFinite(entry) && entry >= 0)
    && Number.isFinite(value.optimalTileSize)
    && value.optimalTileSize >= 0
    && Number.isFinite(value.throughput)
    && value.throughput >= 0
    && Number.isFinite(value.timeMs)
    && value.timeMs >= 0
    && isValidDeviceInfo(value.deviceInfo);
}


export function getDeviceSignature(capabilities) {
  const info = capabilities?.adapterInfo;
  if (hasAdapterIdentity(info)) {
    return [
      normalizeSignaturePart(info.vendor),
      normalizeSignaturePart(info.architecture),
      normalizeSignaturePart(info.device),
    ].join('_');
  }
  return getFallbackSignature(capabilities);
}


export function generateCacheKey(kernelName, inputSizes) {
  return `${kernelName}_${JSON.stringify(inputSizes)}`;
}


export function loadCache(capabilities) {
  if (typeof localStorage === 'undefined') {
    return new Map();
  }

  const signature = getDeviceSignature(capabilities);
  const cacheKey = getTunerConfig().cacheKeyPrefix + signature;

  try {
    const cached = localStorage.getItem(cacheKey);
    if (cached) {
      const data = JSON.parse(cached);
      if (!data || typeof data !== 'object' || Array.isArray(data)) {
        throw new Error('Kernel tuner cache payload must be an object.');
      }
      const records = [];
      for (const [key, value] of Object.entries(data)) {
        if (!isValidTuneRecord(value)) {
          throw new Error(`Kernel tuner cache record "${key}" is malformed.`);
        }
        records.push([key, value]);
      }
      return new Map(records);
    }
  } catch (e) {
    log.warn('KernelTuner', `Failed to load cache: ${e}`);
    localStorage.removeItem(cacheKey);
  }

  return new Map();
}


export function saveCache(cache, capabilities) {
  if (typeof localStorage === 'undefined') return;

  const signature = getDeviceSignature(capabilities);
  const cacheKey = getTunerConfig().cacheKeyPrefix + signature;

  try {
    const data = Object.fromEntries(cache);
    localStorage.setItem(cacheKey, JSON.stringify(data));
  } catch (e) {
    log.warn('KernelTuner', `Failed to save cache: ${e}`);
  }
}


export function clearCacheStorage(capabilities) {
  if (typeof localStorage === 'undefined') return;

  const signature = getDeviceSignature(capabilities);
  localStorage.removeItem(getTunerConfig().cacheKeyPrefix + signature);
}
