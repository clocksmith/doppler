

import { log } from '../../debug/index.js';
import { getRuntimeConfig } from '../../config/runtime.js';


export function getTunerConfig() {
  return getRuntimeConfig().shared.tuner;
}


export function getDeviceSignature(capabilities) {
  
  const info = capabilities?.adapterInfo || { vendor: '', architecture: '', device: '' };
  return `${info.vendor}_${info.architecture}_${info.device}`.replace(/[^a-zA-Z0-9]/g, '_');
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
      return new Map(Object.entries(data));
    }
  } catch (e) {
    log.warn('KernelTuner', `Failed to load cache: ${e}`);
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
