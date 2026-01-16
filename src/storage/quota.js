

import { log } from '../debug/index.js';
import { getRuntimeConfig } from '../config/runtime.js';


function getQuotaConfig() {
  return getRuntimeConfig().loading.storage.quota;
}

// Cached persistence state

let persistenceState = null;


export function isStorageAPIAvailable() {
  return typeof navigator !== 'undefined' &&
    !!navigator.storage &&
    typeof navigator.storage.estimate === 'function';
}


export function isOPFSAvailable() {
  return typeof navigator !== 'undefined' &&
    !!navigator.storage &&
    typeof navigator.storage.getDirectory === 'function';
}


export function isIndexedDBAvailable() {
  return typeof indexedDB !== 'undefined';
}


export async function getQuotaInfo() {
  if (!isStorageAPIAvailable()) {
    // Return conservative defaults when API unavailable
    return {
      usage: 0,
      quota: 0,
      available: 0,
      usagePercent: 0,
      persisted: false,
      lowSpace: true,
      criticalSpace: true
    };
  }

  try {
    const estimate = await navigator.storage.estimate();
    const usage = estimate.usage || 0;
    const quota = estimate.quota || 0;
    const available = Math.max(0, quota - usage);

    // Check persistence state
    const persisted = await isPersisted();

    return {
      usage,
      quota,
      available,
      usagePercent: quota > 0 ? (usage / quota) * 100 : 0,
      persisted,
      lowSpace: available < getQuotaConfig().lowSpaceThresholdBytes,
      criticalSpace: available < getQuotaConfig().criticalSpaceThresholdBytes
    };
  } catch (error) {
    log.warn('Quota', `Failed to get storage quota: ${ (error).message}`);
    return {
      usage: 0,
      quota: 0,
      available: 0,
      usagePercent: 0,
      persisted: false,
      lowSpace: true,
      criticalSpace: true
    };
  }
}


export async function isPersisted() {
  if (persistenceState !== null) {
    return persistenceState;
  }

  if (!isStorageAPIAvailable() || typeof navigator.storage.persisted !== 'function') {
    persistenceState = false;
    return false;
  }

  try {
    persistenceState = await navigator.storage.persisted();
    return persistenceState;
  } catch (error) {
    log.warn('Quota', `Failed to check persistence status: ${ (error).message}`);
    persistenceState = false;
    return false;
  }
}


export async function requestPersistence() {
  if (!isStorageAPIAvailable() || typeof navigator.storage.persist !== 'function') {
    return {
      granted: false,
      reason: 'Storage API not available'
    };
  }

  // Check if already persisted
  const alreadyPersisted = await isPersisted();
  if (alreadyPersisted) {
    return {
      granted: true,
      reason: 'Already persisted'
    };
  }

  try {
    const granted = await navigator.storage.persist();
    persistenceState = granted;

    if (granted) {
      return {
        granted: true,
        reason: 'Persistence granted'
      };
    } else {
      // Browser may deny based on heuristics (engagement, bookmarked, etc.)
      return {
        granted: false,
        reason: 'Browser denied persistence request (try bookmarking the site)'
      };
    }
  } catch (error) {
    return {
      granted: false,
      reason: `Persistence request failed: ${ (error).message}`
    };
  }
}


export async function checkSpaceAvailable(requiredBytes) {
  const info = await getQuotaInfo();

  const hasSpace = info.available >= requiredBytes;
  const shortfall = hasSpace ? 0 : requiredBytes - info.available;

  return {
    hasSpace,
    info,
    shortfall
  };
}


export function formatBytes(bytes) {
  if (bytes === 0) return '0 B';

  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const k = 1024;
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const value = bytes / Math.pow(k, i);

  return `${value.toFixed(i > 0 ? 2 : 0)} ${units[i]}`;
}


async function calculateDirectorySize(dirHandle) {
  let totalSize = 0;

  const entries =  ( (dirHandle)).entries();
  for await (const [_name, handle] of entries) {
    if (handle.kind === 'file') {
      try {
        const file = await  (handle).getFile();
        totalSize += file.size;
      } catch (_e) {
        // File might be locked or inaccessible
      }
    } else if (handle.kind === 'directory') {
      totalSize += await calculateDirectorySize( (handle));
    }
  }

  return totalSize;
}


export async function getStorageReport() {
  const quotaInfo = await getQuotaInfo();

  // Try to get OPFS-specific usage if available
  
  let opfsUsage = null;
  if (isOPFSAvailable()) {
    try {
      const root = await navigator.storage.getDirectory();
      opfsUsage = await calculateDirectorySize(root);
    } catch (_e) {
      // OPFS might not be accessible in all contexts
    }
  }

  return {
    quota: {
      total: formatBytes(quotaInfo.quota),
      used: formatBytes(quotaInfo.usage),
      available: formatBytes(quotaInfo.available),
      usagePercent: quotaInfo.usagePercent.toFixed(1) + '%'
    },
    persisted: quotaInfo.persisted,
    opfsUsage: opfsUsage !== null ? formatBytes(opfsUsage) : 'N/A',
    warnings: {
      lowSpace: quotaInfo.lowSpace,
      criticalSpace: quotaInfo.criticalSpace
    },
    features: {
      storageAPI: isStorageAPIAvailable(),
      opfs: isOPFSAvailable(),
      indexedDB: isIndexedDBAvailable()
    }
  };
}


export class QuotaExceededError extends Error {
  
  constructor(required, available) {
    super(`Insufficient storage: need ${formatBytes(required)}, have ${formatBytes(available)}`);
    this.name = 'QuotaExceededError';
    
    this.required = required;
    
    this.available = available;
    
    this.shortfall = required - available;
  }
}


export function monitorStorage(
  onLowSpace,
  onCriticalSpace,
  intervalMs = getQuotaConfig().monitorIntervalMs
) {
  let wasLow = false;
  let wasCritical = false;

  const check = async () => {
    const info = await getQuotaInfo();

    // Trigger callbacks only on state transitions
    if (info.criticalSpace && !wasCritical) {
      wasCritical = true;
      onCriticalSpace?.(info);
    } else if (!info.criticalSpace) {
      wasCritical = false;
    }

    if (info.lowSpace && !wasLow) {
      wasLow = true;
      onLowSpace?.(info);
    } else if (!info.lowSpace) {
      wasLow = false;
    }
  };

  // Initial check
  check();

  // Periodic checks
  const intervalId = setInterval(check, intervalMs);

  // Return stop function
  return () => clearInterval(intervalId);
}


export function getSuggestions(quotaInfo) {
  
  const suggestions = [];

  if (!quotaInfo.persisted) {
    suggestions.push('Request persistent storage to prevent automatic deletion');
  }

  if (quotaInfo.criticalSpace) {
    suggestions.push('Clear browser cache or delete unused data');
    suggestions.push('Consider using Tier 2 (native) for larger models');
    suggestions.push('Free up disk space on your device');
  } else if (quotaInfo.lowSpace) {
    suggestions.push('Storage space is running low - consider clearing unused models');
  }

  return suggestions;
}


export function clearCache() {
  persistenceState = null;
}
