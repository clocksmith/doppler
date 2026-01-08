/**
 * quota.ts - Storage Quota Management
 *
 * Handles:
 * - Storage persistence requests (navigator.storage.persist())
 * - Quota detection and monitoring
 * - Graceful quota exhaustion handling
 *
 * @module storage/quota
 */

import { log } from '../debug/index.js';
import { getRuntimeConfig } from '../config/runtime.js';

/**
 * @returns {import('../config/schema/runtime-schema.js').QuotaConfig}
 */
function getQuotaConfig() {
  return getRuntimeConfig().storage.quota;
}

// Cached persistence state
/** @type {boolean | null} */
let persistenceState = null;

/**
 * Checks if the Storage API is available
 * @returns {boolean}
 */
export function isStorageAPIAvailable() {
  return typeof navigator !== 'undefined' &&
    !!navigator.storage &&
    typeof navigator.storage.estimate === 'function';
}

/**
 * Checks if OPFS is available
 * @returns {boolean}
 */
export function isOPFSAvailable() {
  return typeof navigator !== 'undefined' &&
    !!navigator.storage &&
    typeof navigator.storage.getDirectory === 'function';
}

/**
 * Checks if IndexedDB is available
 * @returns {boolean}
 */
export function isIndexedDBAvailable() {
  return typeof indexedDB !== 'undefined';
}

/**
 * Gets current storage quota information
 * @returns {Promise<import('./quota.js').QuotaInfo>}
 */
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
    log.warn('Quota', `Failed to get storage quota: ${/** @type {Error} */ (error).message}`);
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

/**
 * Checks if storage is currently persisted
 * @returns {Promise<boolean>}
 */
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
    log.warn('Quota', `Failed to check persistence status: ${/** @type {Error} */ (error).message}`);
    persistenceState = false;
    return false;
  }
}

/**
 * Requests persistent storage from the browser
 * @returns {Promise<import('./quota.js').PersistenceResult>}
 */
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
      reason: `Persistence request failed: ${/** @type {Error} */ (error).message}`
    };
  }
}

/**
 * Checks if there's enough space for a download
 * @param {number} requiredBytes
 * @returns {Promise<import('./quota.js').SpaceCheckResult>}
 */
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

/**
 * Formats bytes into human-readable string
 * @param {number} bytes
 * @returns {string}
 */
export function formatBytes(bytes) {
  if (bytes === 0) return '0 B';

  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const k = 1024;
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const value = bytes / Math.pow(k, i);

  return `${value.toFixed(i > 0 ? 2 : 0)} ${units[i]}`;
}

/**
 * Calculates the total size of an OPFS directory recursively
 * @param {FileSystemDirectoryHandle} dirHandle
 * @returns {Promise<number>}
 */
async function calculateDirectorySize(dirHandle) {
  let totalSize = 0;

  const entries = /** @type {{ entries(): AsyncIterable<[string, FileSystemHandle]> }} */ (/** @type {unknown} */ (dirHandle)).entries();
  for await (const [_name, handle] of entries) {
    if (handle.kind === 'file') {
      try {
        const file = await /** @type {FileSystemFileHandle} */ (handle).getFile();
        totalSize += file.size;
      } catch (_e) {
        // File might be locked or inaccessible
      }
    } else if (handle.kind === 'directory') {
      totalSize += await calculateDirectorySize(/** @type {FileSystemDirectoryHandle} */ (handle));
    }
  }

  return totalSize;
}

/**
 * Gets a detailed storage report for debugging/display
 * @returns {Promise<import('./quota.js').StorageReport>}
 */
export async function getStorageReport() {
  const quotaInfo = await getQuotaInfo();

  // Try to get OPFS-specific usage if available
  /** @type {number | null} */
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

/**
 * Error class for quota-related errors
 */
export class QuotaExceededError extends Error {
  /**
   * @param {number} required
   * @param {number} available
   */
  constructor(required, available) {
    super(`Insufficient storage: need ${formatBytes(required)}, have ${formatBytes(available)}`);
    this.name = 'QuotaExceededError';
    /** @readonly */
    this.required = required;
    /** @readonly */
    this.available = available;
    /** @readonly */
    this.shortfall = required - available;
  }
}

/**
 * Monitors storage and calls callback when thresholds are crossed
 * @param {import('./quota.js').StorageCallback | null} onLowSpace
 * @param {import('./quota.js').StorageCallback | null} onCriticalSpace
 * @param {number} [intervalMs]
 * @returns {() => void} Stop function to cancel monitoring
 */
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

/**
 * Suggests actions when quota is exceeded
 * @param {import('./quota.js').QuotaInfo} quotaInfo
 * @returns {string[]}
 */
export function getSuggestions(quotaInfo) {
  /** @type {string[]} */
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

/**
 * Clears module-level cache (useful for testing)
 * @returns {void}
 */
export function clearCache() {
  persistenceState = null;
}
