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

/**
 * Storage quota information
 */
export interface QuotaInfo {
  /** Current storage usage in bytes */
  usage: number;
  /** Total available quota in bytes */
  quota: number;
  /** Available space (quota - usage) */
  available: number;
  /** Usage as percentage of quota */
  usagePercent: number;
  /** Whether storage is persisted */
  persisted: boolean;
  /** True if available < 500MB */
  lowSpace: boolean;
  /** True if available < 100MB */
  criticalSpace: boolean;
}

/**
 * Persistence request result
 */
export interface PersistenceResult {
  granted: boolean;
  reason: string;
}

/**
 * Space check result
 */
export interface SpaceCheckResult {
  hasSpace: boolean;
  info: QuotaInfo;
  shortfall: number;
}

/**
 * Storage report for debugging/display
 */
export interface StorageReport {
  quota: {
    total: string;
    used: string;
    available: string;
    usagePercent: string;
  };
  persisted: boolean;
  opfsUsage: string;
  warnings: {
    lowSpace: boolean;
    criticalSpace: boolean;
  };
  features: {
    storageAPI: boolean;
    opfs: boolean;
    indexedDB: boolean;
  };
}

/**
 * Storage monitor callback
 */
export type StorageCallback = (info: QuotaInfo) => void;

// Thresholds for space warnings
const LOW_SPACE_THRESHOLD = 500 * 1024 * 1024; // 500MB
const CRITICAL_SPACE_THRESHOLD = 100 * 1024 * 1024; // 100MB

// Cached persistence state
let persistenceState: boolean | null = null;

/**
 * Checks if the Storage API is available
 */
export function isStorageAPIAvailable(): boolean {
  return typeof navigator !== 'undefined' &&
    !!navigator.storage &&
    typeof navigator.storage.estimate === 'function';
}

/**
 * Checks if OPFS is available
 */
export function isOPFSAvailable(): boolean {
  return typeof navigator !== 'undefined' &&
    !!navigator.storage &&
    typeof navigator.storage.getDirectory === 'function';
}

/**
 * Checks if IndexedDB is available
 */
export function isIndexedDBAvailable(): boolean {
  return typeof indexedDB !== 'undefined';
}

/**
 * Gets current storage quota information
 */
export async function getQuotaInfo(): Promise<QuotaInfo> {
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
      lowSpace: available < LOW_SPACE_THRESHOLD,
      criticalSpace: available < CRITICAL_SPACE_THRESHOLD
    };
  } catch (error) {
    console.warn('Failed to get storage quota:', error);
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
 */
export async function isPersisted(): Promise<boolean> {
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
    console.warn('Failed to check persistence status:', error);
    persistenceState = false;
    return false;
  }
}

/**
 * Requests persistent storage from the browser
 */
export async function requestPersistence(): Promise<PersistenceResult> {
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
      reason: `Persistence request failed: ${(error as Error).message}`
    };
  }
}

/**
 * Checks if there's enough space for a download
 */
export async function checkSpaceAvailable(requiredBytes: number): Promise<SpaceCheckResult> {
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
 */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';

  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const k = 1024;
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const value = bytes / Math.pow(k, i);

  return `${value.toFixed(i > 0 ? 2 : 0)} ${units[i]}`;
}

/**
 * Calculates the total size of an OPFS directory recursively
 */
async function calculateDirectorySize(dirHandle: FileSystemDirectoryHandle): Promise<number> {
  let totalSize = 0;

  const entries = (dirHandle as unknown as { entries(): AsyncIterable<[string, FileSystemHandle]> }).entries();
  for await (const [_name, handle] of entries) {
    if (handle.kind === 'file') {
      try {
        const file = await (handle as FileSystemFileHandle).getFile();
        totalSize += file.size;
      } catch (_e) {
        // File might be locked or inaccessible
      }
    } else if (handle.kind === 'directory') {
      totalSize += await calculateDirectorySize(handle as FileSystemDirectoryHandle);
    }
  }

  return totalSize;
}

/**
 * Gets a detailed storage report for debugging/display
 */
export async function getStorageReport(): Promise<StorageReport> {
  const quotaInfo = await getQuotaInfo();

  // Try to get OPFS-specific usage if available
  let opfsUsage: number | null = null;
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
  readonly required: number;
  readonly available: number;
  readonly shortfall: number;

  constructor(required: number, available: number) {
    super(`Insufficient storage: need ${formatBytes(required)}, have ${formatBytes(available)}`);
    this.name = 'QuotaExceededError';
    this.required = required;
    this.available = available;
    this.shortfall = required - available;
  }
}

/**
 * Monitors storage and calls callback when thresholds are crossed
 * @returns Stop function to cancel monitoring
 */
export function monitorStorage(
  onLowSpace: StorageCallback | null,
  onCriticalSpace: StorageCallback | null,
  intervalMs = 30000
): () => void {
  let wasLow = false;
  let wasCritical = false;

  const check = async (): Promise<void> => {
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
 */
export function getSuggestions(quotaInfo: QuotaInfo): string[] {
  const suggestions: string[] = [];

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
 */
export function clearCache(): void {
  persistenceState = null;
}
