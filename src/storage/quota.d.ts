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

/**
 * Checks if the Storage API is available
 */
export declare function isStorageAPIAvailable(): boolean;

/**
 * Checks if OPFS is available
 */
export declare function isOPFSAvailable(): boolean;

/**
 * Checks if IndexedDB is available
 */
export declare function isIndexedDBAvailable(): boolean;

/**
 * Gets current storage quota information
 */
export declare function getQuotaInfo(): Promise<QuotaInfo>;

/**
 * Checks if storage is currently persisted
 */
export declare function isPersisted(): Promise<boolean>;

/**
 * Requests persistent storage from the browser
 */
export declare function requestPersistence(): Promise<PersistenceResult>;

/**
 * Checks if there's enough space for a download
 */
export declare function checkSpaceAvailable(requiredBytes: number): Promise<SpaceCheckResult>;

/**
 * Formats bytes into human-readable string
 */
export declare function formatBytes(bytes: number): string;

/**
 * Gets a detailed storage report for debugging/display
 */
export declare function getStorageReport(): Promise<StorageReport>;

/**
 * Error class for quota-related errors
 */
export declare class QuotaExceededError extends Error {
  readonly required: number;
  readonly available: number;
  readonly shortfall: number;
  constructor(required: number, available: number);
}

/**
 * Monitors storage and calls callback when thresholds are crossed
 * @returns Stop function to cancel monitoring
 */
export declare function monitorStorage(
  onLowSpace: StorageCallback | null,
  onCriticalSpace: StorageCallback | null,
  intervalMs?: number
): () => void;

/**
 * Suggests actions when quota is exceeded
 */
export declare function getSuggestions(quotaInfo: QuotaInfo): string[];

/**
 * Clears module-level cache (useful for testing)
 */
export declare function clearCache(): void;
