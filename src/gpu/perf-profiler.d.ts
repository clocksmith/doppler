/**
 * Performance Profiler (Tier 2 P0)
 *
 * Real-time profiling utilities to identify performance bottlenecks
 * in the inference pipeline.
 */

/** Profile entry for a single operation */
export interface ProfileEntry {
  name: string;
  category: 'kernel' | 'memory' | 'sync' | 'other';
  startTime: number;
  endTime: number;
  duration: number;
  metadata?: Record<string, unknown>;
}

/** Profile report summary */
export interface ProfileReport {
  entries: ProfileEntry[];
  summary: {
    totalTime: number;
    kernelTime: number;
    memoryTime: number;
    syncTime: number;
    otherTime: number;
    kernelCount: number;
    memoryOps: number;
    syncOps: number;
  };
  breakdown: {
    name: string;
    totalTime: number;
    count: number;
    avgTime: number;
    pctOfTotal: number;
  }[];
  bottlenecks: {
    name: string;
    impact: number;
    suggestion: string;
  }[];
}

/**
 * Check if profiling is enabled
 */
export function isProfilingEnabled(): boolean;

/**
 * Enable/disable profiling
 */
export function setProfilingEnabled(enabled: boolean): void;

/**
 * Clear all profile entries
 */
export function clearProfile(): void;

/**
 * Start a new profiling session
 */
export function startProfileSession(): void;

/**
 * Record a profile entry
 */
export function recordProfileEntry(
  name: string,
  category: ProfileEntry['category'],
  startTime: number,
  endTime: number,
  metadata?: Record<string, unknown>
): void;

/**
 * Profile an async operation
 */
export function profileAsync<T>(
  name: string,
  category: ProfileEntry['category'],
  fn: () => Promise<T>,
  metadata?: Record<string, unknown>
): Promise<T>;

/**
 * Profile a sync operation
 */
export function profileSync<T>(
  name: string,
  category: ProfileEntry['category'],
  fn: () => T,
  metadata?: Record<string, unknown>
): T;

/**
 * Profile a GPU kernel dispatch with queue sync
 */
export function profileKernel(
  name: string,
  dispatchFn: () => void,
  metadata?: Record<string, unknown>
): Promise<void>;

/**
 * Generate profile report
 */
export function getProfileReport(): ProfileReport;

/**
 * Print profile report via debug log
 */
export function printProfileReport(report?: ProfileReport): void;

/**
 * Export profile data as JSON
 */
export function exportProfileJSON(report?: ProfileReport): string;

/**
 * Analyze decode performance and suggest optimizations
 */
export function analyzeDecodePerformance(
  tokensGenerated: number,
  totalTimeMs: number,
  targetTokPerSec?: number
): {
  currentTokPerSec: number;
  targetTokPerSec: number;
  gap: number;
  suggestions: string[];
};
