/**
 * DOPPLER Debug Module - Log History and Snapshots
 *
 * Tools for retrieving log history and creating debug snapshots.
 *
 * @module debug/history
 */

import type { LogEntry, TraceCategory } from './config.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Log history filter
 */
export interface LogHistoryFilter {
  level?: string;
  module?: string;
  last?: number;
}

/**
 * Debug snapshot
 */
export interface DebugSnapshot {
  timestamp: string;
  logLevel: string | undefined;
  traceCategories: TraceCategory[];
  enabledModules: string[];
  disabledModules: string[];
  recentLogs: Array<{
    time: string;
    level: string;
    module: string;
    message: string;
  }>;
  errorCount: number;
  warnCount: number;
}

// ============================================================================
// History Functions
// ============================================================================

/**
 * Get log history for debugging.
 */
export declare function getLogHistory(filter?: LogHistoryFilter): LogEntry[];

/**
 * Clear log history.
 */
export declare function clearLogHistory(): void;

/**
 * Print a summary of recent logs.
 */
export declare function printLogSummary(count?: number): void;

/**
 * Export a debug snapshot for bug reports.
 */
export declare function getDebugSnapshot(): DebugSnapshot;
