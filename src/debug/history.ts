/**
 * DOPPLER Debug Module - Log History and Snapshots
 *
 * Tools for retrieving log history and creating debug snapshots.
 *
 * @module debug/history
 */

import {
  LOG_LEVELS,
  type LogLevel,
  type LogEntry,
  type TraceCategory,
  currentLogLevel,
  enabledTraceCategories,
  enabledModules,
  disabledModules,
  logHistory,
} from './config.js';

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
export function getLogHistory(filter: LogHistoryFilter = {}): LogEntry[] {
  let history = [...logHistory];

  if (filter.level) {
    history = history.filter((h) => h.level === filter.level!.toUpperCase());
  }

  if (filter.module) {
    const m = filter.module.toLowerCase();
    history = history.filter((h) => h.module.toLowerCase().includes(m));
  }

  if (filter.last) {
    history = history.slice(-filter.last);
  }

  return history;
}

/**
 * Clear log history.
 */
export function clearLogHistory(): void {
  logHistory.length = 0;
}

/**
 * Print a summary of recent logs.
 */
export function printLogSummary(count = 20): void {
  const recent = logHistory.slice(-count);
  console.log('=== Recent Logs ===');
  for (const entry of recent) {
    const time = entry.perfTime.toFixed(1);
    console.log(`[${time}ms][${entry.level}][${entry.module}] ${entry.message}`);
  }
  console.log('===================');
}

/**
 * Export a debug snapshot for bug reports.
 */
export function getDebugSnapshot(): DebugSnapshot {
  return {
    timestamp: new Date().toISOString(),
    logLevel: Object.keys(LOG_LEVELS).find(
      (k) => LOG_LEVELS[k as LogLevel] === currentLogLevel
    ),
    traceCategories: [...enabledTraceCategories],
    enabledModules: [...enabledModules],
    disabledModules: [...disabledModules],
    recentLogs: logHistory.slice(-50).map((e) => ({
      time: e.perfTime.toFixed(1),
      level: e.level,
      module: e.module,
      message: e.message,
    })),
    errorCount: logHistory.filter((e) => e.level === 'ERROR').length,
    warnCount: logHistory.filter((e) => e.level === 'WARN').length,
  };
}
