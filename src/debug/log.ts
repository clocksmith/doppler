/**
 * DOPPLER Debug Module - Core Logging Interface
 *
 * Provides structured logging with level filtering and history tracking.
 *
 * @module debug/log
 */

import { getRuntimeConfig } from '../config/runtime.js';
import {
  LOG_LEVELS,
  type LogLevelValue,
  currentLogLevel,
  enabledModules,
  disabledModules,
  logHistory,
} from './config.js';

// ============================================================================
// Internal Helpers
// ============================================================================

/**
 * Check if logging is enabled for a module at a level.
 */
function shouldLog(module: string, level: LogLevelValue): boolean {
  if (level < currentLogLevel) return false;

  const moduleLower = module.toLowerCase();

  if (enabledModules.size > 0 && !enabledModules.has(moduleLower)) {
    return false;
  }

  if (disabledModules.has(moduleLower)) {
    return false;
  }

  return true;
}

/**
 * Format a log message with timestamp and module tag.
 */
function formatMessage(module: string, message: string): string {
  const timestamp = performance.now().toFixed(1);
  return `[${timestamp}ms][${module}] ${message}`;
}

/**
 * Store log in history for later retrieval.
 */
function storeLog(level: string, module: string, message: string, data?: unknown): void {
  logHistory.push({
    time: Date.now(),
    perfTime: performance.now(),
    level,
    module,
    message,
    data,
  });

  const maxHistory = getRuntimeConfig().debug.logHistory.maxLogHistoryEntries;
  if (logHistory.length > maxHistory) {
    logHistory.shift();
  }
}

// ============================================================================
// Logging Interface
// ============================================================================

/**
 * Main logging interface.
 */
export const log = {
  /**
   * Debug level logging (most verbose).
   */
  debug(module: string, message: string, data?: unknown): void {
    if (!shouldLog(module, LOG_LEVELS.DEBUG)) return;
    const formatted = formatMessage(module, message);
    storeLog('DEBUG', module, message, data);
    if (data !== undefined) {
      console.debug(formatted, data);
    } else {
      console.debug(formatted);
    }
  },

  /**
   * Verbose level logging (detailed operational info).
   */
  verbose(module: string, message: string, data?: unknown): void {
    if (!shouldLog(module, LOG_LEVELS.VERBOSE)) return;
    const formatted = formatMessage(module, message);
    storeLog('VERBOSE', module, message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Info level logging (normal operations).
   */
  info(module: string, message: string, data?: unknown): void {
    if (!shouldLog(module, LOG_LEVELS.INFO)) return;
    const formatted = formatMessage(module, message);
    storeLog('INFO', module, message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Warning level logging.
   */
  warn(module: string, message: string, data?: unknown): void {
    if (!shouldLog(module, LOG_LEVELS.WARN)) return;
    const formatted = formatMessage(module, message);
    storeLog('WARN', module, message, data);
    if (data !== undefined) {
      console.warn(formatted, data);
    } else {
      console.warn(formatted);
    }
  },

  /**
   * Error level logging.
   */
  error(module: string, message: string, data?: unknown): void {
    if (!shouldLog(module, LOG_LEVELS.ERROR)) return;
    const formatted = formatMessage(module, message);
    storeLog('ERROR', module, message, data);
    if (data !== undefined) {
      console.error(formatted, data);
    } else {
      console.error(formatted);
    }
  },

  /**
   * Always log regardless of level (for critical messages).
   */
  always(module: string, message: string, data?: unknown): void {
    const formatted = formatMessage(module, message);
    storeLog('ALWAYS', module, message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },
};
