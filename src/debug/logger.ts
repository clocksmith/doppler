/**
 * Core Logging Interface
 *
 * @module debug/logger
 */

import { LOG_LEVELS, type LogLevelValue } from './types.js';
import {
  currentLogLevel,
  enabledModules,
  disabledModules,
  logHistory,
  pushHistory,
  shiftHistory,
} from './state.js';
import { getRuntimeConfig } from '../config/runtime.js';

/**
 * Format a log message with timestamp and module tag.
 */
export function formatMessage(module: string, message: string): string {
  const timestamp = performance.now().toFixed(1);
  return `[${timestamp}ms][${module}] ${message}`;
}

/**
 * Store log in history for later retrieval.
 */
export function storeLog(level: string, module: string, message: string, data?: unknown): void {
  pushHistory({
    time: Date.now(),
    perfTime: performance.now(),
    level,
    module,
    message,
    data,
  });

  const maxHistory = getRuntimeConfig().debug.logHistory.maxLogHistoryEntries;
  if (logHistory.length > maxHistory) {
    shiftHistory();
  }
}

/**
 * Check if logging is enabled for a module at a level.
 */
export function shouldLog(module: string, level: LogLevelValue): boolean {
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
 * Main logging interface.
 */
export const log = {
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