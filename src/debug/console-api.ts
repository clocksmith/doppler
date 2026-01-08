/**
 * Browser Console API
 *
 * @module debug/console-api
 */

import {
  type TraceCategory,
  type DebugSnapshot,
  LOG_LEVELS,
  SIGNALS,
} from './types.js';
import {
  currentLogLevel,
  enabledTraceCategories,
  enabledModules,
  disabledModules,
  logHistory,
  benchmarkMode,
  setBenchmarkModeState,
} from './state.js';
import { log, setLogLevel, getLogLevel } from './logger.js';
import { trace, setTrace, getTrace } from './tracer.js';
import { tensor } from './tensor.js';
import { perf } from './perf.js';
import {
  signalDone,
  signalResult,
  signalError,
  signalProgress,
} from './signals.js';
import { getLogHistory, printLogSummary, getDebugSnapshot } from './history.js'; // I'll create this next

/**
 * Initialize logging and tracing from URL parameters.
 */
export function initFromUrlParams(): void {
  if (typeof window === 'undefined') return;

  const params = new URLSearchParams(window.location.search);

  // Log level
  const logLevel = params.get('log');
  if (logLevel) {
    setLogLevel(logLevel);
  }

  // Trace categories
  const traceParam = params.get('trace');
  if (traceParam) {
    const layers = params.get('layers')?.split(',').map(Number).filter(n => !isNaN(n));
    const breakOn = params.get('break') === '1';
    setTrace(traceParam, { layers, breakOnAnomaly: breakOn });
  }

  // Debug mode (legacy param support)
  const debugParam = params.get('debug');
  if (debugParam === '1' && !traceParam) {
    setTrace('all');
    setLogLevel('verbose');
  }
}

/**
 * DOPPLER debug API exposed to browser console.
 */
export interface DopplerDebugAPI {
  trace: typeof trace;
  setTrace: typeof setTrace;
  getTrace: typeof getTrace;
  log: typeof log;
  setLogLevel: typeof setLogLevel;
  getLogLevel: typeof getLogLevel;
  tensor: typeof tensor;
  inspect: typeof tensor.inspect;
  perf: typeof perf;
  setBenchmarkMode: (enabled: boolean) => void;
  isBenchmarkMode: () => boolean;
  getLogHistory: typeof getLogHistory;
  printLogSummary: typeof printLogSummary;
  getDebugSnapshot: typeof getDebugSnapshot;
  SIGNALS: typeof SIGNALS;
  signalDone: typeof signalDone;
  signalResult: typeof signalResult;
  signalError: typeof signalError;
  signalProgress: typeof signalProgress;
}

/**
 * Enable benchmark mode - silences all console.log/debug/info calls.
 */
const originalConsoleLog = console.log;
const originalConsoleDebug = console.debug;
const originalConsoleInfo = console.info;

export function setBenchmarkMode(enabled: boolean): void {
  setBenchmarkModeState(enabled);
  if (enabled) {
    const noop = () => {};
    console.log = noop;
    console.debug = noop;
    console.info = noop;
    originalConsoleLog('[Doppler] Benchmark mode enabled - logging silenced');
  } else {
    console.log = originalConsoleLog;
    console.debug = originalConsoleDebug;
    console.info = originalConsoleInfo;
    console.log('[Doppler] Benchmark mode disabled - logging restored');
  }
}

export function isBenchmarkMode(): boolean {
  return benchmarkMode;
}

export const DOPPLER_API: DopplerDebugAPI = {
  trace,
  setTrace,
  getTrace,
  log,
  setLogLevel,
  getLogLevel,
  tensor,
  inspect: tensor.inspect.bind(tensor),
  perf,
  setBenchmarkMode,
  isBenchmarkMode,
  getLogHistory,
  printLogSummary,
  getDebugSnapshot,
  SIGNALS,
  signalDone,
  signalResult,
  signalError,
  signalProgress,
};

// Expose to window in browser environment
if (typeof window !== 'undefined') {
  (window as any).DOPPLER = {
    ...((window as any).DOPPLER || {}),
    ...DOPPLER_API,
  };

  // Auto-init from URL params on load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initFromUrlParams);
  } else {
    initFromUrlParams();
  }
}
