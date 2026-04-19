/**
 * Debug module — central re-export hub for logging, tracing, and signals.
 *
 * CONCURRENT PIPELINE WARNING: The debug state (log level, trace categories,
 * module filters, decode step counter) is global and shared across all
 * pipelines. When multiple pipelines run concurrently, debug configuration
 * changes from one pipeline will affect all others. If you need per-pipeline
 * debug isolation, pass a `debugContext` options object to your pipeline and
 * use it to namespace log entries rather than mutating global state.
 */

// ============================================================================
// Re-exports from signals.js
// ============================================================================

export {
  SIGNALS,
  signalDone,
  signalResult,
  signalError,
  signalProgress,
} from './signals.js';

// ============================================================================
// Re-exports from config.js
// ============================================================================

export {
  // Types and constants
  LOG_LEVELS,
  TRACE_CATEGORIES,
  // Configuration functions
  setLogLevel,
  getLogLevel,
  setTrace,
  getTrace,
  applyDebugConfig,
  isTraceEnabled,
  incrementDecodeStep,
  resetDecodeStep,
  getDecodeStep,
  shouldBreakOnAnomaly,
  setSilentMode,
  isSilentMode,
  enableModules,
  disableModules,
  resetModuleFilters,
  setGPUDevice,
} from './config.js';

// ============================================================================
// Re-exports from log.js
// ============================================================================

export { log } from './log.js';

// ============================================================================
// Re-exports from trace.js
// ============================================================================

export { trace } from './trace.js';

// ============================================================================
// Re-exports from tensor.js
// ============================================================================

export { tensor } from './tensor.js';

// ============================================================================
// Re-exports from capture-policy.js
// ============================================================================

export {
  CAPTURE_LEVELS,
  resolveCapturePolicy,
  escalateCaptureLevel,
  buildCaptureArtifact,
  createEscalationPolicy,
  createDefaultCaptureConfig,
  validateCaptureConfig,
} from './capture-policy.js';

// ============================================================================
// Re-exports from perf.js
// ============================================================================

export { perf } from './perf.js';

// ============================================================================
// Re-exports from history.js
// ============================================================================

export {
  getLogHistory,
  clearLogHistory,
  printLogSummary,
  getDebugSnapshot,
} from './history.js';

// ============================================================================
// Browser Console Global API
// ============================================================================

import { log } from './log.js';
import { trace } from './trace.js';
import { tensor } from './tensor.js';
import { perf } from './perf.js';
import {
  SIGNALS,
  signalDone,
  signalResult,
  signalError,
  signalProgress,
} from './signals.js';
import {
  LOG_LEVELS,
  TRACE_CATEGORIES,
  setLogLevel,
  getLogLevel,
  setTrace,
  getTrace,
  isTraceEnabled,
  setSilentMode,
  isSilentMode,
  enableModules,
  disableModules,
  resetModuleFilters,
  setGPUDevice,
} from './config.js';
import {
  getLogHistory,
  clearLogHistory,
  printLogSummary,
  getDebugSnapshot,
} from './history.js';

const DOPPLER_API = {
  // Trace categories
  trace,
  setTrace,
  getTrace,
  // Log levels
  log,
  setLogLevel,
  getLogLevel,
  // Tensor inspection
  tensor,
  inspect: tensor.inspect.bind(tensor),
  // Performance
  perf,
  // Other
  setSilentMode,
  isSilentMode,
  // History
  getLogHistory,
  printLogSummary,
  getDebugSnapshot,
  // Completion signals
  SIGNALS,
  signalDone,
  signalResult,
  signalError,
  signalProgress,
};

// Expose to globalThis in browser-like environments
if (typeof globalThis !== 'undefined') {
  globalThis.DOPPLER = {
    ...(globalThis.DOPPLER || {}),
    ...DOPPLER_API,
  };

}

// ============================================================================
// Default Export
// ============================================================================

export default {
  log,
  trace,
  tensor,
  perf,
  setLogLevel,
  getLogLevel,
  setTrace,
  getTrace,
  isTraceEnabled,
  setSilentMode,
  isSilentMode,
  setGPUDevice,
  enableModules,
  disableModules,
  resetModuleFilters,
  getLogHistory,
  clearLogHistory,
  printLogSummary,
  getDebugSnapshot,
  LOG_LEVELS,
  TRACE_CATEGORIES,
  // Completion signals
  SIGNALS,
  signalDone,
  signalResult,
  signalError,
  signalProgress,
};
