
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

// Opt in from an application entrypoint, never from a reusable import.
export function installDebugGlobal(target = globalThis) {
  if (!target || (typeof target !== 'object' && typeof target !== 'function')) {
    throw new Error('installDebugGlobal requires a global object.');
  }
  const previous = Object.getOwnPropertyDescriptor(target, 'DOPPLER');
  const api = Object.freeze({ ...target.DOPPLER, ...DOPPLER_API });
  Object.defineProperty(target, 'DOPPLER', { value: api, configurable: true, writable: true, enumerable: true });
  return () => {
    if (target.DOPPLER !== api) return;
    if (previous) Object.defineProperty(target, 'DOPPLER', previous);
    else delete target.DOPPLER;
  };
}

// ============================================================================
// Default Export
// ============================================================================

export default {
  log,
  trace,
  tensor,
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
