/**
 * DOPPLER Debug Module - Unified Logging and Tracing
 *
 * Single source of truth for all logging and debugging.
 *
 * ## Log Levels (verbosity - how much to show)
 *   silent  - nothing
 *   error   - errors only
 *   warn    - errors + warnings
 *   info    - normal operation (default)
 *   verbose - detailed info
 *   debug   - everything
 *
 * ## Trace Categories (what to show when tracing)
 *   loader  - model loading (shards, weights)
 *   kernels - GPU kernel execution
 *   logits  - logit computation
 *   embed   - embedding layer
 *   attn    - attention computation
 *   ffn     - feed-forward network
 *   kv      - KV cache operations
 *   sample  - token sampling
 *   buffers - GPU buffer stats (expensive!)
 *   perf    - timing info
 *   all     - everything
 *
 * ## Usage
 *   import { log, trace, setLogLevel, setTrace } from '../debug/index.js';
 *
 *   // Log levels (verbosity)
 *   log.info('Pipeline', 'Model loaded');
 *   log.verbose('Loader', 'Shard 0 from OPFS');
 *   log.debug('Attention', `heads=${numHeads}`);
 *
 *   // Trace categories (only logs if category enabled)
 *   trace.loader('Loading shard 0 from OPFS');
 *   trace.kernels('matmul M=1 K=1152 N=1024');
 *   trace.logits({ min: -2.3, max: 15.7 });
 *
 *   // Configure
 *   setLogLevel('verbose');
 *   setTrace('kernels,logits');       // enable specific
 *   setTrace('all,-buffers');         // all except buffers
 *   setTrace(false);                  // disable all
 *
 * ## CLI Flags -> URL Params (auto-mapped)
 *   --verbose, -v     ->  ?log=verbose
 *   --debug           ->  ?log=debug
 *   --quiet, -q       ->  ?log=silent
 *   --trace           ->  ?trace=all
 *   --trace kernels   ->  ?trace=kernels
 *   --trace all,-buf  ->  ?trace=all,-buffers
 *   --layers 0,5      ->  ?layers=0,5
 *   --break           ->  ?break=1
 *
 * @module debug
 */

// ============================================================================
// Re-exports from signals.ts
// ============================================================================

export {
  SIGNALS,
  type SignalType,
  type DonePayload,
  signalDone,
  signalResult,
  signalError,
  signalProgress,
} from './signals.js';

// ============================================================================
// Re-exports from config.ts
// ============================================================================

export {
  // Types and constants
  LOG_LEVELS,
  TRACE_CATEGORIES,
  type LogLevel,
  type LogLevelValue,
  type TraceCategory,
  type LogEntry,
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
  setBenchmarkMode,
  isBenchmarkMode,
  enableModules,
  disableModules,
  resetModuleFilters,
  setGPUDevice,
  initFromUrlParams,
} from './config.js';

// ============================================================================
// Re-exports from log.ts
// ============================================================================

export { log } from './log.js';

// ============================================================================
// Re-exports from trace.ts
// ============================================================================

export { trace } from './trace.js';

// ============================================================================
// Re-exports from tensor.ts
// ============================================================================

export {
  tensor,
  type TensorStats,
  type TensorCompareResult,
  type TensorHealthResult,
  type TensorInspectOptions,
} from './tensor.js';

// ============================================================================
// Re-exports from perf.ts
// ============================================================================

export { perf } from './perf.js';

// ============================================================================
// Re-exports from history.ts
// ============================================================================

export {
  getLogHistory,
  clearLogHistory,
  printLogSummary,
  getDebugSnapshot,
  type LogHistoryFilter,
  type DebugSnapshot,
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
  setBenchmarkMode,
  isBenchmarkMode,
  enableModules,
  disableModules,
  resetModuleFilters,
  setGPUDevice,
  initFromUrlParams,
} from './config.js';
import {
  getLogHistory,
  clearLogHistory,
  printLogSummary,
  getDebugSnapshot,
} from './history.js';

/**
 * DOPPLER debug API exposed to browser console.
 */
export interface DopplerDebugAPI {
  // Trace categories
  trace: typeof trace;
  setTrace: typeof setTrace;
  getTrace: typeof getTrace;
  // Log levels
  log: typeof log;
  setLogLevel: typeof setLogLevel;
  getLogLevel: typeof getLogLevel;
  // Tensor inspection
  tensor: typeof tensor;
  inspect: typeof tensor.inspect;
  // Performance
  perf: typeof perf;
  // Other
  setBenchmarkMode: typeof setBenchmarkMode;
  isBenchmarkMode: typeof isBenchmarkMode;
  // History
  getLogHistory: typeof getLogHistory;
  printLogSummary: typeof printLogSummary;
  getDebugSnapshot: typeof getDebugSnapshot;
  // Completion signals
  SIGNALS: typeof SIGNALS;
  signalDone: typeof signalDone;
  signalResult: typeof signalResult;
  signalError: typeof signalError;
  signalProgress: typeof signalProgress;
}

const DOPPLER_API: DopplerDebugAPI = {
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
  setBenchmarkMode,
  isBenchmarkMode,
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
  setBenchmarkMode,
  isBenchmarkMode,
  setGPUDevice,
  enableModules,
  disableModules,
  resetModuleFilters,
  getLogHistory,
  clearLogHistory,
  printLogSummary,
  getDebugSnapshot,
  initFromUrlParams,
  LOG_LEVELS,
  TRACE_CATEGORIES,
  // Completion signals
  SIGNALS,
  signalDone,
  signalResult,
  signalError,
  signalProgress,
};
