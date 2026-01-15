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
 * ## Config-Only Overrides
 *   Configure log level and trace categories via runtime config.
 *
 * @module debug
 */

// ============================================================================
// Re-exports from signals.js
// ============================================================================

export {
  SIGNALS,
  SignalType,
  DonePayload,
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
  LogLevel,
  LogLevelValue,
  TraceCategory,
  LogEntry,
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
  setBenchmarkMode,
  isBenchmarkMode,
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

export {
  tensor,
  TensorStats,
  TensorCompareResult,
  TensorHealthResult,
  TensorInspectOptions,
} from './tensor.js';

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
  LogHistoryFilter,
  DebugSnapshot,
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
  setLogLevel,
  getLogLevel,
  setTrace,
  getTrace,
  isTraceEnabled,
  setSilentMode,
  isSilentMode,
  setBenchmarkMode,
  isBenchmarkMode,
} from './config.js';
import {
  getLogHistory,
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
  setSilentMode: typeof setSilentMode;
  isSilentMode: typeof isSilentMode;
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

// ============================================================================
// Default Export
// ============================================================================

declare const _default: {
  log: typeof log;
  trace: typeof trace;
  tensor: typeof tensor;
  perf: typeof perf;
  setLogLevel: typeof setLogLevel;
  getLogLevel: typeof getLogLevel;
  setTrace: typeof setTrace;
  getTrace: typeof getTrace;
  isTraceEnabled: typeof isTraceEnabled;
  setSilentMode: typeof setSilentMode;
  isSilentMode: typeof isSilentMode;
  setBenchmarkMode: typeof setBenchmarkMode;
  isBenchmarkMode: typeof isBenchmarkMode;
  setGPUDevice: (device: GPUDevice) => void;
  enableModules: (...modules: string[]) => void;
  disableModules: (...modules: string[]) => void;
  resetModuleFilters: () => void;
  getLogHistory: typeof getLogHistory;
  clearLogHistory: () => void;
  printLogSummary: typeof printLogSummary;
  getDebugSnapshot: typeof getDebugSnapshot;
  LOG_LEVELS: {
    readonly DEBUG: 0;
    readonly VERBOSE: 1;
    readonly INFO: 2;
    readonly WARN: 3;
    readonly ERROR: 4;
    readonly SILENT: 5;
  };
  TRACE_CATEGORIES: readonly [
    'loader',
    'kernels',
    'logits',
    'embed',
    'attn',
    'ffn',
    'kv',
    'sample',
    'buffers',
    'perf',
  ];
  // Completion signals
  SIGNALS: typeof SIGNALS;
  signalDone: typeof signalDone;
  signalResult: typeof signalResult;
  signalError: typeof signalError;
  signalProgress: typeof signalProgress;
};

export default _default;

declare global {
  interface Window {
    DOPPLER?: DopplerDebugAPI;
  }
}
