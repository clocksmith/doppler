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
  setBenchmarkMode,
  isBenchmarkMode,
  enableModules,
  disableModules,
  resetModuleFilters,
  setGPUDevice,
  initFromUrlParams,
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
  window.DOPPLER = {
    ...(window.DOPPLER || {}),
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
  setSilentMode,
  isSilentMode,
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
