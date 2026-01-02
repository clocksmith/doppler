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
 * ## CLI Flags → URL Params (auto-mapped)
 *   --verbose, -v     →  ?log=verbose
 *   --debug           →  ?log=debug
 *   --quiet, -q       →  ?log=silent
 *   --trace           →  ?trace=all
 *   --trace kernels   →  ?trace=kernels
 *   --trace all,-buf  →  ?trace=all,-buffers
 *   --layers 0,5      →  ?layers=0,5
 *   --break           →  ?break=1
 *
 * @module debug
 */

import type { DebugConfigSchema } from '../config/schema/debug.schema.js';
import { getRuntimeConfig } from '../config/runtime.js';

// ============================================================================
// Types and Interfaces
// ============================================================================

/**
 * Log level values (higher = less verbose)
 */
export const LOG_LEVELS = {
  DEBUG: 0,
  VERBOSE: 1,
  INFO: 2,
  WARN: 3,
  ERROR: 4,
  SILENT: 5,
} as const;

/**
 * Trace categories
 */
export const TRACE_CATEGORIES = [
  'loader',   // Model loading (shards, weights)
  'kernels',  // GPU kernel execution
  'logits',   // Logit computation
  'embed',    // Embedding layer
  'attn',     // Attention
  'ffn',      // Feed-forward
  'kv',       // KV cache
  'sample',   // Token sampling
  'buffers',  // GPU buffer stats (expensive!)
  'perf',     // Timing
] as const;

export type TraceCategory = (typeof TRACE_CATEGORIES)[number];

export type LogLevel = keyof typeof LOG_LEVELS;
export type LogLevelValue = (typeof LOG_LEVELS)[LogLevel];

/**
 * Log entry for history
 */
export interface LogEntry {
  time: number;
  perfTime: number;
  level: string;
  module: string;
  message: string;
  data?: unknown;
}

/**
 * Tensor statistics
 */
export interface TensorStats {
  label: string;
  shape: number[];
  size: number;
  isGPU: boolean;
  min: number;
  max: number;
  mean: number;
  std: number;
  nanCount: number;
  infCount: number;
  zeroCount: number;
  zeroPercent: string;
  first: string[];
  last: string[];
}

/**
 * Tensor comparison result
 */
export interface TensorCompareResult {
  label: string;
  match: boolean;
  maxDiff: number;
  maxDiffIdx: number;
  avgDiff: number;
  mismatchCount: number;
  mismatchPercent: string;
  error?: string;
}

/**
 * Tensor health check result
 */
export interface TensorHealthResult {
  label: string;
  healthy: boolean;
  issues: string[];
}

/**
 * Tensor inspect options
 */
export interface TensorInspectOptions {
  shape?: number[];
  maxPrint?: number;
  checkNaN?: boolean;
}

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
// Global State
// ============================================================================

let currentLogLevel: LogLevelValue = LOG_LEVELS.INFO;
let enabledModules = new Set<string>();
let disabledModules = new Set<string>();
let logHistory: LogEntry[] = [];

// GPU device reference for tensor inspection
let gpuDevice: GPUDevice | null = null;

// Trace categories state
let enabledTraceCategories = new Set<TraceCategory>();
let traceLayerFilter: number[] = [];  // Empty = all layers
let traceDecodeStep = 0;
let traceMaxDecodeSteps = 0;  // 0 = unlimited
let traceBreakOnAnomaly = false;

// ============================================================================
// Configuration Functions
// ============================================================================

/**
 * Set the global log level.
 */
export function setLogLevel(level: string): void {
  const levelMap: Record<string, LogLevelValue> = {
    debug: LOG_LEVELS.DEBUG,
    verbose: LOG_LEVELS.VERBOSE,
    info: LOG_LEVELS.INFO,
    warn: LOG_LEVELS.WARN,
    error: LOG_LEVELS.ERROR,
    silent: LOG_LEVELS.SILENT,
  };
  currentLogLevel = levelMap[level.toLowerCase()] ?? LOG_LEVELS.INFO;
  console.log(`[Doppler] Log level set to: ${level.toUpperCase()}`);
}

/**
 * Get current log level name.
 */
export function getLogLevel(): string {
  for (const [name, value] of Object.entries(LOG_LEVELS)) {
    if (value === currentLogLevel) return name.toLowerCase();
  }
  return 'info';
}

/**
 * Set trace categories.
 *
 * @param categories - Comma-separated categories, 'all', false to disable, or array
 *   Examples:
 *   - 'kernels,logits' - enable kernels and logits
 *   - 'all' - enable all categories
 *   - 'all,-buffers' - all except buffers
 *   - false - disable all tracing
 *   - ['kernels', 'logits'] - array form
 */
export function setTrace(
  categories: string | TraceCategory[] | false,
  options?: { layers?: number[]; maxDecodeSteps?: number; breakOnAnomaly?: boolean }
): void {
  // Handle false = disable all
  if (categories === false) {
    enabledTraceCategories.clear();
    console.log('[Doppler] Trace disabled');
    return;
  }

  // Parse string into array
  const catArray = typeof categories === 'string'
    ? categories.split(',').map(s => s.trim())
    : categories;

  // Clear and rebuild
  enabledTraceCategories.clear();

  // Check for 'all' first
  const hasAll = catArray.includes('all');
  if (hasAll) {
    for (const cat of TRACE_CATEGORIES) {
      enabledTraceCategories.add(cat);
    }
  }

  // Add inclusions and handle exclusions (prefixed with -)
  for (const cat of catArray) {
    if (cat === 'all') continue;

    if (cat.startsWith('-')) {
      const exclude = cat.slice(1) as TraceCategory;
      enabledTraceCategories.delete(exclude);
    } else if (TRACE_CATEGORIES.includes(cat as TraceCategory)) {
      enabledTraceCategories.add(cat as TraceCategory);
    }
  }

  // Apply options
  if (options?.layers) {
    traceLayerFilter = options.layers;
  }
  if (options?.maxDecodeSteps !== undefined) {
    traceMaxDecodeSteps = options.maxDecodeSteps;
  }
  if (options?.breakOnAnomaly !== undefined) {
    traceBreakOnAnomaly = options.breakOnAnomaly;
  }

  const enabled = [...enabledTraceCategories].join(',') || 'none';
  console.log(`[Doppler] Trace categories: ${enabled}`);
}

/**
 * Apply debug config defaults unless URL params already set them.
 */
export function applyDebugConfig(
  config: DebugConfigSchema,
  options: { respectUrlParams?: boolean } = {}
): void {
  const respectUrlParams = options.respectUrlParams !== false;
  let hasLogParam = false;
  let hasTraceParam = false;

  if (respectUrlParams && typeof window !== 'undefined') {
    const params = new URLSearchParams(window.location.search);
    hasLogParam = params.has('log');
    hasTraceParam = params.has('trace');
  }

  if (!hasLogParam && config.logLevel?.defaultLogLevel) {
    const desired = config.logLevel.defaultLogLevel;
    if (desired && desired !== getLogLevel()) {
      setLogLevel(desired);
    }
  }

  if (!hasTraceParam) {
    if (config.trace?.enabled) {
      const categories = config.trace.categories?.length
        ? config.trace.categories
        : ['all'];
      setTrace(categories, {
        layers: config.trace.layers ?? undefined,
        maxDecodeSteps: config.trace.maxDecodeSteps || undefined,
      });
    } else if (getTrace().length > 0) {
      setTrace(false);
    }
  }
}

/**
 * Get enabled trace categories.
 */
export function getTrace(): TraceCategory[] {
  return [...enabledTraceCategories];
}

/**
 * Check if a trace category is enabled.
 */
export function isTraceEnabled(category: TraceCategory, layerIdx?: number): boolean {
  if (!enabledTraceCategories.has(category)) return false;

  // Check layer filter
  if (layerIdx !== undefined && traceLayerFilter.length > 0) {
    if (!traceLayerFilter.includes(layerIdx)) return false;
  }

  // Check decode step limit
  if (traceMaxDecodeSteps > 0 && traceDecodeStep > traceMaxDecodeSteps) {
    return false;
  }

  return true;
}

/**
 * Increment decode step counter (call after each decode step).
 */
export function incrementDecodeStep(): number {
  return ++traceDecodeStep;
}

/**
 * Reset decode step counter (call at start of generation).
 */
export function resetDecodeStep(): void {
  traceDecodeStep = 0;
}

/**
 * Get current decode step.
 */
export function getDecodeStep(): number {
  return traceDecodeStep;
}

/**
 * Check if we should break on anomaly.
 */
export function shouldBreakOnAnomaly(): boolean {
  return traceBreakOnAnomaly;
}

// Benchmark mode state
let benchmarkMode = false;
const originalConsoleLog = console.log;
const originalConsoleDebug = console.debug;
const originalConsoleInfo = console.info;

/**
 * Enable benchmark mode - silences all console.log/debug/info calls.
 */
export function setBenchmarkMode(enabled: boolean): void {
  benchmarkMode = enabled;
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

/**
 * Check if benchmark mode is active.
 */
export function isBenchmarkMode(): boolean {
  return benchmarkMode;
}

/**
 * Enable logging for specific modules only.
 */
export function enableModules(...modules: string[]): void {
  enabledModules = new Set(modules.map((m) => m.toLowerCase()));
  console.log(`[Doppler] Enabled modules: ${modules.join(', ')}`);
}

/**
 * Disable logging for specific modules.
 */
export function disableModules(...modules: string[]): void {
  for (const m of modules) {
    disabledModules.add(m.toLowerCase());
  }
  console.log(`[Doppler] Disabled modules: ${modules.join(', ')}`);
}

/**
 * Reset module filters.
 */
export function resetModuleFilters(): void {
  enabledModules.clear();
  disabledModules.clear();
}

/**
 * Set GPU device for tensor inspection.
 */
export function setGPUDevice(device: GPUDevice): void {
  gpuDevice = device;
}

// ============================================================================
// URL Parameter Auto-Detection
// ============================================================================

/**
 * Initialize logging and tracing from URL parameters.
 * Called automatically in browser environment.
 *
 * Supported params:
 *   ?log=verbose          - Set log level
 *   ?trace=kernels,logits - Enable specific trace categories
 *   ?trace=all,-buffers   - All categories except buffers
 *   ?layers=0,5           - Filter to specific layers
 *   ?break=1              - Break on anomaly (NaN/explosion)
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
 * Format a trace message with category tag.
 */
function formatTraceMessage(category: TraceCategory, message: string, layerIdx?: number): string {
  const timestamp = performance.now().toFixed(1);
  const layerTag = layerIdx !== undefined ? `L${layerIdx}:` : '';
  return `[${timestamp}ms][TRACE:${category}] ${layerTag}${message}`;
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

/**
 * F16 to F32 conversion helper.
 */
function f16ToF32(h: number): number {
  const sign = (h >> 15) & 0x1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;

  if (exp === 0) {
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024);
  } else if (exp === 31) {
    return mant === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }

  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
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

// ============================================================================
// Trace Interface
// ============================================================================

/**
 * Trace logging interface - only logs if category is enabled.
 */
export const trace = {
  /**
   * Trace model loading operations.
   */
  loader(message: string, data?: unknown): void {
    if (!isTraceEnabled('loader')) return;
    const formatted = formatTraceMessage('loader', message);
    storeLog('TRACE:loader', 'Loader', message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace kernel execution.
   */
  kernels(message: string, data?: unknown): void {
    if (!isTraceEnabled('kernels')) return;
    const formatted = formatTraceMessage('kernels', message);
    storeLog('TRACE:kernels', 'Kernels', message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace logit computation.
   */
  logits(message: string, data?: unknown): void {
    if (!isTraceEnabled('logits')) return;
    const formatted = formatTraceMessage('logits', message);
    storeLog('TRACE:logits', 'Logits', message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace embedding layer.
   */
  embed(message: string, data?: unknown): void {
    if (!isTraceEnabled('embed')) return;
    const formatted = formatTraceMessage('embed', message);
    storeLog('TRACE:embed', 'Embed', message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace attention computation.
   */
  attn(layerIdx: number, message: string, data?: unknown): void {
    if (!isTraceEnabled('attn', layerIdx)) return;
    const formatted = formatTraceMessage('attn', message, layerIdx);
    storeLog('TRACE:attn', `Attn:L${layerIdx}`, message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace feed-forward network.
   */
  ffn(layerIdx: number, message: string, data?: unknown): void {
    if (!isTraceEnabled('ffn', layerIdx)) return;
    const formatted = formatTraceMessage('ffn', message, layerIdx);
    storeLog('TRACE:ffn', `FFN:L${layerIdx}`, message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace KV cache operations.
   */
  kv(layerIdx: number, message: string, data?: unknown): void {
    if (!isTraceEnabled('kv', layerIdx)) return;
    const formatted = formatTraceMessage('kv', message, layerIdx);
    storeLog('TRACE:kv', `KV:L${layerIdx}`, message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace token sampling.
   */
  sample(message: string, data?: unknown): void {
    if (!isTraceEnabled('sample')) return;
    const formatted = formatTraceMessage('sample', message);
    storeLog('TRACE:sample', 'Sample', message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace buffer stats (expensive - requires GPU readback).
   */
  buffers(message: string, data?: unknown): void {
    if (!isTraceEnabled('buffers')) return;
    const formatted = formatTraceMessage('buffers', message);
    storeLog('TRACE:buffers', 'Buffers', message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },

  /**
   * Trace performance timing.
   */
  perf(message: string, data?: unknown): void {
    if (!isTraceEnabled('perf')) return;
    const formatted = formatTraceMessage('perf', message);
    storeLog('TRACE:perf', 'Perf', message, data);
    if (data !== undefined) {
      console.log(formatted, data);
    } else {
      console.log(formatted);
    }
  },
};

// ============================================================================
// Tensor Inspection Interface
// ============================================================================

/**
 * Tensor inspection utilities.
 */
export const tensor = {
  /**
   * Inspect a GPU or CPU tensor and log statistics.
   */
  async inspect(
    buffer: GPUBuffer | Float32Array | Float64Array | Uint16Array,
    label: string,
    options: TensorInspectOptions = {}
  ): Promise<TensorStats | null> {
    const { shape = [], maxPrint = 8, checkNaN = true } = options;

    let data: Float32Array;
    let isGPU = false;

    // Handle GPU buffers
    if (buffer && typeof (buffer as GPUBuffer).mapAsync === 'function') {
      const gpuBuffer = buffer as GPUBuffer;
      await gpuBuffer.mapAsync(GPUMapMode.READ);
      data = new Float32Array(gpuBuffer.getMappedRange().slice(0));
      gpuBuffer.unmap();
    } else if (buffer && (buffer as GPUBuffer).size !== undefined && gpuDevice) {
      isGPU = true;
      const gpuBuffer = buffer as GPUBuffer;
      const readSize = Math.min(gpuBuffer.size, 4096);
      const staging = gpuDevice.createBuffer({
        label: `debug_staging_${label}`,
        size: readSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      const encoder = gpuDevice.createCommandEncoder();
      encoder.copyBufferToBuffer(gpuBuffer, 0, staging, 0, readSize);
      gpuDevice.queue.submit([encoder.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      data = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
    } else if (buffer instanceof Float32Array || buffer instanceof Float64Array) {
      data = buffer instanceof Float32Array ? buffer : new Float32Array(buffer);
    } else if (buffer instanceof Uint16Array) {
      data = new Float32Array(buffer.length);
      for (let i = 0; i < buffer.length; i++) {
        data[i] = f16ToF32(buffer[i]);
      }
    } else {
      log.warn('Debug', `Cannot inspect tensor "${label}": unknown type`);
      return null;
    }

    // Compute statistics
    let min = Infinity,
      max = -Infinity,
      sum = 0,
      sumSq = 0;
    let nanCount = 0,
      infCount = 0,
      zeroCount = 0;

    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (Number.isNaN(v)) {
        nanCount++;
        continue;
      }
      if (!Number.isFinite(v)) {
        infCount++;
        continue;
      }
      if (v === 0) zeroCount++;
      min = Math.min(min, v);
      max = Math.max(max, v);
      sum += v;
      sumSq += v * v;
    }

    const validCount = data.length - nanCount - infCount;
    const mean = validCount > 0 ? sum / validCount : 0;
    const variance = validCount > 0 ? sumSq / validCount - mean * mean : 0;
    const std = Math.sqrt(Math.max(0, variance));

    const stats: TensorStats = {
      label,
      shape,
      size: data.length,
      isGPU,
      min,
      max,
      mean,
      std,
      nanCount,
      infCount,
      zeroCount,
      zeroPercent: ((zeroCount / data.length) * 100).toFixed(1),
      first: Array.from(data.slice(0, maxPrint)).map((v) => v.toFixed(4)),
      last: Array.from(data.slice(-maxPrint)).map((v) => v.toFixed(4)),
    };

    const shapeStr = shape.length > 0 ? `[${shape.join('x')}]` : `[${data.length}]`;
    log.debug(
      'Tensor',
      `${label} ${shapeStr}: min=${min.toFixed(4)}, max=${max.toFixed(4)}, mean=${mean.toFixed(4)}, std=${std.toFixed(4)}`
    );

    if (checkNaN && (nanCount > 0 || infCount > 0)) {
      log.warn('Tensor', `${label} has ${nanCount} NaN and ${infCount} Inf values!`);
    }

    return stats;
  },

  /**
   * Compare two tensors element-wise.
   */
  compare(
    a: Float32Array,
    b: Float32Array,
    label: string,
    tolerance = 1e-5
  ): TensorCompareResult {
    if (a.length !== b.length) {
      log.error('Tensor', `${label}: size mismatch ${a.length} vs ${b.length}`);
      return { label, match: false, error: 'size_mismatch', maxDiff: 0, maxDiffIdx: 0, avgDiff: 0, mismatchCount: 0, mismatchPercent: '0' };
    }

    let maxDiff = 0,
      maxDiffIdx = 0;
    let sumDiff = 0;
    let mismatchCount = 0;

    for (let i = 0; i < a.length; i++) {
      const diff = Math.abs(a[i] - b[i]);
      sumDiff += diff;
      if (diff > maxDiff) {
        maxDiff = diff;
        maxDiffIdx = i;
      }
      if (diff > tolerance) {
        mismatchCount++;
      }
    }

    const avgDiff = sumDiff / a.length;
    const match = mismatchCount === 0;

    const result: TensorCompareResult = {
      label,
      match,
      maxDiff,
      maxDiffIdx,
      avgDiff,
      mismatchCount,
      mismatchPercent: ((mismatchCount / a.length) * 100).toFixed(2),
    };

    if (match) {
      log.debug('Tensor', `${label}: MATCH (maxDiff=${maxDiff.toExponential(2)})`);
    } else {
      log.warn(
        'Tensor',
        `${label}: MISMATCH ${mismatchCount}/${a.length} (${result.mismatchPercent}%) maxDiff=${maxDiff.toFixed(6)} at idx=${maxDiffIdx}`
      );
    }

    return result;
  },

  /**
   * Check tensor for common issues.
   */
  healthCheck(data: Float32Array, label: string): TensorHealthResult {
    const issues: string[] = [];

    const allZero = data.every((v) => v === 0);
    if (allZero) {
      issues.push('ALL_ZEROS');
    }

    const hasNaN = data.some((v) => Number.isNaN(v));
    const hasInf = data.some((v) => !Number.isFinite(v) && !Number.isNaN(v));
    if (hasNaN) issues.push('HAS_NAN');
    if (hasInf) issues.push('HAS_INF');

    const maxAbs = Math.max(...Array.from(data).map(Math.abs).filter(Number.isFinite));
    if (maxAbs > 1e6) issues.push(`EXTREME_VALUES (max=${maxAbs.toExponential(2)})`);

    const tinyCount = data.filter((v) => Math.abs(v) > 0 && Math.abs(v) < 1e-30).length;
    if (tinyCount > data.length * 0.1) {
      issues.push(`POTENTIAL_UNDERFLOW (${tinyCount} tiny values)`);
    }

    const healthy = issues.length === 0;

    if (healthy) {
      log.debug('Tensor', `${label}: healthy`);
    } else {
      log.warn('Tensor', `${label}: issues found - ${issues.join(', ')}`);
    }

    return { label, healthy, issues };
  },
};

// ============================================================================
// Performance Timing Interface
// ============================================================================

/**
 * Performance timing utilities.
 */
export const perf = {
  marks: new Map<string, number>(),

  /**
   * Start a timing mark.
   */
  mark(label: string): void {
    this.marks.set(label, performance.now());
  },

  /**
   * End a timing mark and log duration.
   */
  measure(label: string, module = 'Perf'): number {
    const start = this.marks.get(label);
    if (start === undefined) {
      log.warn(module, `No mark found for "${label}"`);
      return 0;
    }

    const duration = performance.now() - start;
    this.marks.delete(label);
    log.debug(module, `${label}: ${duration.toFixed(2)}ms`);
    return duration;
  },

  /**
   * Time an async operation.
   */
  async time<T>(label: string, fn: () => Promise<T>): Promise<{ result: T; durationMs: number }> {
    const start = performance.now();
    const result = await fn();
    const durationMs = performance.now() - start;
    log.debug('Perf', `${label}: ${durationMs.toFixed(2)}ms`);
    return { result, durationMs };
  },
};

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
  logHistory = [];
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


// ============================================================================
// Browser Console Global API
// ============================================================================

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
};

// ============================================================================
// Re-exports from debug utilities
// ============================================================================

// Tensor debug utilities (SafeTensors/RDRR comparison)
export {
  bf16ToF32,
  readSafetensorsTensor,
  readRDRRTensor,
  checkTensorStats,
} from './tensor.js';
