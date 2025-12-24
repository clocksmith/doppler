/**
 * DOPPLER Debug Module - Unified Logging, Debugging, and Testing
 *
 * Centralizes all debug/logging functionality across the DOPPLER project.
 * Provides consistent log levels, module tags, tensor inspection, and
 * integration with GPU profiler.
 *
 * Usage:
 *   import { log, tensor, setLogLevel } from '../debug/index.js';
 *
 *   log.info('Pipeline', 'Model loaded');
 *   log.debug('Attention', `heads=${numHeads}, dim=${headDim}`);
 *   tensor.inspect(buffer, 'qkv_output', { shape: [numTokens, hiddenSize] });
 *   setLogLevel('debug'); // Enable all debug logs
 *
 * @module debug
 */

// ============================================================================
// Types and Interfaces
// ============================================================================

/**
 * Log level values
 */
export const LOG_LEVELS = {
  DEBUG: 0,
  INFO: 1,
  WARN: 2,
  ERROR: 3,
  SILENT: 4,
} as const;

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
const MAX_HISTORY = 1000;

// GPU device reference for tensor inspection
let gpuDevice: GPUDevice | null = null;

// ============================================================================
// Configuration Functions
// ============================================================================

/**
 * Set the global log level.
 */
export function setLogLevel(level: string): void {
  const levelMap: Record<string, LogLevelValue> = {
    debug: LOG_LEVELS.DEBUG,
    info: LOG_LEVELS.INFO,
    warn: LOG_LEVELS.WARN,
    error: LOG_LEVELS.ERROR,
    silent: LOG_LEVELS.SILENT,
  };
  currentLogLevel = levelMap[level.toLowerCase()] ?? LOG_LEVELS.INFO;
  console.log(`[DOPPLER] Log level set to: ${level.toUpperCase()}`);
}

// Benchmark mode state
let benchmarkMode = false;
const originalConsoleLog = console.log;
const originalConsoleDebug = console.debug;
const originalConsoleInfo = console.info;

/**
 * Enable benchmark mode - silences all console.log/debug/info calls.
 * Use this during performance benchmarks to eliminate logging overhead.
 * Errors and warnings are still logged.
 */
export function setBenchmarkMode(enabled: boolean): void {
  benchmarkMode = enabled;
  if (enabled) {
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    const noop = () => {};
    console.log = noop;
    console.debug = noop;
    console.info = noop;
    originalConsoleLog('[DOPPLER] Benchmark mode enabled - logging silenced');
  } else {
    console.log = originalConsoleLog;
    console.debug = originalConsoleDebug;
    console.info = originalConsoleInfo;
    console.log('[DOPPLER] Benchmark mode disabled - logging restored');
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
  console.log(`[DOPPLER] Enabled modules: ${modules.join(', ')}`);
}

/**
 * Disable logging for specific modules.
 */
export function disableModules(...modules: string[]): void {
  for (const m of modules) {
    disabledModules.add(m.toLowerCase());
  }
  console.log(`[DOPPLER] Disabled modules: ${modules.join(', ')}`);
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

  if (logHistory.length > MAX_HISTORY) {
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
   * Debug level logging (verbose).
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
// Pipeline Debug Categories Integration
// ============================================================================

import {
  setDebugCategories,
  resetDebugConfig,
  getDebugConfig,
  DEBUG_PRESETS,
  type DebugCategory,
  type DebugConfig as PipelineDebugConfig,
} from '../inference/pipeline/debug-utils.js';

export type { DebugCategory, PipelineDebugConfig };
export { DEBUG_PRESETS };

/**
 * Main debug control function for pipeline categories.
 *
 * Usage in browser console:
 *   DOPPLER.debug({ embed: true, logits: true })  // Enable categories
 *   DOPPLER.debug('quick')                         // Use preset
 *   DOPPLER.debug('off')                           // Disable all
 *   DOPPLER.debug()                                // Show current config
 *
 * @param config - Categories object, preset name, or 'off'
 * @param options - Additional options (layers, maxDecodeSteps, etc.)
 */
export function debug(
  config?: Partial<Record<DebugCategory, boolean>> | keyof typeof DEBUG_PRESETS | 'off',
  options?: Partial<Omit<PipelineDebugConfig, 'categories'>>
): PipelineDebugConfig {
  if (config === undefined) {
    const current = getDebugConfig();
    console.log('Pipeline debug config:', current);
    console.log('Presets:', Object.keys(DEBUG_PRESETS).join(', '), '+ off');
    console.log('Categories: embed, layer, attn, ffn, kv, logits, sample, io, perf, all');
    return current;
  }

  if (config === 'off') {
    resetDebugConfig();
    console.log('Pipeline debug disabled');
    return getDebugConfig();
  }

  if (typeof config === 'string') {
    const preset = DEBUG_PRESETS[config as keyof typeof DEBUG_PRESETS];
    if (preset) {
      setDebugCategories(preset, options);
      console.log(`Pipeline debug preset '${config}' enabled:`, preset);
    } else {
      console.error(`Unknown preset: ${config}. Available: ${Object.keys(DEBUG_PRESETS).join(', ')}`);
    }
    return getDebugConfig();
  }

  setDebugCategories(config, options);
  console.log('Pipeline debug categories:', config);
  return getDebugConfig();
}

/**
 * Quick enable for specific layer debugging.
 */
export function debugLayer(
  layerIdx: number | number[],
  categories?: Partial<Record<DebugCategory, boolean>>
): void {
  const layers = Array.isArray(layerIdx) ? layerIdx : [layerIdx];
  setDebugCategories(categories || { layer: true, attn: true, ffn: true }, { layers });
  console.log(`Debug enabled for layer(s) ${layers.join(', ')}`);
}

/**
 * Quick buffer stats toggle (expensive - requires GPU readback).
 */
export function debugBuffers(enabled: boolean = true): void {
  setDebugCategories({}, { bufferStats: enabled });
  console.log(`Buffer stats ${enabled ? 'enabled (expensive!)' : 'disabled'}`);
}

// ============================================================================
// Browser Console Global API
// ============================================================================

/**
 * DOPPLER debug API exposed to browser console.
 */
export interface DopplerDebugAPI {
  // Pipeline debug
  debug: typeof debug;
  debugLayer: typeof debugLayer;
  debugBuffers: typeof debugBuffers;
  presets: typeof DEBUG_PRESETS;
  getConfig: typeof getDebugConfig;
  resetDebug: typeof resetDebugConfig;
  // Module logging
  log: typeof log;
  setLogLevel: typeof setLogLevel;
  setBenchmarkMode: typeof setBenchmarkMode;
  isBenchmarkMode: typeof isBenchmarkMode;
  enableModules: typeof enableModules;
  disableModules: typeof disableModules;
  // Tensor inspection
  tensor: typeof tensor;
  // Performance
  perf: typeof perf;
  // History
  getLogHistory: typeof getLogHistory;
  printLogSummary: typeof printLogSummary;
  getDebugSnapshot: typeof getDebugSnapshot;
}

const DOPPLER_API: DopplerDebugAPI = {
  debug,
  debugLayer,
  debugBuffers,
  presets: DEBUG_PRESETS,
  getConfig: getDebugConfig,
  resetDebug: resetDebugConfig,
  log,
  setLogLevel,
  setBenchmarkMode,
  isBenchmarkMode,
  enableModules,
  disableModules,
  tensor,
  perf,
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
}

// ============================================================================
// Default Export
// ============================================================================

export default {
  log,
  tensor,
  perf,
  debug,
  debugLayer,
  debugBuffers,
  setLogLevel,
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
  LOG_LEVELS,
  DEBUG_PRESETS,
};
