/**
 * DOPPLER Debug Module - Configuration and State Management
 *
 * Manages log levels, trace categories, and module filters.
 *
 * @module debug/config
 */

import type { DebugConfigSchema } from '../config/schema/debug.schema.js';

// ============================================================================
// Types and Constants
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

export type LogLevel = keyof typeof LOG_LEVELS;
export type LogLevelValue = (typeof LOG_LEVELS)[LogLevel];

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

// ============================================================================
// Global State
// ============================================================================

export let currentLogLevel: LogLevelValue = LOG_LEVELS.INFO;
export let enabledModules = new Set<string>();
export let disabledModules = new Set<string>();
export let logHistory: LogEntry[] = [];

// GPU device reference for tensor inspection
export let gpuDevice: GPUDevice | null = null;

// Trace categories state
export let enabledTraceCategories = new Set<TraceCategory>();
export let traceLayerFilter: number[] = [];  // Empty = all layers
export let traceDecodeStep = 0;
export let traceMaxDecodeSteps = 0;  // 0 = unlimited
export let traceBreakOnAnomaly = false;

// Benchmark mode state
let benchmarkMode = false;
const originalConsoleLog = console.log;
const originalConsoleDebug = console.debug;
const originalConsoleInfo = console.info;

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
        ? config.trace.categories.join(',')
        : 'all';
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
