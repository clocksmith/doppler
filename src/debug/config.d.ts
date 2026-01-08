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
export declare const LOG_LEVELS: {
  readonly DEBUG: 0;
  readonly VERBOSE: 1;
  readonly INFO: 2;
  readonly WARN: 3;
  readonly ERROR: 4;
  readonly SILENT: 5;
};

export type LogLevel = keyof typeof LOG_LEVELS;
export type LogLevelValue = (typeof LOG_LEVELS)[LogLevel];

/**
 * Trace categories
 */
export declare const TRACE_CATEGORIES: readonly [
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

export declare let currentLogLevel: LogLevelValue;
export declare let enabledModules: Set<string>;
export declare let disabledModules: Set<string>;
export declare let logHistory: LogEntry[];

export declare let gpuDevice: GPUDevice | null;

export declare let enabledTraceCategories: Set<TraceCategory>;
export declare let traceLayerFilter: number[];
export declare let traceDecodeStep: number;
export declare let traceMaxDecodeSteps: number;
export declare let traceBreakOnAnomaly: boolean;

// ============================================================================
// Configuration Functions
// ============================================================================

/**
 * Set the global log level.
 */
export declare function setLogLevel(level: string): void;

/**
 * Get current log level name.
 */
export declare function getLogLevel(): string;

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
export declare function setTrace(
  categories: string | TraceCategory[] | false,
  options?: { layers?: number[]; maxDecodeSteps?: number; breakOnAnomaly?: boolean }
): void;

/**
 * Apply debug config defaults unless URL params already set them.
 */
export declare function applyDebugConfig(
  config: DebugConfigSchema,
  options?: { respectUrlParams?: boolean }
): void;

/**
 * Get enabled trace categories.
 */
export declare function getTrace(): TraceCategory[];

/**
 * Check if a trace category is enabled.
 */
export declare function isTraceEnabled(category: TraceCategory, layerIdx?: number): boolean;

/**
 * Increment decode step counter (call after each decode step).
 */
export declare function incrementDecodeStep(): number;

/**
 * Reset decode step counter (call at start of generation).
 */
export declare function resetDecodeStep(): void;

/**
 * Get current decode step.
 */
export declare function getDecodeStep(): number;

/**
 * Check if we should break on anomaly.
 */
export declare function shouldBreakOnAnomaly(): boolean;

/**
 * Enable benchmark mode - silences all console.log/debug/info calls.
 */
export declare function setBenchmarkMode(enabled: boolean): void;

/**
 * Check if benchmark mode is active.
 */
export declare function isBenchmarkMode(): boolean;

/**
 * Enable logging for specific modules only.
 */
export declare function enableModules(...modules: string[]): void;

/**
 * Disable logging for specific modules.
 */
export declare function disableModules(...modules: string[]): void;

/**
 * Reset module filters.
 */
export declare function resetModuleFilters(): void;

/**
 * Set GPU device for tensor inspection.
 */
export declare function setGPUDevice(device: GPUDevice): void;

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
export declare function initFromUrlParams(): void;
