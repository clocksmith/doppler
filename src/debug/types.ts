/**
 * Debug Types and Constants
 *
 * @module debug/types
 */

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
 * Standard completion signal prefixes for CLI/automation detection.
 */
export const SIGNALS = {
  /** Task completed (success or error) - always emitted at end */
  DONE: '[DOPPLER:DONE]',
  /** Full result payload (JSON) - emitted before DONE for data extraction */
  RESULT: '[DOPPLER:RESULT]',
  /** Error occurred - can be emitted before DONE */
  ERROR: '[DOPPLER:ERROR]',
  /** Progress update (optional) */
  PROGRESS: '[DOPPLER:PROGRESS]',
} as const;

export type SignalType = keyof typeof SIGNALS;

/**
 * Completion payload for DONE signal.
 */
export interface DonePayload {
  status: 'success' | 'error';
  elapsed: number;
  tokens?: number;
  tokensPerSecond?: number;
  error?: string;
}

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
