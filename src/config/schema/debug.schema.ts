/**
 * Debug Config Schema
 *
 * Configuration for the DOPPLER debug module, including log history limits,
 * default log levels, trace categories, and decode step limits.
 *
 * @module config/schema/debug
 */

// =============================================================================
// Log Output Config
// =============================================================================

/**
 * Configuration for log output destinations.
 *
 * Controls where logs are written: stdout, file, or both.
 */
export interface LogOutputConfigSchema {
  /** Write logs to stdout/console (default: true) */
  stdout: boolean;
  /** Path to log file (null = no file output) */
  file: string | null;
  /** Append to existing file vs overwrite (default: true) */
  append: boolean;
}

/** Default log output configuration */
export const DEFAULT_LOG_OUTPUT_CONFIG: LogOutputConfigSchema = {
  stdout: true,
  file: null,
  append: true,
};

// =============================================================================
// Log History Config
// =============================================================================

/**
 * Configuration for log history retention.
 *
 * Controls how many log entries are kept in memory for debugging
 * and diagnostic purposes.
 */
export interface LogHistoryConfigSchema {
  /** Maximum number of log entries to retain in memory */
  maxLogHistoryEntries: number;
}

/** Default log history configuration */
export const DEFAULT_LOG_HISTORY_CONFIG: LogHistoryConfigSchema = {
  maxLogHistoryEntries: 1000,
};

// =============================================================================
// Log Level Config
// =============================================================================

/**
 * Configuration for default log level.
 *
 * Controls the initial verbosity level when the debug module initializes.
 */
export interface LogLevelConfigSchema {
  /** Default log level (debug, verbose, info, warn, error, silent) */
  defaultLogLevel: string;
}

/** Default log level configuration */
export const DEFAULT_LOG_LEVEL_CONFIG: LogLevelConfigSchema = {
  defaultLogLevel: 'info',
};

// =============================================================================
// Trace Config
// =============================================================================

/** Available trace categories */
export type TraceCategory =
  | 'loader'
  | 'kernels'
  | 'logits'
  | 'embed'
  | 'attn'
  | 'ffn'
  | 'kv'
  | 'sample'
  | 'buffers'
  | 'perf'
  | 'all';

/**
 * Configuration for trace output.
 *
 * Controls trace categories, output destination, and limits.
 */
export interface TraceConfigSchema {
  /** Enable tracing (default: false) */
  enabled: boolean;
  /** Trace categories to enable (default: all) */
  categories: TraceCategory[];
  /** Filter to specific layer indices (null = all layers) */
  layers: number[] | null;
  /** Maximum decode steps to trace (0 = unlimited) */
  maxDecodeSteps: number;
  /** Path to trace file for JSONL output (null = no file) */
  file: string | null;
}

/** Default trace configuration */
export const DEFAULT_TRACE_CONFIG: TraceConfigSchema = {
  enabled: false,
  categories: ['all'],
  layers: null,
  maxDecodeSteps: 0,
  file: null,
};

// =============================================================================
// Complete Debug Config
// =============================================================================

/**
 * Complete debug configuration schema.
 *
 * Combines log output, log history, log level, and trace settings.
 */
export interface DebugConfigSchema {
  logOutput: LogOutputConfigSchema;
  logHistory: LogHistoryConfigSchema;
  logLevel: LogLevelConfigSchema;
  trace: TraceConfigSchema;
}

/** Default debug configuration */
export const DEFAULT_DEBUG_CONFIG: DebugConfigSchema = {
  logOutput: DEFAULT_LOG_OUTPUT_CONFIG,
  logHistory: DEFAULT_LOG_HISTORY_CONFIG,
  logLevel: DEFAULT_LOG_LEVEL_CONFIG,
  trace: DEFAULT_TRACE_CONFIG,
};
