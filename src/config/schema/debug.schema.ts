/**
 * Debug Config Schema
 *
 * Configuration for the DOPPLER debug module, including log history limits,
 * default log levels, trace categories, and decode step limits.
 *
 * @module config/schema/debug
 */

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

/**
 * Configuration for trace decoding limits.
 *
 * Controls maximum decode steps to trace before automatically
 * disabling trace output to prevent log flooding.
 */
export interface TraceConfigSchema {
  /** Maximum decode steps to trace (0 = unlimited) */
  maxDecodeSteps: number;
}

/** Default trace configuration */
export const DEFAULT_TRACE_CONFIG: TraceConfigSchema = {
  maxDecodeSteps: 0,
};

// =============================================================================
// Complete Debug Config
// =============================================================================

/**
 * Complete debug configuration schema.
 *
 * Combines log history, log level, and trace settings.
 */
export interface DebugConfigSchema {
  logHistory: LogHistoryConfigSchema;
  logLevel: LogLevelConfigSchema;
  trace: TraceConfigSchema;
}

/** Default debug configuration */
export const DEFAULT_DEBUG_CONFIG: DebugConfigSchema = {
  logHistory: DEFAULT_LOG_HISTORY_CONFIG,
  logLevel: DEFAULT_LOG_LEVEL_CONFIG,
  trace: DEFAULT_TRACE_CONFIG,
};
