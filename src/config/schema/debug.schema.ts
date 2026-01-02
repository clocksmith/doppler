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
// Pipeline Debug Config (debug-utils)
// =============================================================================

/** Debug categories for pipeline debug-utils (kernel/layer inspection) */
export type PipelineDebugCategory =
  | 'embed'
  | 'layer'
  | 'attn'
  | 'ffn'
  | 'kv'
  | 'logits'
  | 'sample'
  | 'io'
  | 'perf'
  | 'kernel'
  | 'all';

/**
 * Pipeline debug configuration.
 *
 * Controls debug-utils categories and expensive readback helpers.
 */
export interface PipelineDebugConfigSchema {
  /** Enable pipeline debug (default: false) */
  enabled: boolean;
  /** Debug categories to enable (default: none) */
  categories: PipelineDebugCategory[];
  /** Filter to specific layer indices (null = all layers) */
  layers: number[] | null;
  /** Maximum decode steps to log (0 = unlimited) */
  maxDecodeSteps: number;
  /** Warn if maxAbs exceeds this */
  maxAbsThreshold: number;
  /** Enable expensive GPU buffer stats */
  bufferStats: boolean;
}

/** Default pipeline debug configuration */
export const DEFAULT_PIPELINE_DEBUG_CONFIG: PipelineDebugConfigSchema = {
  enabled: false,
  categories: [],
  layers: null,
  maxDecodeSteps: 0,
  maxAbsThreshold: 10000,
  bufferStats: false,
};

// =============================================================================
// Probe Config
// =============================================================================

/** Pipeline probe stages */
export type ProbeStage =
  | 'embed_out'
  | 'attn_out'
  | 'post_attn'
  | 'ffn_in'
  | 'ffn_out'
  | 'layer_out'
  | 'pre_final_norm'
  | 'final_norm'
  | 'logits'
  | 'logits_final';

/**
 * Probe configuration for targeted value inspection.
 *
 * Probes read specific token/dimension values from GPU buffers at
 * named pipeline stages.
 */
export interface ProbeConfigSchema {
  /** Optional probe id (included in logs) */
  id?: string;
  /** Stage to probe */
  stage: ProbeStage;
  /** Restrict to specific layers (null = all layers) */
  layers?: number[] | null;
  /** Token indices to sample (null = default to token 0) */
  tokens?: number[] | null;
  /** Dimension indices to sample */
  dims: number[];
  /** Override trace category (defaults to stage category) */
  category?: TraceCategory;
}

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
  pipeline: PipelineDebugConfigSchema;
  probes: ProbeConfigSchema[];
}

/** Default debug configuration */
export const DEFAULT_DEBUG_CONFIG: DebugConfigSchema = {
  logOutput: DEFAULT_LOG_OUTPUT_CONFIG,
  logHistory: DEFAULT_LOG_HISTORY_CONFIG,
  logLevel: DEFAULT_LOG_LEVEL_CONFIG,
  trace: DEFAULT_TRACE_CONFIG,
  pipeline: DEFAULT_PIPELINE_DEBUG_CONFIG,
  probes: [],
};
