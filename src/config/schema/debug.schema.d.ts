/**
 * Debug Config Schema
 *
 * Configuration for the DOPPLER debug module, including log history limits,
 * default log levels, trace categories, and decode step limits.
 *
 * @module config/schema/debug
 */

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
export declare const DEFAULT_LOG_OUTPUT_CONFIG: LogOutputConfigSchema;

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
export declare const DEFAULT_LOG_HISTORY_CONFIG: LogHistoryConfigSchema;

/** Valid log levels */
export declare const LOG_LEVELS: readonly ['debug', 'verbose', 'info', 'warn', 'error', 'silent'];

/** Log level type */
export type LogLevel = typeof LOG_LEVELS[number];

/**
 * Configuration for default log level.
 *
 * Controls the initial verbosity level when the debug module initializes.
 */
export interface LogLevelConfigSchema {
  /** Default log level (debug, verbose, info, warn, error, silent) */
  defaultLogLevel: LogLevel;
}

/** Default log level configuration */
export declare const DEFAULT_LOG_LEVEL_CONFIG: LogLevelConfigSchema;

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
export declare const DEFAULT_TRACE_CONFIG: TraceConfigSchema;

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
  /** Maximum bytes to readback for debug samples (default: 512) */
  readbackSampleSize: number;
}

/** Default pipeline debug configuration */
export declare const DEFAULT_PIPELINE_DEBUG_CONFIG: PipelineDebugConfigSchema;

/**
 * Profiler configuration.
 */
export interface ProfilerConfigSchema {
  /** Enable GPU profiling */
  enabled: boolean;
  /** Maximum number of timestamp pairs to allocate */
  queryCapacity: number;
  /** Absolute upper bound for query capacity */
  maxQueries: number;
  /** Fallback query capacity when device limit is unavailable */
  defaultQueryLimit: number;
  /** Maximum samples retained per label */
  maxSamples: number;
  /** GPU timing sanity limit before falling back to CPU */
  maxDurationMs: number;
  /** Log every N decode profile steps (<=1 logs all steps) */
  logEveryDecodeSteps: number;
}

/** Default profiler configuration */
export declare const DEFAULT_PROFILER_CONFIG: ProfilerConfigSchema;

/**
 * Performance guard configuration.
 *
 * Controls GPU readback and tracking behavior.
 */
export interface PerfGuardsConfigSchema {
  /** Allow GPU readbacks (default: true) */
  allowGPUReadback: boolean;
  /** Track queue.submit() calls (default: false) */
  trackSubmitCount: boolean;
  /** Track buffer allocations (default: false) */
  trackAllocations: boolean;
  /** Log expensive operations (default: false) */
  logExpensiveOps: boolean;
  /** Throw on blocked operations (default: false) */
  strictMode: boolean;
}

/** Default performance guard configuration */
export declare const DEFAULT_PERF_GUARDS_CONFIG: PerfGuardsConfigSchema;

/** Pipeline probe stages */
export type ProbeStage =
  | 'embed_out'
  // Attention stages (per-layer)
  | 'attn_input'      // Input to attention (after residual from previous layer)
  | 'attn_normed'     // After input RMSNorm
  | 'q_proj'          // Q projection output
  | 'k_proj'          // K projection output
  | 'v_proj'          // V projection output
  | 'q_rope'          // Q after RoPE
  | 'k_rope'          // K after RoPE
  | 'attn_scores'     // Attention scores (pre-softmax)
  | 'attn_out'        // Attention output (before o_proj)
  | 'o_proj'          // Output projection
  | 'post_attn'       // After attention residual
  // FFN stages (per-layer)
  | 'ffn_normed'      // After post-attention RMSNorm
  | 'ffn_in'          // FFN gate/up input
  | 'ffn_gate'        // Gate projection output
  | 'ffn_up'          // Up projection output
  | 'ffn_act'         // After activation
  | 'ffn_out'         // Down projection output
  | 'layer_out'       // Final layer output (after FFN residual)
  // Final stages
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
  profiler: ProfilerConfigSchema;
  perfGuards: PerfGuardsConfigSchema;
}

/** Default debug configuration */
export declare const DEFAULT_DEBUG_CONFIG: DebugConfigSchema;
