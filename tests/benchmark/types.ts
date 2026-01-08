/**
 * Benchmark Types
 *
 * TypeScript interfaces for the DOPPLER benchmark harness.
 * Follows the JSON schema defined in docs/spec/BENCHMARK_HARNESS.md
 *
 * @module tests/benchmark/types
 */

import type { KernelPathRef, RuntimeConfigSchema } from '../../src/config/schema/index.js';

// ============================================================================
// Result Schema (matches BENCHMARK_HARNESS.md)
// ============================================================================

/**
 * Complete benchmark result following the spec schema.
 */
export interface BenchmarkResult {
  /** Schema version for forwards compatibility */
  schemaVersion: 1;

  /** ISO 8601 timestamp */
  timestamp: string;

  /** Benchmark suite type */
  suite: 'kernel' | 'pipeline' | 'system';

  /** Run type: cold (from scratch) or warm (cached) */
  runType: 'cold' | 'warm';

  /** Environment metadata */
  env: EnvironmentInfo;

  /** Model configuration */
  model: ModelInfo;

  /** Workload configuration */
  workload: WorkloadInfo;

  /** Measured metrics */
  metrics: PipelineMetrics;

  /** Optional raw data for analysis */
  raw?: RawMetrics;
}

// ============================================================================
// Environment Info
// ============================================================================

export interface EnvironmentInfo {
  browser: BrowserInfo;
  os: OSInfo;
  gpu: GPUInfo;
  webgpu: WebGPUFeatures;
}

export interface BrowserInfo {
  name: string;
  version: string;
}

export interface OSInfo {
  name: string;
  version: string;
}

export interface GPUInfo {
  vendor: string;
  device: string;
  description: string;
}

export interface WebGPUFeatures {
  hasF16: boolean;
  hasSubgroups: boolean;
  hasTimestampQuery: boolean;
}

// ============================================================================
// Model Info
// ============================================================================

export interface ModelInfo {
  /** Model ID or hash */
  modelId: string;

  /** Human-readable name */
  modelName?: string;

  /** Quantization format */
  quantization: string;

  /** Total model size in bytes */
  totalSizeBytes: number;

  /** Number of tensors in the model */
  tensorCount: number;

  /** Number of layers */
  numLayers?: number;

  /** Hidden dimension size */
  hiddenSize?: number;
}

// ============================================================================
// Workload Info
// ============================================================================

export interface WorkloadInfo {
  /** Prompt category: xs, short, medium, long */
  promptName: 'xs' | 'short' | 'medium' | 'long' | 'custom';

  /** Number of prompt tokens */
  promptTokens: number;

  /** Maximum tokens to generate */
  maxNewTokens: number;

  /** Sampling configuration */
  sampling: SamplingConfig;
}

export interface SamplingConfig {
  temperature: number;
  topK: number;
  topP: number;
}

// ============================================================================
// Pipeline Metrics (Required)
// ============================================================================

export interface PipelineMetrics {
  // Latency and throughput
  /** Time to first token (ms) */
  ttft_ms: number;

  /** Prefill wall time (ms) */
  prefill_ms: number;

  /** Prefill throughput (tokens/sec) */
  prefill_tokens_per_sec: number;

  /** Total decode wall time (ms) */
  decode_ms_total: number;

  /** Decode throughput (tokens/sec) */
  decode_tokens_per_sec: number;

  // Decode latency distribution
  /** P50 decode latency per token (ms) */
  decode_ms_per_token_p50?: number;

  /** P90 decode latency per token (ms) */
  decode_ms_per_token_p90?: number;

  /** P99 decode latency per token (ms) */
  decode_ms_per_token_p99?: number;

  // GPU scheduling
  /** GPU submit count during prefill */
  gpu_submit_count_prefill: number;

  /** GPU submit count during decode */
  gpu_submit_count_decode: number;

  /** Total GPU submit count (perf guard counter) */
  gpu_submit_count_total?: number;

  /** GPU buffer allocation count (perf guard counter) */
  gpu_allocation_count_total?: number;

  /** GPU readback count (perf guard counter) */
  gpu_readback_count_total?: number;

  /** Total bytes read back from GPU */
  gpu_readback_bytes_total?: number;

  /** Whether timestamp queries are available */
  gpu_timestamp_available?: boolean;

  /** GPU time during prefill (if timestamp available) */
  gpu_time_ms_prefill?: number;

  /** GPU time during decode (if timestamp available) */
  gpu_time_ms_decode?: number;

  // Memory
  /** Peak VRAM allocation estimate (bytes) */
  estimated_vram_bytes_peak?: number;

  /** KV cache data type */
  kv_cache_dtype?: 'f16' | 'f32';

  /** KV cache maximum sequence length */
  kv_cache_max_seq_len?: number;

  // Storage (cold start)
  /** Storage mode used */
  storage_mode?: 'opfs' | 'native_bridge' | 'http_only';

  /** Whether storage is persisted */
  storage_persisted?: boolean;

  /** OPFS usage in bytes */
  opfs_usage_bytes?: number;

  // Distribution (cold start)
  /** Bytes downloaded from origin */
  origin_bytes_downloaded?: number;

  /** Bytes written to OPFS */
  opfs_bytes_written?: number;

  /** Wall time for download (ms) */
  download_wall_ms?: number;
}

// ============================================================================
// Raw Metrics (Optional)
// ============================================================================

export interface RawMetrics {
  /** Per-token decode latencies (ms) */
  decode_latencies_ms?: number[];

  /** Per-submit GPU times (ms) if available */
  submit_times_ms?: number[];

  /** Token IDs generated */
  generated_token_ids?: number[];

  /** Generated text */
  generated_text?: string;
}

// ============================================================================
// Benchmark Configuration
// ============================================================================

export interface BenchmarkConfig {
  /** Model manifest URL or path */
  modelPath: string;

  /** Which prompt workload to use */
  promptName: 'xs' | 'short' | 'medium' | 'long' | 'custom';

  /** Custom prompt text (if promptName is 'custom') */
  customPrompt?: string;

  /** Maximum tokens to generate */
  maxNewTokens: number;

  /** Run type: cold or warm */
  runType: 'cold' | 'warm';

  /** Number of warmup iterations */
  warmupRuns: number;

  /** Number of timed iterations */
  timedRuns: number;

  /** Sampling options */
  sampling: SamplingConfig;

  /** Output path for JSON results */
  outputPath?: string;

  /** Enable debug logging */
  debug?: boolean;

  /** Specific layers to checkpoint during debug (e.g., [0, 12, 25]) */
  debugLayers?: number[];

  /** Apply model-specific chat template to prompt before tokenization/inference. */
  useChatTemplate?: boolean;

  /** Enable GPU timestamp profiling for per-kernel timing analysis.
   *  Requires 'timestamp-query' WebGPU feature. */
  profile?: boolean;

  /** Runtime overrides (kernel path). */
  runtime?: {
    kernelPath?: KernelPathRef;
  };
  /** Full runtime config overrides */
  runtimeConfig?: Partial<RuntimeConfigSchema>;
}

// ============================================================================
// Benchmark Session
// ============================================================================

export interface BenchmarkSession {
  /** Session ID */
  sessionId: string;

  /** Start timestamp */
  startTime: string;

  /** All results from this session */
  results: BenchmarkResult[];

  /** Aggregate summary */
  summary?: BenchmarkSummary;
}

export interface BenchmarkSummary {
  /** Total benchmarks run */
  totalRuns: number;

  /** Successful runs */
  successfulRuns: number;

  /** Failed runs */
  failedRuns: number;

  /** Average TTFT across runs */
  avgTtftMs: number;

  /** Average decode throughput */
  avgDecodeTokensPerSec: number;
}

// ============================================================================
// Comparison Types
// ============================================================================

export interface ComparisonResult {
  /** DOPPLER result */
  doppler: BenchmarkResult;

  /** Competitor result (WebLLM, etc.) */
  competitor: BenchmarkResult;

  /** Comparison metadata */
  comparison: {
    runtime: string;
    version: string;
    notes: string;
  };

  /** Computed deltas */
  deltas: {
    ttft_ms_diff: number;
    decode_tokens_per_sec_diff: number;
    prefill_tokens_per_sec_diff: number;
  };
}

// ============================================================================
// Helper Types
// ============================================================================

export type PromptCategory = 'xs' | 'short' | 'medium' | 'long';

export interface Prompt {
  name: PromptCategory;
  text: string;
  expectedTokenCount: { min: number; max: number };
}

/**
 * Default benchmark configuration
 */
export const DEFAULT_BENCHMARK_CONFIG: Partial<BenchmarkConfig> = {
  promptName: 'medium',
  maxNewTokens: 128,
  runType: 'warm',
  warmupRuns: 2,
  timedRuns: 3,
  sampling: {
    temperature: 0,
    topK: 1,
    topP: 1,
  },
  debug: false,
  useChatTemplate: undefined,  // Auto-detect based on model name
};
