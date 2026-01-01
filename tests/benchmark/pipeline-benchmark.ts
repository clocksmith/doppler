/**
 * Pipeline Benchmark Harness
 *
 * Measures end-to-end inference performance following the spec in
 * docs/spec/BENCHMARK_HARNESS.md
 *
 * Usage:
 *   const harness = new PipelineBenchmark(config);
 *   const result = await harness.run();
 *   console.log(JSON.stringify(result, null, 2));
 *
 * @module tests/benchmark/pipeline-benchmark
 */

import type {
  BenchmarkResult,
  BenchmarkConfig,
  PipelineMetrics,
  EnvironmentInfo,
  ModelInfo,
  WorkloadInfo,
  RawMetrics,
  SamplingConfig,
  PromptCategory,
} from './types.js';
import { DEFAULT_BENCHMARK_CONFIG } from './types.js';
import { getPrompt } from './prompts.js';
import { applyGemmaChatTemplate, applyLlama3ChatTemplate } from '../../inference/pipeline/init.js';

// Track GPU readback bytes globally during benchmark
let readbackBytesTotal = 0;

function resetReadbackTracking() {
  readbackBytesTotal = 0;
}

function trackReadback(bytes: number) {
  readbackBytesTotal += bytes;
}

function getReadbackBytes(): number {
  return readbackBytesTotal;
}

// ============================================================================
// Pipeline Benchmark Class
// ============================================================================

export class PipelineBenchmark {
  private config: BenchmarkConfig;
  private pipeline: any = null;
  private manifest: any = null;
  private profiler: any = null;
  private hasTimestampQuery = false;

  constructor(config: Partial<BenchmarkConfig> & { modelPath: string }) {
    this.config = {
      ...DEFAULT_BENCHMARK_CONFIG,
      ...config,
    } as BenchmarkConfig;
  }

  /**
   * Run the benchmark and return results in spec-compliant JSON format.
   */
  async run(): Promise<BenchmarkResult> {
    const startTime = performance.now();

    // Collect environment info
    const env = await this.collectEnvironment();

    // Load model (measures cold start if applicable)
    const loadMetrics = await this.loadModel();

    // Get prompt
    const prompt = this.getPromptText();

    // Warmup runs (not measured)
    for (let i = 0; i < this.config.warmupRuns; i++) {
      await this.runInference(prompt, true);
      this.pipeline.reset();
    }

    // Timed runs
    const runResults: TimedRunResult[] = [];
    for (let i = 0; i < this.config.timedRuns; i++) {
      const result = await this.runInference(prompt, false);
      runResults.push(result);
      this.pipeline.reset();
    }

    // Aggregate metrics
    const metrics = this.aggregateMetrics(runResults, loadMetrics);
    const raw = this.collectRawMetrics(runResults);

    // Build result
    const result: BenchmarkResult = {
      schemaVersion: 1,
      timestamp: new Date().toISOString(),
      suite: 'pipeline',
      runType: this.config.runType,
      env,
      model: this.getModelInfo(),
      workload: this.getWorkloadInfo(runResults[0]?.promptTokens ?? 0),
      metrics,
      raw,
    };

    if (this.config.debug) {
      console.log(`[Benchmark] Total time: ${(performance.now() - startTime).toFixed(0)}ms`);
    }

    return result;
  }

  // ==========================================================================
  // Environment Collection
  // ==========================================================================

  private async collectEnvironment(): Promise<EnvironmentInfo> {
    const ua = typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown';
    const browser = this.parseBrowserInfo(ua);
    const os = this.parseOSInfo(ua);

    // GPU info from WebGPU adapter
    let gpu = { vendor: 'unknown', device: 'unknown', description: 'unknown' };
    let webgpu = { hasF16: false, hasSubgroups: false, hasTimestampQuery: false };

    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        const adapter = await (navigator as any).gpu.requestAdapter();
        if (adapter) {
          const info = await adapter.requestAdapterInfo?.() ?? {};
          gpu = {
            vendor: info.vendor ?? 'unknown',
            device: info.device ?? 'unknown',
            description: info.description ?? adapter.name ?? 'unknown',
          };

          webgpu = {
            hasF16: adapter.features.has('shader-f16'),
            hasSubgroups: adapter.features.has('subgroups'),
            hasTimestampQuery: adapter.features.has('timestamp-query'),
          };
        }
      } catch (e) {
        console.warn('[Benchmark] Failed to get GPU info:', e);
      }
    }

    return { browser, os, gpu, webgpu };
  }

  private parseBrowserInfo(ua: string): { name: string; version: string } {
    if (ua.includes('Chrome')) {
      const match = ua.match(/Chrome\/(\d+\.\d+\.\d+\.\d+)/);
      return { name: 'Chrome', version: match?.[1] ?? 'unknown' };
    }
    if (ua.includes('Safari') && !ua.includes('Chrome')) {
      const match = ua.match(/Version\/(\d+\.\d+)/);
      return { name: 'Safari', version: match?.[1] ?? 'unknown' };
    }
    if (ua.includes('Firefox')) {
      const match = ua.match(/Firefox\/(\d+\.\d+)/);
      return { name: 'Firefox', version: match?.[1] ?? 'unknown' };
    }
    return { name: 'unknown', version: 'unknown' };
  }

  private parseOSInfo(ua: string): { name: string; version: string } {
    if (ua.includes('Mac OS X')) {
      const match = ua.match(/Mac OS X (\d+[._]\d+[._]?\d*)/);
      return { name: 'macOS', version: match?.[1]?.replace(/_/g, '.') ?? 'unknown' };
    }
    if (ua.includes('Windows')) {
      const match = ua.match(/Windows NT (\d+\.\d+)/);
      return { name: 'Windows', version: match?.[1] ?? 'unknown' };
    }
    if (ua.includes('Linux')) {
      return { name: 'Linux', version: 'unknown' };
    }
    return { name: 'unknown', version: 'unknown' };
  }

  // ==========================================================================
  // Model Loading
  // ==========================================================================

  private async loadModel(): Promise<LoadMetrics> {
    const { createPipeline } = await import('../../inference/pipeline.js');
    const { initDevice, hasFeature, FEATURES } = await import('../../gpu/device.js');
    const { createProfiler } = await import('../../gpu/profiler.js');
    const { getBufferPool } = await import('../../gpu/buffer-pool.js');
    const { downloadModel } = await import('../../storage/downloader.js');
    const { modelExists, initOPFS } = await import('../../storage/shard-manager.js');

    const loadStart = performance.now();

    // Initialize GPU
    const device = await initDevice();
    if (!device) {
      throw new Error('WebGPU not available');
    }

    // Initialize profiler for GPU timestamp queries
    this.hasTimestampQuery = hasFeature(FEATURES.TIMESTAMP_QUERY);
    if (this.hasTimestampQuery) {
      this.profiler = createProfiler(device);
    }

    // Reset buffer pool stats for accurate peak tracking
    const bufferPool = getBufferPool();
    // Note: We can't reset peakBytesAllocated directly, but stats are cumulative

    // Fetch manifest from HTTP
    const manifestUrl = this.config.modelPath.endsWith('.json')
      ? this.config.modelPath
      : `${this.config.modelPath}/manifest.json`;

    const manifestResponse = await fetch(manifestUrl);
    this.manifest = await manifestResponse.json();
    const modelId = this.manifest.modelId || this.manifest.model_id;

    // Check if model is in OPFS, download if not.
    // NOTE: shard-manager APIs require selecting a model directory before reading files.
    // modelExists() checks for a manifest without relying on global currentModelDir.
    await initOPFS();
    const isCached = await modelExists(modelId);
    if (!isCached) {
      console.log('[Benchmark] Model not in OPFS, downloading...');
      const baseUrl = this.config.modelPath.replace(/\/manifest\.json$/, '');
      await downloadModel(baseUrl, (progress) => {
        if (progress.percent % 10 === 0) {
          console.log(`[Benchmark] Download: ${progress.percent}%`);
        }
      }, { modelId });
      console.log('[Benchmark] Download complete');
    }

    // Create pipeline
    const runtime: Record<string, unknown> = { debug: this.config.debug };
    if (this.config.runtime?.attentionKernel) {
      runtime.attentionKernel = this.config.runtime.attentionKernel;
    }
    if (this.config.runtime?.kernelHints) {
      runtime.kernelHints = this.config.runtime.kernelHints;
    }

    this.pipeline = await createPipeline(this.manifest, {
      gpu: { device },
      baseUrl: this.config.modelPath.replace(/\/manifest\.json$/, ''),
      runtime,
    });

    const loadEnd = performance.now();

    // Get storage metrics if available
    let opfsUsageBytes: number | undefined;
    let storagePersisted: boolean | undefined;
    try {
      if (typeof navigator !== 'undefined' && navigator.storage) {
        storagePersisted = await navigator.storage.persisted?.();
        const estimate = await navigator.storage.estimate?.();
        opfsUsageBytes = estimate?.usage;
      }
    } catch (e) {
      // Storage API not available
    }

    return {
      loadTimeMs: loadEnd - loadStart,
      opfsUsageBytes,
      storagePersisted,
    };
  }

  // ==========================================================================
  // Inference
  // ==========================================================================

  private async runInference(prompt: string, isWarmup: boolean): Promise<TimedRunResult> {
    const { setTrackSubmits, resetSubmitStats, getSubmitStats, setSubmitPhase, getPhaseSubmitStats, logAllPhaseSubmitStats } = await import('../../gpu/submit-tracker.js');
    const { getBufferPool } = await import('../../gpu/buffer-pool.js');
    const { enableBenchmarkMode, resetPerfCounters, getPerfCounters } = await import('../../gpu/perf-guards.js');

    // Enable submit tracking
    setTrackSubmits(true);
    resetSubmitStats();
    resetReadbackTracking();
    enableBenchmarkMode();
    resetPerfCounters();

    // Reset profiler for GPU timing
    if (this.profiler) {
      this.profiler.reset();
    }

    const tokens: number[] = [];
    const decodeLatencies: number[] = [];
    let text = '';

    const inferenceStart = performance.now();
    let ttft = 0;
    let prefillEnd = 0;
    let tokenCount = 0;
    let lastTokenTime = 0;

    // Start GPU timing for prefill and set phase
    setSubmitPhase('prefill');
    if (this.profiler) {
      this.profiler.begin('prefill');
    }

    // Run generation
    // Auto-detect instruction-tuned models for chat template
    const modelId = this.manifest?.modelId ?? '';
    const isInstructModel = /instruct|chat|it(?:-|$)/i.test(modelId);
    const useChatTemplate = this.config.useChatTemplate ?? isInstructModel;
    const generator = this.pipeline.generate(prompt, {
      maxTokens: this.config.maxNewTokens,
      temperature: this.config.sampling.temperature,
      topK: this.config.sampling.topK,
      topP: this.config.sampling.topP,
      debug: this.config.debug ?? true,
      // Use debugLayers from config for selective layer checkpointing
      // Without debugLayers, debug mode syncs at EVERY layer (very slow)
      debugLayers: this.config.debugLayers,
      // GPU timestamp profiling for per-kernel timing
      profile: this.config.profile ?? false,
      useChatTemplate,
      onToken: (id: number, t: string) => {
        const now = performance.now();
        tokens.push(id);
        text += t;
        tokenCount++;

        if (tokenCount === 1) {
          ttft = now - inferenceStart;
          prefillEnd = now;
          // End prefill timing, start decode timing
          if (this.profiler) {
            this.profiler.end('prefill');
            this.profiler.begin('decode');
          }
          // Switch to decode phase for submit tracking
          setSubmitPhase('decode');
          // Track logits readback (~vocab_size * 4 bytes for f32 logits)
          const vocabSize = this.manifest?.config?.vocab_size ?? 32000;
          trackReadback(vocabSize * 4);
        } else {
          decodeLatencies.push(now - lastTokenTime);
          // Each decode step reads back logits (or just argmax result if GPU sampling)
          // Estimate: 4 bytes for token ID if GPU sampling, full logits otherwise
          trackReadback(4); // Conservative: assume GPU sampling
        }
        lastTokenTime = now;
      },
    });

    // Consume generator
    for await (const chunk of generator) {
      // Token callback handles timing
    }

    // End decode timing
    if (this.profiler) {
      this.profiler.end('decode');
    }
    setSubmitPhase('other');

    const inferenceEnd = performance.now();

    // Resolve GPU timestamps
    let gpuTimePrefillMs: number | undefined;
    let gpuTimeDecodeMs: number | undefined;
    if (this.profiler) {
      await this.profiler.resolve();
      const prefillResult = this.profiler.getResult('prefill');
      const decodeResult = this.profiler.getResult('decode');
      gpuTimePrefillMs = prefillResult?.avg;
      gpuTimeDecodeMs = decodeResult?.avg;
    }

    // Get submit stats by phase (accurate, not estimated)
    const prefillStats = getPhaseSubmitStats('prefill');
    const decodeStats = getPhaseSubmitStats('decode');
    const globalStats = getSubmitStats();

    // Log submit stats with sources if not warmup
    if (!isWarmup && this.config.verbose) {
      logAllPhaseSubmitStats(`Submit stats for ${prompt.slice(0, 20)}...`);
      console.warn(`[Benchmark] Total submits: ${globalStats.count}, Decode submits: ${decodeStats.count}, Tokens: ${tokenCount}, Submits/token: ${(decodeStats.count / Math.max(tokenCount - 1, 1)).toFixed(2)}`);
    }

    setTrackSubmits(false);

    // Get pipeline stats
    const pipelineStats = this.pipeline.getStats();
    if (pipelineStats.gpuTimePrefillMs !== undefined) {
      gpuTimePrefillMs = pipelineStats.gpuTimePrefillMs;
    }
    if (pipelineStats.gpuTimeDecodeMs !== undefined) {
      gpuTimeDecodeMs = pipelineStats.gpuTimeDecodeMs;
    }

    // Get buffer pool stats for peak VRAM
    const bufferStats = getBufferPool().getStats();

    let promptForTokenCount = prompt;
    if (useChatTemplate) {
      if (this.manifest?.architecture?.includes('Gemma3') || /gemma/i.test(modelId)) {
        promptForTokenCount = applyGemmaChatTemplate(prompt);
      } else if (/llama.*3|llama3/i.test(modelId)) {
        promptForTokenCount = applyLlama3ChatTemplate(prompt);
      }
    }

    return {
      ttftMs: ttft,
      prefillMs: pipelineStats.prefillTimeMs,
      decodeMsTotal: pipelineStats.decodeTimeMs,
      decodeLatencies,
      tokens,
      text,
      promptTokens: this.pipeline.tokenizer?.encode(promptForTokenCount)?.length ?? 0,
      generatedTokens: tokens.length,
      gpuSubmitCountPrefill: prefillStats.count,
      gpuSubmitCountDecode: decodeStats.count,
      submitTimesMs: globalStats.timestamps,
      gpuTimePrefillMs,
      gpuTimeDecodeMs,
      gpuReadbackBytes: getReadbackBytes(),
      peakVramBytes: bufferStats.peakBytesAllocated,
      perfSubmits: getPerfCounters().submits,
      perfAllocations: getPerfCounters().allocations,
      perfReadbacks: getPerfCounters().readbacks,
    };
  }

  // ==========================================================================
  // Metrics Aggregation
  // ==========================================================================

  private aggregateMetrics(runs: TimedRunResult[], loadMetrics: LoadMetrics): PipelineMetrics {
    // Average across runs
    const avgTtft = this.average(runs.map(r => r.ttftMs));
    const avgPrefill = this.average(runs.map(r => r.prefillMs));
    const avgDecodeTotal = this.average(runs.map(r => r.decodeMsTotal));
    const avgPromptTokens = this.average(runs.map(r => r.promptTokens));
    const avgGeneratedTokens = this.average(runs.map(r => r.generatedTokens));

    // Decode latency distribution (flatten all runs)
    const allDecodeLatencies = runs.flatMap(r => r.decodeLatencies);
    const sortedLatencies = [...allDecodeLatencies].sort((a, b) => a - b);

    const metrics: PipelineMetrics = {
      ttft_ms: Math.round(avgTtft),
      prefill_ms: Math.round(avgPrefill),
      prefill_tokens_per_sec: avgPrefill > 0 ? Math.round(avgPromptTokens / (avgPrefill / 1000)) : 0,
      decode_ms_total: Math.round(avgDecodeTotal),
      decode_tokens_per_sec: avgDecodeTotal > 0 ? Math.round((avgGeneratedTokens - 1) / (avgDecodeTotal / 1000)) : 0,
      gpu_submit_count_prefill: Math.round(this.average(runs.map(r => r.gpuSubmitCountPrefill))),
      gpu_submit_count_decode: Math.round(this.average(runs.map(r => r.gpuSubmitCountDecode))),
      gpu_submit_count_total: Math.round(this.average(runs.map(r => r.perfSubmits))),
      gpu_allocation_count_total: Math.round(this.average(runs.map(r => r.perfAllocations))),
      gpu_readback_count_total: Math.round(this.average(runs.map(r => r.perfReadbacks))),
    };

    // Percentiles
    if (sortedLatencies.length > 0) {
      metrics.decode_ms_per_token_p50 = this.percentile(sortedLatencies, 50);
      metrics.decode_ms_per_token_p90 = this.percentile(sortedLatencies, 90);
      metrics.decode_ms_per_token_p99 = this.percentile(sortedLatencies, 99);
    }

    // GPU readback bytes (sum across all runs)
    const totalReadbackBytes = runs.reduce((sum, r) => sum + r.gpuReadbackBytes, 0);
    metrics.gpu_readback_bytes_total = totalReadbackBytes;

    // GPU timestamp timing (if available)
    const gpuPrefillTimes = runs.map(r => r.gpuTimePrefillMs).filter((t): t is number => t !== undefined);
    const gpuDecodeTimes = runs.map(r => r.gpuTimeDecodeMs).filter((t): t is number => t !== undefined);
    if (gpuPrefillTimes.length > 0) {
      metrics.gpu_timestamp_available = true;
      metrics.gpu_time_ms_prefill = Math.round(this.average(gpuPrefillTimes));
    }
    if (gpuDecodeTimes.length > 0) {
      metrics.gpu_time_ms_decode = Math.round(this.average(gpuDecodeTimes));
    }
    if (gpuPrefillTimes.length === 0 && gpuDecodeTimes.length === 0) {
      metrics.gpu_timestamp_available = false;
    }

    // Peak VRAM (max across all runs)
    const peakVram = Math.max(...runs.map(r => r.peakVramBytes));
    metrics.estimated_vram_bytes_peak = peakVram;

    // Storage metrics
    if (loadMetrics.opfsUsageBytes !== undefined) {
      metrics.opfs_usage_bytes = loadMetrics.opfsUsageBytes;
    }
    if (loadMetrics.storagePersisted !== undefined) {
      metrics.storage_persisted = loadMetrics.storagePersisted;
    }

    // Cold start metrics
    if (this.config.runType === 'cold') {
      metrics.download_wall_ms = loadMetrics.loadTimeMs;
    }

    return metrics;
  }

  private collectRawMetrics(runs: TimedRunResult[]): RawMetrics {
    // Use last run for raw data
    const lastRun = runs[runs.length - 1];
    return {
      decode_latencies_ms: lastRun?.decodeLatencies,
      submit_times_ms: lastRun?.submitTimesMs,
      generated_token_ids: lastRun?.tokens,
      generated_text: lastRun?.text,
    };
  }

  // ==========================================================================
  // Info Helpers
  // ==========================================================================

  private getModelInfo(): ModelInfo {
    const m = this.manifest;
    return {
      modelId: m?.id ?? m?.modelId ?? 'unknown',
      modelName: m?.name ?? m?.modelName ?? undefined,
      quantization: m?.quantization ?? 'unknown',
      totalSizeBytes: m?.totalSize ?? m?.totalSizeBytes ?? 0,
      tensorCount: m?.tensors?.length ?? m?.tensorCount ?? 0,
      numLayers: m?.config?.num_hidden_layers ?? m?.numLayers ?? undefined,
      hiddenSize: m?.config?.hidden_size ?? m?.hiddenSize ?? undefined,
    };
  }

  private getWorkloadInfo(promptTokens: number): WorkloadInfo {
    return {
      promptName: this.config.promptName,
      promptTokens,
      maxNewTokens: this.config.maxNewTokens,
      sampling: this.config.sampling,
    };
  }

  private getPromptText(): string {
    if (this.config.promptName === 'custom' && this.config.customPrompt) {
      return this.config.customPrompt;
    }
    return getPrompt(this.config.promptName as PromptCategory).text;
  }

  // ==========================================================================
  // Math Helpers
  // ==========================================================================

  private average(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  private percentile(sorted: number[], p: number): number {
    if (sorted.length === 0) return 0;
    if (sorted.length === 1) return sorted[0];
    // Linear interpolation for accurate percentiles
    const idx = (p / 100) * (sorted.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  }
}

// ============================================================================
// Internal Types
// ============================================================================

interface LoadMetrics {
  loadTimeMs: number;
  downloadBytes?: number;
  opfsWriteBytes?: number;
  opfsUsageBytes?: number;
  storagePersisted?: boolean;
}

interface TimedRunResult {
  ttftMs: number;
  prefillMs: number;
  decodeMsTotal: number;
  decodeLatencies: number[];
  tokens: number[];
  text: string;
  promptTokens: number;
  generatedTokens: number;
  gpuSubmitCountPrefill: number;
  gpuSubmitCountDecode: number;
  submitTimesMs: number[];
  gpuTimePrefillMs?: number;
  gpuTimeDecodeMs?: number;
  gpuReadbackBytes: number;
  peakVramBytes: number;
  perfSubmits: number;
  perfAllocations: number;
  perfReadbacks: number;
}

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Run a quick benchmark with defaults.
 */
export async function runQuickBenchmark(modelPath: string): Promise<BenchmarkResult> {
  const harness = new PipelineBenchmark({
    modelPath,
    promptName: 'short',
    maxNewTokens: 32,
    warmupRuns: 1,
    timedRuns: 1,
  });
  return harness.run();
}

/**
 * Run a full benchmark suite.
 */
export async function runFullBenchmark(modelPath: string): Promise<BenchmarkResult[]> {
  const results: BenchmarkResult[] = [];

  for (const promptName of ['short', 'medium', 'long'] as const) {
    const harness = new PipelineBenchmark({
      modelPath,
      promptName,
      maxNewTokens: 128,
      warmupRuns: 2,
      timedRuns: 3,
    });
    results.push(await harness.run());
  }

  return results;
}

/**
 * Format result as readable summary.
 */
export function formatBenchmarkSummary(result: BenchmarkResult): string {
  const m = result.metrics;
  return [
    `=== ${result.model.modelName ?? result.model.modelId} ===`,
    `Prompt: ${result.workload.promptName} (${result.workload.promptTokens} tokens)`,
    `TTFT: ${m.ttft_ms}ms`,
    `Prefill: ${m.prefill_ms}ms (${m.prefill_tokens_per_sec} tok/s)`,
    `Decode: ${m.decode_ms_total}ms (${m.decode_tokens_per_sec} tok/s)`,
    `GPU Submits: ${m.gpu_submit_count_prefill} prefill, ${m.gpu_submit_count_decode} decode`,
    m.decode_ms_per_token_p50 ? `Latency P50/P90/P99: ${m.decode_ms_per_token_p50}/${m.decode_ms_per_token_p90}/${m.decode_ms_per_token_p99}ms` : '',
  ].filter(Boolean).join('\n');
}
