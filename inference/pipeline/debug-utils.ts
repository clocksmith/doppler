/**
 * Debug utilities for pipeline tracing.
 *
 * Toggleable log categories for surgical debugging without noise.
 * Enable via: setDebugCategories({ embed: true, layer: true })
 *
 * Categories:
 * - embed: Embedding layer output
 * - layer: Per-layer entry/exit, hidden state stats
 * - attn: Attention computation details
 * - ffn: FFN computation details
 * - kv: KV cache operations
 * - logits: Logits computation and top-k
 * - sample: Sampling decisions
 * - io: GPU buffer read/writes
 * - perf: Timing and benchmarks
 * - kernel: Kernel step debugging - inspect tensor state after each kernel (SLOW!)
 * - all: Enable everything (use sparingly)
 *
 * Log format: [CATEGORY] message
 * This enables post-filtering: grep -E "^\[LAYER\]|\[ATTN\]"
 *
 * Kernel Step Debugging:
 * Enable with setDebugCategories({ kernel: true }, { bufferStats: true })
 * or use DEBUG_PRESETS.kernelStep
 *
 * Use dumpTensor() to inspect any GPU buffer's contents.
 * Use dumpKVCache() to inspect KV cache state for a layer.
 *
 * @example
 * // Enable kernel step debugging for layer 0 only
 * setDebugCategories({ kernel: true }, { layers: [0], bufferStats: true });
 *
 * // In pipeline code:
 * if (isKernelDebugEnabled(layerIdx)) {
 *   await dumpTensor(outputBuffer, 'matmul_output', { layerIdx });
 * }
 *
 * @module inference/pipeline/debug-utils
 */

import { readBuffer } from '../../gpu/buffer-pool.js';
import type { KVCache, SlidingWindowKVCache } from '../kv-cache.js';

// ============================================================================
// Debug Configuration
// ============================================================================

export type DebugCategory =
  | 'embed'
  | 'layer'
  | 'attn'
  | 'ffn'
  | 'kv'
  | 'logits'
  | 'sample'
  | 'io'
  | 'perf'
  | 'kernel'  // Log after every kernel operation (expensive!)
  | 'all';

export interface DebugConfig {
  /** Which categories are enabled */
  categories: Partial<Record<DebugCategory, boolean>>;
  /** Only log these layer indices (empty = all) */
  layers?: number[];
  /** Only log first N decode steps (0 = all) */
  maxDecodeSteps?: number;
  /** Warn if maxAbs exceeds this */
  maxAbsThreshold?: number;
  /** Log GPU buffer stats (expensive - requires readback) */
  bufferStats?: boolean;
}

const defaultConfig: DebugConfig = {
  categories: {},
  layers: [],
  maxDecodeSteps: 5,
  maxAbsThreshold: 10000,
  bufferStats: false,
};

let config: DebugConfig = { ...defaultConfig };
let decodeStep = 0;

// ============================================================================
// Configuration API
// ============================================================================

/**
 * Set debug categories. Merges with existing config.
 *
 * @example
 * setDebugCategories({ embed: true, layer: true });
 * setDebugCategories({ all: true }); // Enable everything
 * setDebugCategories({ all: true, io: false }); // All except io
 */
export function setDebugCategories(
  categories: Partial<Record<DebugCategory, boolean>>,
  options?: Partial<Omit<DebugConfig, 'categories'>>
): void {
  config = {
    ...config,
    ...options,
    categories: { ...config.categories, ...categories },
  };
}

/**
 * Reset debug config to defaults (all off).
 */
export function resetDebugConfig(): void {
  config = { ...defaultConfig, categories: {} };
  decodeStep = 0;
}

/**
 * Get current debug config (for inspection).
 */
export function getDebugConfig(): DebugConfig {
  return { ...config };
}

/**
 * Increment decode step counter.
 */
export function incrementDecodeStep(): number {
  return ++decodeStep;
}

/**
 * Reset decode step counter (call at start of generation).
 */
export function resetDecodeStep(): void {
  decodeStep = 0;
}

/**
 * Get current decode step.
 */
export function getDecodeStep(): number {
  return decodeStep;
}

// ============================================================================
// Internal Helpers
// ============================================================================

function isEnabled(category: DebugCategory, layerIdx?: number): boolean {
  // Check if category is enabled
  if (!config.categories.all && !config.categories[category]) {
    return false;
  }

  // Check layer filter
  if (layerIdx !== undefined && config.layers?.length) {
    if (!config.layers.includes(layerIdx)) {
      return false;
    }
  }

  // Check decode step limit
  if (config.maxDecodeSteps && decodeStep > config.maxDecodeSteps) {
    // Only apply to non-prefill logs
    if (decodeStep > 0) {
      return false;
    }
  }

  return true;
}

function formatTag(category: string, layerIdx?: number, step?: number): string {
  let tag = `[${category.toUpperCase()}]`;
  if (layerIdx !== undefined) tag += `[L${layerIdx}]`;
  if (step !== undefined) tag += `[S${step}]`;
  return tag;
}

// ============================================================================
// Logging Functions
// ============================================================================

/**
 * Log embedding info.
 */
export function logEmbed(
  tokenIds: number[],
  info: { maxAbs?: number; nonZero?: number; total?: number; sample?: number[] }
): void {
  if (!isEnabled('embed')) return;

  const tag = formatTag('embed');
  const tokens = tokenIds.length > 3
    ? `[${tokenIds.slice(0, 3).join(',')},...] (${tokenIds.length} total)`
    : `[${tokenIds.join(',')}]`;

  let msg = `${tag} tokens=${tokens}`;
  if (info.maxAbs !== undefined) msg += ` maxAbs=${info.maxAbs.toFixed(2)}`;
  if (info.nonZero !== undefined) msg += ` nonZero=${info.nonZero}/${info.total}`;
  if (info.sample?.length) msg += ` sample=[${info.sample.map(v => v.toFixed(3)).join(',')}]`;

  console.log(msg);
}

/**
 * Log layer entry/exit.
 */
export function logLayer(
  layerIdx: number,
  phase: 'enter' | 'exit',
  isPrefill: boolean,
  info: { numTokens?: number; maxAbs?: number; sample?: number[] }
): void {
  if (!isEnabled('layer', layerIdx)) return;

  const tag = formatTag('layer', layerIdx);
  const mode = isPrefill ? 'prefill' : `decode:${decodeStep}`;

  let msg = `${tag} ${phase} ${mode}`;
  if (info.numTokens !== undefined) msg += ` n=${info.numTokens}`;
  if (info.maxAbs !== undefined) msg += ` maxAbs=${info.maxAbs.toFixed(2)}`;
  if (info.sample?.length) msg += ` sample=[${info.sample.map(v => v.toFixed(3)).join(',')}]`;

  // Warn on explosion
  if (info.maxAbs !== undefined && config.maxAbsThreshold && info.maxAbs > config.maxAbsThreshold) {
    msg += ` ☡ EXPLOSION`;
  }

  console.log(msg);
}

/**
 * Log attention details.
 */
export function logAttn(
  layerIdx: number,
  isPrefill: boolean,
  info: {
    numTokens: number;
    kvLen: number;
    startPos?: number;
    maxAbsQ?: number;
    maxAbsK?: number;
    maxAbsV?: number;
    maxAbsOut?: number;
  }
): void {
  if (!isEnabled('attn', layerIdx)) return;

  const tag = formatTag('attn', layerIdx);
  const mode = isPrefill ? 'prefill' : `decode:${decodeStep}`;

  let msg = `${tag} ${mode} n=${info.numTokens} kvLen=${info.kvLen}`;
  if (info.startPos !== undefined) msg += ` startPos=${info.startPos}`;
  if (info.maxAbsQ !== undefined) msg += ` Q=${info.maxAbsQ.toFixed(1)}`;
  if (info.maxAbsK !== undefined) msg += ` K=${info.maxAbsK.toFixed(1)}`;
  if (info.maxAbsV !== undefined) msg += ` V=${info.maxAbsV.toFixed(1)}`;
  if (info.maxAbsOut !== undefined) msg += ` out=${info.maxAbsOut.toFixed(1)}`;

  console.log(msg);
}

/**
 * Log FFN details.
 */
export function logFFN(
  layerIdx: number,
  info: { maxAbsGate?: number; maxAbsUp?: number; maxAbsOut?: number }
): void {
  if (!isEnabled('ffn', layerIdx)) return;

  const tag = formatTag('ffn', layerIdx);
  let msg = tag;
  if (info.maxAbsGate !== undefined) msg += ` gate=${info.maxAbsGate.toFixed(1)}`;
  if (info.maxAbsUp !== undefined) msg += ` up=${info.maxAbsUp.toFixed(1)}`;
  if (info.maxAbsOut !== undefined) msg += ` out=${info.maxAbsOut.toFixed(1)}`;

  console.log(msg);
}

/**
 * Log KV cache operations.
 */
export function logKV(
  layerIdx: number,
  op: 'write' | 'read' | 'init' | 'clear',
  info: { seqLen?: number; kvLen?: number; startPos?: number }
): void {
  if (!isEnabled('kv', layerIdx)) return;

  const tag = formatTag('kv', layerIdx);
  let msg = `${tag} ${op}`;
  if (info.seqLen !== undefined) msg += ` seqLen=${info.seqLen}`;
  if (info.kvLen !== undefined) msg += ` kvLen=${info.kvLen}`;
  if (info.startPos !== undefined) msg += ` startPos=${info.startPos}`;

  console.log(msg);
}

/**
 * Log logits computation.
 */
export function logLogits(
  phase: 'prefill' | 'decode',
  info: {
    min: number;
    max: number;
    topK?: Array<{ token: number | string; prob: number; text?: string }>;
  }
): void {
  if (!isEnabled('logits')) return;

  const tag = phase === 'prefill' ? '[LOGITS][PREFILL]' : `[LOGITS][S${decodeStep}]`;
  let msg = `${tag} min=${info.min.toFixed(2)} max=${info.max.toFixed(2)}`;

  if (info.topK?.length) {
    const topStr = info.topK
      .slice(0, 5)
      .map(t => `"${t.text || t.token}"(${(t.prob * 100).toFixed(1)}%)`)
      .join(', ');
    msg += ` top-5: ${topStr}`;
  }

  console.log(msg);
}

/**
 * Log sampling decision.
 */
export function logSample(
  tokenId: number,
  tokenText: string,
  info: { prob?: number; temperature?: number; topK?: number }
): void {
  if (!isEnabled('sample')) return;

  const tag = decodeStep === 0 ? '[SAMPLE][PREFILL]' : `[SAMPLE][S${decodeStep}]`;
  let msg = `${tag} -> ${tokenId} "${tokenText}"`;
  if (info.prob !== undefined) msg += ` p=${(info.prob * 100).toFixed(1)}%`;
  if (info.temperature !== undefined) msg += ` T=${info.temperature}`;

  console.log(msg);
}

/**
 * Log GPU buffer I/O.
 */
export function logIO(
  op: 'read' | 'write' | 'copy',
  label: string,
  bytes: number
): void {
  if (!isEnabled('io')) return;

  const tag = '[IO]';
  const kb = (bytes / 1024).toFixed(1);
  console.log(`${tag} ${op} ${label}: ${kb}KB`);
}

/**
 * Log performance timing.
 */
export function logPerf(
  phase: string,
  ms: number,
  extra?: Record<string, number | string>
): void {
  if (!isEnabled('perf')) return;

  const tag = '[PERF]';
  let msg = `${tag} ${phase}: ${ms.toFixed(1)}ms`;
  if (extra) {
    for (const [k, v] of Object.entries(extra)) {
      msg += ` ${k}=${typeof v === 'number' ? v.toFixed(1) : v}`;
    }
  }

  console.log(msg);
}

// ============================================================================
// Buffer Stats Helper (Expensive - use sparingly)
// ============================================================================

/**
 * Read GPU buffer and compute stats. Only use when bufferStats is enabled.
 */
export async function getBufferStats(
  buffer: GPUBuffer
): Promise<{ min: number; max: number; maxAbs: number; sample: number[]; nanCount: number } | null> {
  if (!config.bufferStats) return null;

  try {
    const data = await readBuffer(buffer);
    const arr = new Float32Array(data);
    let min = Infinity;
    let max = -Infinity;
    let nanCount = 0;

    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (!Number.isFinite(v)) {
        nanCount++;
      } else {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }

    const maxAbs = Math.max(Math.abs(min), Math.abs(max));
    const sample = Array.from(arr.slice(0, 5));

    return { min, max, maxAbs, sample, nanCount };
  } catch {
    return null;
  }
}

// ============================================================================
// Convenience: Preset Configurations
// ============================================================================

export const DEBUG_PRESETS = {
  /** Quick check: just embedding and final logits */
  quick: { embed: true, logits: true, sample: true } as Partial<Record<DebugCategory, boolean>>,

  /** Layer tracing: watch values flow through layers */
  layers: { layer: true } as Partial<Record<DebugCategory, boolean>>,

  /** Attention focus: debug attention computation */
  attention: { attn: true, kv: true } as Partial<Record<DebugCategory, boolean>>,

  /** Full trace: everything (very verbose) */
  full: { all: true } as Partial<Record<DebugCategory, boolean>>,

  /** Performance only: timing info */
  perf: { perf: true } as Partial<Record<DebugCategory, boolean>>,

  /** Kernel step debugging: inspect tensor state after every kernel (very slow!) */
  kernelStep: { kernel: true } as Partial<Record<DebugCategory, boolean>>,
};

// ============================================================================
// Kernel Step Debugging (Expensive - sync after every op)
// ============================================================================

/** Tensor statistics from GPU readback */
export interface TensorStats {
  shape: string;
  dtype: string;
  min: number;
  max: number;
  maxAbs: number;
  mean: number;
  nonZero: number;
  total: number;
  nanCount: number;
  infCount: number;
  sample: number[];
}

/**
 * Dump a GPU tensor's contents for debugging.
 * This is expensive (requires GPU sync + readback) - use sparingly.
 *
 * @param buffer - GPU buffer to inspect
 * @param label - Descriptive label for logging
 * @param options - Additional options
 */
export async function dumpTensor(
  buffer: GPUBuffer,
  label: string,
  options: {
    layerIdx?: number;
    shape?: [number, number] | [number];
    dtype?: 'f32' | 'f16' | 'bf16';
    sampleCount?: number;
    warnThreshold?: number;
  } = {}
): Promise<TensorStats | null> {
  if (!isEnabled('kernel', options.layerIdx)) return null;

  const { shape, dtype = 'f32', sampleCount = 8, warnThreshold = 10000 } = options;

  try {
    const data = await readBuffer(buffer);
    const arr = dtype === 'f32' ? new Float32Array(data) : new Float32Array(data);

    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let nanCount = 0;
    let infCount = 0;
    let nonZero = 0;

    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (Number.isNaN(v)) {
        nanCount++;
      } else if (!Number.isFinite(v)) {
        infCount++;
      } else {
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        if (v !== 0) nonZero++;
      }
    }

    const maxAbs = Math.max(Math.abs(min), Math.abs(max));
    const mean = sum / (arr.length - nanCount - infCount);
    const sample = Array.from(arr.slice(0, sampleCount));

    const shapeStr = shape ? `[${shape.join('x')}]` : `[${arr.length}]`;

    const stats: TensorStats = {
      shape: shapeStr,
      dtype,
      min,
      max,
      maxAbs,
      mean,
      nonZero,
      total: arr.length,
      nanCount,
      infCount,
      sample,
    };

    // Format log message
    const tag = options.layerIdx !== undefined
      ? `[KERNEL][L${options.layerIdx}]`
      : '[KERNEL]';

    let msg = `${tag} ${label} ${shapeStr}`;
    msg += ` min=${min.toFixed(3)} max=${max.toFixed(3)} maxAbs=${maxAbs.toFixed(3)}`;
    msg += ` mean=${mean.toFixed(3)} nonZero=${nonZero}/${arr.length}`;

    if (nanCount > 0) msg += ` NaN=${nanCount}`;
    if (infCount > 0) msg += ` Inf=${infCount}`;

    msg += `\n  sample=[${sample.map(v => v.toFixed(4)).join(', ')}]`;

    // Warnings
    if (maxAbs > warnThreshold) {
      msg += `\n  ☡ VALUE EXPLOSION: maxAbs=${maxAbs.toFixed(1)} > ${warnThreshold}`;
    }
    if (nanCount > 0 || infCount > 0) {
      msg += `\n  ☡ NUMERICAL INSTABILITY: ${nanCount} NaN, ${infCount} Inf`;
    }
    if (nonZero === 0 && arr.length > 0) {
      msg += `\n  ☡ ALL ZEROS`;
    }

    console.log(msg);
    return stats;
  } catch (e) {
    console.log(`[KERNEL] ${label} ERROR: ${e}`);
    return null;
  }
}

/**
 * Dump stats for a single token row within a 2D [numTokens, rowSize] buffer.
 * Use this when matching per-token reference implementations (e.g., HuggingFace hooks).
 */
export async function dumpTokenVector(
  buffer: GPUBuffer,
  label: string,
  options: {
    layerIdx?: number;
    tokenIdx: number;
    rowSize: number;
    dtype?: 'f32' | 'f16' | 'bf16';
    sampleCount?: number;
    warnThreshold?: number;
  }
): Promise<TensorStats | null> {
  if (!isEnabled('kernel', options.layerIdx)) return null;

  const {
    tokenIdx,
    rowSize,
    dtype = 'f32',
    sampleCount = 8,
    warnThreshold = 10000,
  } = options;

  try {
    const data = await readBuffer(buffer);
    const arr = dtype === 'f32' ? new Float32Array(data) : new Float32Array(data);

    const offset = tokenIdx * rowSize;
    const end = offset + rowSize;
    if (offset < 0 || end > arr.length) {
      console.log(`[KERNEL] ${label} ERROR: token slice out of bounds (tokenIdx=${tokenIdx}, rowSize=${rowSize}, len=${arr.length})`);
      return null;
    }

    const row = arr.subarray(offset, end);

    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let nanCount = 0;
    let infCount = 0;
    let nonZero = 0;

    for (let i = 0; i < row.length; i++) {
      const v = row[i];
      if (Number.isNaN(v)) {
        nanCount++;
      } else if (!Number.isFinite(v)) {
        infCount++;
      } else {
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        if (v !== 0) nonZero++;
      }
    }

    const maxAbs = Math.max(Math.abs(min), Math.abs(max));
    const denom = row.length - nanCount - infCount;
    const mean = denom > 0 ? sum / denom : 0;
    const sample = Array.from(row.slice(0, sampleCount));

    const shapeStr = `[t${tokenIdx}x${rowSize}]`;

    const stats: TensorStats = {
      shape: shapeStr,
      dtype,
      min,
      max,
      maxAbs,
      mean,
      nonZero,
      total: row.length,
      nanCount,
      infCount,
      sample,
    };

    const tag = options.layerIdx !== undefined
      ? `[KERNEL][L${options.layerIdx}]`
      : '[KERNEL]';

    let msg = `${tag} ${label} ${shapeStr}`;
    msg += ` min=${min.toFixed(3)} max=${max.toFixed(3)} maxAbs=${maxAbs.toFixed(3)}`;
    msg += ` mean=${mean.toFixed(3)} nonZero=${nonZero}/${row.length}`;

    if (nanCount > 0) msg += ` NaN=${nanCount}`;
    if (infCount > 0) msg += ` Inf=${infCount}`;

    msg += `\n  sample=[${sample.map(v => v.toFixed(4)).join(', ')}]`;

    if (maxAbs > warnThreshold) {
      msg += `\n  ☡ VALUE EXPLOSION: maxAbs=${maxAbs.toFixed(1)} > ${warnThreshold}`;
    }
    if (nanCount > 0 || infCount > 0) {
      msg += `\n  ☡ NUMERICAL INSTABILITY: ${nanCount} NaN, ${infCount} Inf`;
    }
    if (nonZero === 0 && row.length > 0) {
      msg += `\n  ☡ ALL ZEROS`;
    }

    console.log(msg);
    return stats;
  } catch (e) {
    console.log(`[KERNEL] ${label} ERROR: ${e}`);
    return null;
  }
}

/**
 * Log a kernel step with optional tensor dump.
 * Use this after kernel invocations to trace execution.
 *
 * @param kernelName - Name of the kernel (e.g., 'matmul', 'rmsnorm')
 * @param info - Additional info to log
 */
export function logKernelStep(
  kernelName: string,
  info: {
    layerIdx?: number;
    M?: number;
    N?: number;
    K?: number;
    size?: number;
    label?: string;
  }
): void {
  if (!isEnabled('kernel', info.layerIdx)) return;

  const tag = info.layerIdx !== undefined
    ? `[KERNEL][L${info.layerIdx}]`
    : '[KERNEL]';

  let msg = `${tag} ${kernelName}`;
  if (info.label) msg += ` (${info.label})`;
  if (info.M !== undefined) msg += ` M=${info.M}`;
  if (info.N !== undefined) msg += ` N=${info.N}`;
  if (info.K !== undefined) msg += ` K=${info.K}`;
  if (info.size !== undefined) msg += ` size=${info.size}`;

  console.log(msg);
}

/**
 * Dump KV cache state for a specific layer.
 * Reads both keys and values buffers and reports statistics.
 *
 * @param kvCache - KV cache instance
 * @param layerIdx - Layer index to inspect
 */
export async function dumpKVCache(
  kvCache: KVCache | SlidingWindowKVCache,
  layerIdx: number
): Promise<{ keys: TensorStats | null; values: TensorStats | null } | null> {
  if (!isEnabled('kernel', layerIdx) && !isEnabled('kv', layerIdx)) return null;

  const tag = `[KV][L${layerIdx}]`;

  try {
    if (!kvCache?.hasGPUCache?.()) {
      console.log(`${tag} No GPU cache available`);
      return null;
    }

    const gpuBuffers = kvCache.getGPUBuffers(layerIdx);
    if (!gpuBuffers) {
      console.log(`${tag} No buffers for layer ${layerIdx}`);
      return null;
    }

    const { keysGPU, valuesGPU, seqLen } = gpuBuffers;
    const numHeads = kvCache.numHeads || 0;
    const headDim = kvCache.headDim || 0;

    console.log(`${tag} seqLen=${seqLen} numHeads=${numHeads} headDim=${headDim}`);

    const keysStats = keysGPU
      ? await dumpTensor(keysGPU, 'K_cache', {
          layerIdx,
          shape: [seqLen, numHeads * headDim],
        })
      : null;

    const valuesStats = valuesGPU
      ? await dumpTensor(valuesGPU, 'V_cache', {
          layerIdx,
          shape: [seqLen, numHeads * headDim],
        })
      : null;

    return { keys: keysStats, values: valuesStats };
  } catch (e) {
    console.log(`${tag} ERROR: ${e}`);
    return null;
  }
}

/**
 * Check if kernel step debugging is enabled.
 * Use this to gate expensive debug operations.
 */
export function isKernelDebugEnabled(layerIdx?: number): boolean {
  return isEnabled('kernel', layerIdx);
}
