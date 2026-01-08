/**
 * Attention Types and Utilities
 *
 * Shared interfaces, debug helpers, and utility functions for attention operations.
 *
 * @module inference/pipeline/attention/types
 */

import { getDevice } from '../../../gpu/device.js';
import { releaseBuffer } from '../../../gpu/buffer-pool.js';
import type { CommandRecorder } from '../../../gpu/kernel-selector.js';
import type { KVCacheInterface } from '../types.js';

// ============================================================================
// Attention Interfaces
// ============================================================================

/**
 * Attention configuration for a layer.
 */
export interface AttentionConfig {
  layerIdx: number;
  numTokens: number;
  isPrefill: boolean;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  hiddenSize: number;
  rmsNormEps: number;
  currentSeqLen: number;
  slidingWindow?: number | null;
  layerType?: string;
  /** Residual tensor for fused o_proj + residual add (decode only) */
  residualTensor?: import('../../../gpu/tensor.js').Tensor | null;
  /** Skip input RMSNorm even if weights are present */
  skipInputNorm?: boolean;
  /** Gemma 2 attention softcapping: score = tanh(score / softcap) * softcap. 0 = disabled. */
  attnSoftcap?: number;
  /** Gemma 2 attention scaling: uses head_dim (256) instead of sqrt(head_dim) (16). */
  queryPreAttnScalar?: number;
  /** Apply query/key RMSNorm even when per-head weights are absent. */
  queryKeyNorm?: boolean;
}

/**
 * Attention state passed between operations.
 */
export interface AttentionState {
  ropeFreqsCos: GPUBuffer | null;
  ropeFreqsSin: GPUBuffer | null;
  kvCache: KVCacheInterface;
}

/**
 * Result from attention layer execution.
 */
export interface AttentionResult {
  /** Output tensor after attention + o_proj */
  output: import('../../../gpu/tensor.js').Tensor;
  /** Whether the attention residual was fused into o_proj (layer.ts should skip residual add) */
  residualFused: boolean;
}

/**
 * Debug flags for attention - tracks which layer/stage combos have been logged.
 * Uses a Set of "L{layer}_{stage}" keys to prevent duplicate logging.
 */
export interface AttentionDebugFlags {
  /** Layers to debug (null = none, empty = layer 0 only for backward compat) */
  debugLayers?: number[] | null;
  /** Set of "L{layer}_{stage}" keys that have been logged */
  loggedStages?: Set<string>;
}

// ============================================================================
// Debug Helpers
// ============================================================================

/**
 * Check if a layer should be debugged
 */
export function shouldDebugLayer(layerIdx: number, debugLayers: number[] | null | undefined): boolean {
  if (debugLayers === null) return false;
  if (debugLayers === undefined || debugLayers.length === 0) {
    // Backward compat: default to layer 0 only
    return layerIdx === 0;
  }
  return debugLayers.includes(layerIdx);
}

/**
 * Check if a stage has been logged for a layer, and mark it as logged
 */
export function markStageLogged(layerIdx: number, stage: string, flags: AttentionDebugFlags): boolean {
  if (!flags.loggedStages) {
    flags.loggedStages = new Set();
  }
  const key = `L${layerIdx}_${stage}`;
  if (flags.loggedStages.has(key)) {
    return true; // Already logged
  }
  flags.loggedStages.add(key);
  return false; // First time
}

/**
 * Release buffer or track for later cleanup (recording mode).
 */
export function releaseOrTrack(recorder: CommandRecorder | undefined, buffer: GPUBuffer): void {
  if (recorder) {
    recorder.trackTemporaryBuffer(buffer);
  } else {
    releaseBuffer(buffer);
  }
}

// ============================================================================
// Q/K Norm Cache
// ============================================================================

const qkNormOnesCache = new Map<number, GPUBuffer>();

/**
 * Get or create a buffer of ones for Q/K norm when per-head weights are absent.
 */
export function getQKNormOnesBuffer(headDim: number): GPUBuffer {
  const cached = qkNormOnesCache.get(headDim);
  if (cached) return cached;
  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for Q/K norm buffer');
  }
  const data = new Float32Array(headDim);
  data.fill(1);
  const buffer = device.createBuffer({
    label: `qk_norm_ones_${headDim}`,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  qkNormOnesCache.set(headDim, buffer);
  return buffer;
}
