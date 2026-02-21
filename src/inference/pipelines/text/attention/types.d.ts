/**
 * Attention Types and Utilities
 *
 * Shared interfaces, debug helpers, and utility functions for attention operations.
 *
 * @module inference/pipelines/text/attention/types
 */

import type { CommandRecorder } from '../../../../gpu/kernel-selector.js';
import type { KVCacheInterface } from '../types.js';
import type { Tensor } from '../../../../gpu/tensor.js';

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
  /** Expected activation dtype from runtime config. */
  activationDtype?: 'f16' | 'f32';
  slidingWindow?: number | null;
  layerType?: string;
  /** Residual tensor for fused o_proj + residual add (decode only) */
  residualTensor?: Tensor | null;
  /** Skip input RMSNorm even if weights are present */
  skipInputNorm?: boolean;
  /** Gemma 2 attention softcapping: score = tanh(score / softcap) * softcap. 0 = disabled. */
  attnSoftcap?: number;
  /** Gemma 2 attention scaling: uses head_dim (256) instead of sqrt(head_dim) (16). */
  queryPreAttnScalar?: number;
  /** Apply query/key RMSNorm even when per-head weights are absent. */
  queryKeyNorm?: boolean;
  /** Gemma 2 RMS scaling: (1+w)*x */
  rmsNormWeightOffset?: boolean;
  /** Token IDs for the current micro-batch (required by BDPA KV ingestion). */
  tokenIds?: number[] | null;
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
  output: Tensor;
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

/**
 * Check if a layer should be debugged
 */
export function shouldDebugLayer(layerIdx: number, debugLayers: number[] | null | undefined): boolean;

/**
 * Check if a stage has been logged for a layer, and mark it as logged
 */
export function markStageLogged(layerIdx: number, stage: string, flags: AttentionDebugFlags): boolean;

/**
 * Release buffer or track for later cleanup (recording mode).
 */
export function releaseOrTrack(recorder: CommandRecorder | undefined, buffer: GPUBuffer): void;

/**
 * Get or create a buffer of ones for Q/K norm when per-head weights are absent.
 */
export function getQKNormOnesBuffer(headDim: number): GPUBuffer;
