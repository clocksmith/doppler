/**
 * Weight buffer management utilities.
 *
 * This module handles:
 * - Creating GPU buffers from CPU weight data
 * - Handling RMSNorm weight buffers (offset is applied at runtime)
 * - Type guards for layer weight structures
 * - Buffer lifecycle management
 *
 * @module inference/pipelines/text/weights
 */

import type { LayerWeights } from './types.js';
import type { WeightBuffer, CpuWeightBuffer } from '../../../gpu/weight-buffer.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Configuration for weight buffer operations.
 */
export interface WeightBufferConfig {
  /** Whether RMSNorm uses (1 + weight) scaling at runtime */
  rmsNormWeightOffset: boolean;
}

/**
 * Debug flags for weight buffer operations.
 */
export interface WeightDebugFlags {
  normBufferTypeLogged?: boolean;
  normOffsetDebugDone?: boolean;
}

/**
 * Get layer weights from weights map with type narrowing.
 */
export function getLayerWeights(
  weights: Map<string, LayerWeights | Float32Array | GPUBuffer>,
  key: string
): LayerWeights | null;

/**
 * Get or create GPU buffer for a weight tensor.
 */
export function getWeightBuffer(
  weight: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | ArrayBuffer,
  label: string
): GPUBuffer | WeightBuffer;

/**
 * Get or create GPU buffer for RMSNorm weight tensor.
 */
export function getNormWeightBuffer(
  weight: GPUBuffer | WeightBuffer | Float32Array | ArrayBuffer | { buffer: ArrayBuffer; byteOffset: number; byteLength: number } | CpuWeightBuffer,
  label: string,
  config: WeightBufferConfig,
  debugFlags?: WeightDebugFlags
): GPUBuffer;
