/**
 * Weight buffer management utilities.
 *
 * This module handles:
 * - Creating GPU buffers from CPU weight data
 * - Applying weight transformations (e.g., Gemma norm offset)
 * - Type guards for layer weight structures
 * - Buffer lifecycle management
 *
 * @module inference/pipeline/weights
 */

import type { LayerWeights } from './types.js';
import type { WeightBuffer, CpuWeightBuffer } from '../../gpu/weight-buffer.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Configuration for weight buffer operations.
 */
export interface WeightBufferConfig {
  /** Whether to apply +1 offset for Gemma-style norm weights */
  rmsNormWeightOffset: boolean;
}

/**
 * Debug flags for weight buffer operations.
 */
export interface WeightDebugFlags {
  normBufferTypeLogged?: boolean;
  normOffsetDebugDone?: boolean;
}

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Type guard to check if a weight value is a LayerWeights object.
 */
export function isLayerWeights(value: unknown): value is LayerWeights;

/**
 * Get layer weights from weights map with type narrowing.
 */
export function getLayerWeights(
  weights: Map<string, LayerWeights | Float32Array | GPUBuffer>,
  key: string
): LayerWeights | null;

// ============================================================================
// Weight Buffer Creation
// ============================================================================

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
  weight: GPUBuffer | Float32Array | ArrayBuffer | { buffer: ArrayBuffer; byteOffset: number; byteLength: number } | CpuWeightBuffer,
  label: string,
  config: WeightBufferConfig,
  debugFlags?: WeightDebugFlags
): GPUBuffer;

/**
 * Get GPU weight buffer, ensuring it's on GPU.
 */
export function getGPUWeightBuffer(
  weight: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | ArrayBuffer,
  label: string
): GPUBuffer;

// ============================================================================
// Weight Buffer Factory
// ============================================================================

/**
 * Create weight buffer helper functions bound to a specific config.
 */
export function createWeightBufferHelpers(
  config: WeightBufferConfig,
  debugFlags?: WeightDebugFlags
): {
  getWeightBuffer: (weight: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer | WeightBuffer;
  getNormWeightBuffer: (weight: GPUBuffer | Float32Array | ArrayBuffer | CpuWeightBuffer, label: string) => GPUBuffer;
  getGPUWeightBuffer: (weight: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer;
};

// ============================================================================
// Batch Buffer Tracking
// ============================================================================

/**
 * Buffer tracking for batched command execution.
 */
export class BatchBufferTracker {
  track(buffer: GPUBuffer | Float32Array | ArrayBuffer): void;
  getTracked(): GPUBuffer[];
  clear(): void;
}
