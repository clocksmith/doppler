/**
 * Loader State - Encapsulates loaded model state.
 *
 * Manages the state of a loaded model including:
 * - Layer weights (attention, FFN, norms)
 * - Embeddings and LM head
 * - GPU buffer tracking for cleanup
 * - Expert weights for MoE models
 *
 * @module loader/loader-state
 */

import { releaseBuffer } from '../gpu/buffer-pool.js';
import {
  isWeightBuffer,
  isCpuWeightBuffer,
  type WeightBuffer,
  type CpuWeightBuffer,
} from '../gpu/weight-buffer.js';
import type { LayerWeights } from './loader-types.js';
import type { ExpertWeights } from './weights.js';
import { trace as debugTrace } from '../debug/index.js';

// ============================================================================
// Types
// ============================================================================

/** Possible types for embedding/lmHead weights */
export type EmbeddingWeight = GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null;

/** Possible types for norm weights */
export type NormWeight = GPUBuffer | Float32Array | null;

/** State snapshot for serialization/debugging */
export interface LoaderStateSnapshot {
  isLoaded: boolean;
  layerCount: number;
  expertCount: number;
  gpuBufferCount: number;
  hasEmbeddings: boolean;
  hasLmHead: boolean;
  hasFinalNorm: boolean;
}

// ============================================================================
// LoaderState Class
// ============================================================================

/**
 * Encapsulates the state of a loaded model.
 *
 * Provides methods for:
 * - Adding/accessing layer weights
 * - Managing GPU buffer lifecycle
 * - Tracking expert weights for MoE
 * - Clean state reset
 */
export class LoaderState {
  // Model weights
  embeddings: EmbeddingWeight = null;
  lmHead: EmbeddingWeight = null;
  finalNorm: NormWeight = null;

  // Layer weights indexed by layer number
  readonly layers = new Map<number, LayerWeights>();

  // Expert weights for MoE models (key: "layer_expert")
  readonly experts = new Map<string, ExpertWeights>();

  // GPU buffers to release on unload
  readonly gpuBuffers = new Set<GPUBuffer>();

  // Loaded state flag
  isLoaded = false;

  // ============================================================================
  // Layer Management
  // ============================================================================

  /**
   * Set weights for a layer.
   */
  setLayer(layerIndex: number, weights: LayerWeights): void {
    this.layers.set(layerIndex, weights);
  }

  /**
   * Get weights for a layer.
   */
  getLayer(layerIndex: number): LayerWeights | undefined {
    return this.layers.get(layerIndex);
  }

  /**
   * Check if a layer is loaded.
   */
  hasLayer(layerIndex: number): boolean {
    return this.layers.has(layerIndex);
  }

  /**
   * Get all layer indices.
   */
  getLayerIndices(): number[] {
    return Array.from(this.layers.keys()).sort((a, b) => a - b);
  }

  // ============================================================================
  // Expert Management (MoE)
  // ============================================================================

  /**
   * Generate expert key from layer and expert indices.
   */
  static expertKey(layerIndex: number, expertIndex: number): string {
    return `${layerIndex}_${expertIndex}`;
  }

  /**
   * Set weights for an expert.
   */
  setExpert(layerIndex: number, expertIndex: number, weights: ExpertWeights): void {
    const key = LoaderState.expertKey(layerIndex, expertIndex);
    this.experts.set(key, weights);
  }

  /**
   * Get weights for an expert.
   */
  getExpert(layerIndex: number, expertIndex: number): ExpertWeights | undefined {
    const key = LoaderState.expertKey(layerIndex, expertIndex);
    return this.experts.get(key);
  }

  /**
   * Check if an expert is loaded.
   */
  hasExpert(layerIndex: number, expertIndex: number): boolean {
    const key = LoaderState.expertKey(layerIndex, expertIndex);
    return this.experts.has(key);
  }

  // ============================================================================
  // GPU Buffer Management
  // ============================================================================

  /**
   * Track a GPU buffer for cleanup on unload.
   */
  trackBuffer(buffer: GPUBuffer): void {
    this.gpuBuffers.add(buffer);
  }

  /**
   * Track multiple GPU buffers.
   */
  trackBuffers(buffers: GPUBuffer[]): void {
    for (const buffer of buffers) {
      this.gpuBuffers.add(buffer);
    }
  }

  /**
   * Release a specific GPU buffer.
   */
  releaseBuffer(buffer: GPUBuffer): void {
    if (this.gpuBuffers.has(buffer)) {
      releaseBuffer(buffer);
      this.gpuBuffers.delete(buffer);
    }
  }

  /**
   * Release all tracked GPU buffers.
   */
  releaseAllBuffers(): void {
    for (const buffer of this.gpuBuffers) {
      releaseBuffer(buffer);
    }
    this.gpuBuffers.clear();
  }

  // ============================================================================
  // State Queries
  // ============================================================================

  /**
   * Get a snapshot of current state for debugging.
   */
  getSnapshot(): LoaderStateSnapshot {
    return {
      isLoaded: this.isLoaded,
      layerCount: this.layers.size,
      expertCount: this.experts.size,
      gpuBufferCount: this.gpuBuffers.size,
      hasEmbeddings: this.embeddings !== null,
      hasLmHead: this.lmHead !== null,
      hasFinalNorm: this.finalNorm !== null,
    };
  }

  /**
   * Check if any weights are loaded.
   */
  hasAnyWeights(): boolean {
    return (
      this.embeddings !== null ||
      this.lmHead !== null ||
      this.finalNorm !== null ||
      this.layers.size > 0 ||
      this.experts.size > 0
    );
  }

  // ============================================================================
  // State Reset
  // ============================================================================

  /**
   * Clear all loaded state and release GPU resources.
   */
  clear(): void {
    debugTrace.loader('Clearing loader state...');

    // Release all GPU buffers
    this.releaseAllBuffers();

    // Clear weight references
    this.embeddings = null;
    this.lmHead = null;
    this.finalNorm = null;

    // Clear collections
    this.layers.clear();
    this.experts.clear();

    // Reset loaded flag
    this.isLoaded = false;

    debugTrace.loader('Loader state cleared');
  }

  /**
   * Prepare for loading a new model (clears state if needed).
   */
  prepareForLoad(): void {
    if (this.hasAnyWeights()) {
      debugTrace.loader('Clearing existing state before new load');
      this.clear();
    }
  }

  /**
   * Mark loading as complete.
   */
  markLoaded(): void {
    this.isLoaded = true;
  }

  // ============================================================================
  // Weight Type Helpers
  // ============================================================================

  /**
   * Extract GPU buffer from various weight types.
   */
  static getGPUBuffer(weight: EmbeddingWeight | NormWeight): GPUBuffer | null {
    if (!weight) return null;
    if (weight instanceof GPUBuffer) return weight;
    if (isWeightBuffer(weight)) return weight.buffer;
    return null;
  }

  /**
   * Check if weight is a GPU-backed type.
   */
  static isGPUBacked(weight: EmbeddingWeight | NormWeight): boolean {
    if (!weight) return false;
    if (weight instanceof GPUBuffer) return true;
    if (isWeightBuffer(weight)) return true;
    if (isCpuWeightBuffer(weight)) return false;
    if (weight instanceof Float32Array) return false;
    return false;
  }
}

// ============================================================================
// Factory
// ============================================================================

/**
 * Create a new loader state instance.
 */
export function createLoaderState(): LoaderState {
  return new LoaderState();
}
