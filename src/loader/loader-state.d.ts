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

import type { WeightBuffer, CpuWeightBuffer } from '../gpu/weight-buffer.js';
import type { LayerWeights } from './loader-types.js';
import type { ExpertWeights } from './weights.js';

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

/**
 * Encapsulates the state of a loaded model.
 *
 * Provides methods for:
 * - Adding/accessing layer weights
 * - Managing GPU buffer lifecycle
 * - Tracking expert weights for MoE
 * - Clean state reset
 */
export declare class LoaderState {
  embeddings: EmbeddingWeight;
  lmHead: EmbeddingWeight;
  finalNorm: NormWeight;
  readonly layers: Map<number, LayerWeights>;
  readonly experts: Map<string, ExpertWeights>;
  readonly gpuBuffers: Set<GPUBuffer>;
  isLoaded: boolean;

  /**
   * Set weights for a layer.
   */
  setLayer(layerIndex: number, weights: LayerWeights): void;

  /**
   * Get weights for a layer.
   */
  getLayer(layerIndex: number): LayerWeights | undefined;

  /**
   * Check if a layer is loaded.
   */
  hasLayer(layerIndex: number): boolean;

  /**
   * Get all layer indices.
   */
  getLayerIndices(): number[];

  /**
   * Generate expert key from layer and expert indices.
   */
  static expertKey(layerIndex: number, expertIndex: number): string;

  /**
   * Set weights for an expert.
   */
  setExpert(layerIndex: number, expertIndex: number, weights: ExpertWeights): void;

  /**
   * Get weights for an expert.
   */
  getExpert(layerIndex: number, expertIndex: number): ExpertWeights | undefined;

  /**
   * Check if an expert is loaded.
   */
  hasExpert(layerIndex: number, expertIndex: number): boolean;

  /**
   * Track a GPU buffer for cleanup on unload.
   */
  trackBuffer(buffer: GPUBuffer): void;

  /**
   * Track multiple GPU buffers.
   */
  trackBuffers(buffers: GPUBuffer[]): void;

  /**
   * Release a specific GPU buffer.
   */
  releaseBuffer(buffer: GPUBuffer): void;

  /**
   * Release all tracked GPU buffers.
   */
  releaseAllBuffers(): void;

  /**
   * Get a snapshot of current state for debugging.
   */
  getSnapshot(): LoaderStateSnapshot;

  /**
   * Check if any weights are loaded.
   */
  hasAnyWeights(): boolean;

  /**
   * Clear all loaded state and release GPU resources.
   */
  clear(): void;

  /**
   * Prepare for loading a new model (clears state if needed).
   */
  prepareForLoad(): void;

  /**
   * Mark loading as complete.
   */
  markLoaded(): void;

  /**
   * Extract GPU buffer from various weight types.
   */
  static getGPUBuffer(weight: EmbeddingWeight | NormWeight): GPUBuffer | null;

  /**
   * Check if weight is a GPU-backed type.
   */
  static isGPUBacked(weight: EmbeddingWeight | NormWeight): boolean;
}

/**
 * Create a new loader state instance.
 */
export declare function createLoaderState(): LoaderState;
