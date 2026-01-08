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
import { isWeightBuffer, isCpuWeightBuffer } from '../gpu/weight-buffer.js';
import { trace as debugTrace } from '../debug/index.js';

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
  /** @type {GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | import('../gpu/weight-buffer.js').CpuWeightBuffer | Float32Array | null} */
  embeddings = null;

  /** @type {GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | import('../gpu/weight-buffer.js').CpuWeightBuffer | Float32Array | null} */
  lmHead = null;

  /** @type {GPUBuffer | Float32Array | null} */
  finalNorm = null;

  /** @type {Map<number, import('./loader-types.js').LayerWeights>} */
  layers = new Map();

  /** @type {Map<string, import('./weights.js').ExpertWeights>} */
  experts = new Map();

  /** @type {Set<GPUBuffer>} */
  gpuBuffers = new Set();

  /** @type {boolean} */
  isLoaded = false;

  // ============================================================================
  // Layer Management
  // ============================================================================

  /**
   * Set weights for a layer.
   * @param {number} layerIndex
   * @param {import('./loader-types.js').LayerWeights} weights
   */
  setLayer(layerIndex, weights) {
    this.layers.set(layerIndex, weights);
  }

  /**
   * Get weights for a layer.
   * @param {number} layerIndex
   * @returns {import('./loader-types.js').LayerWeights | undefined}
   */
  getLayer(layerIndex) {
    return this.layers.get(layerIndex);
  }

  /**
   * Check if a layer is loaded.
   * @param {number} layerIndex
   * @returns {boolean}
   */
  hasLayer(layerIndex) {
    return this.layers.has(layerIndex);
  }

  /**
   * Get all layer indices.
   * @returns {number[]}
   */
  getLayerIndices() {
    return Array.from(this.layers.keys()).sort((a, b) => a - b);
  }

  // ============================================================================
  // Expert Management (MoE)
  // ============================================================================

  /**
   * Generate expert key from layer and expert indices.
   * @param {number} layerIndex
   * @param {number} expertIndex
   * @returns {string}
   */
  static expertKey(layerIndex, expertIndex) {
    return `${layerIndex}_${expertIndex}`;
  }

  /**
   * Set weights for an expert.
   * @param {number} layerIndex
   * @param {number} expertIndex
   * @param {import('./weights.js').ExpertWeights} weights
   */
  setExpert(layerIndex, expertIndex, weights) {
    const key = LoaderState.expertKey(layerIndex, expertIndex);
    this.experts.set(key, weights);
  }

  /**
   * Get weights for an expert.
   * @param {number} layerIndex
   * @param {number} expertIndex
   * @returns {import('./weights.js').ExpertWeights | undefined}
   */
  getExpert(layerIndex, expertIndex) {
    const key = LoaderState.expertKey(layerIndex, expertIndex);
    return this.experts.get(key);
  }

  /**
   * Check if an expert is loaded.
   * @param {number} layerIndex
   * @param {number} expertIndex
   * @returns {boolean}
   */
  hasExpert(layerIndex, expertIndex) {
    const key = LoaderState.expertKey(layerIndex, expertIndex);
    return this.experts.has(key);
  }

  // ============================================================================
  // GPU Buffer Management
  // ============================================================================

  /**
   * Track a GPU buffer for cleanup on unload.
   * @param {GPUBuffer} buffer
   */
  trackBuffer(buffer) {
    this.gpuBuffers.add(buffer);
  }

  /**
   * Track multiple GPU buffers.
   * @param {GPUBuffer[]} buffers
   */
  trackBuffers(buffers) {
    for (const buffer of buffers) {
      this.gpuBuffers.add(buffer);
    }
  }

  /**
   * Release a specific GPU buffer.
   * @param {GPUBuffer} buffer
   */
  releaseBuffer(buffer) {
    if (this.gpuBuffers.has(buffer)) {
      releaseBuffer(buffer);
      this.gpuBuffers.delete(buffer);
    }
  }

  /**
   * Release all tracked GPU buffers.
   */
  releaseAllBuffers() {
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
   * @returns {import('./loader-state.js').LoaderStateSnapshot}
   */
  getSnapshot() {
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
   * @returns {boolean}
   */
  hasAnyWeights() {
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
  clear() {
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
  prepareForLoad() {
    if (this.hasAnyWeights()) {
      debugTrace.loader('Clearing existing state before new load');
      this.clear();
    }
  }

  /**
   * Mark loading as complete.
   */
  markLoaded() {
    this.isLoaded = true;
  }

  // ============================================================================
  // Weight Type Helpers
  // ============================================================================

  /**
   * Extract GPU buffer from various weight types.
   * @param {GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | import('../gpu/weight-buffer.js').CpuWeightBuffer | Float32Array | null} weight
   * @returns {GPUBuffer | null}
   */
  static getGPUBuffer(weight) {
    if (!weight) return null;
    if (weight instanceof GPUBuffer) return weight;
    if (isWeightBuffer(weight)) return weight.buffer;
    return null;
  }

  /**
   * Check if weight is a GPU-backed type.
   * @param {GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | import('../gpu/weight-buffer.js').CpuWeightBuffer | Float32Array | null} weight
   * @returns {boolean}
   */
  static isGPUBacked(weight) {
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
 * @returns {LoaderState}
 */
export function createLoaderState() {
  return new LoaderState();
}
