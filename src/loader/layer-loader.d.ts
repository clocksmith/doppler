/**
 * Layer Loader - Load transformer layer weights.
 *
 * Handles loading of all weights for a single transformer layer:
 * - Attention weights (Q, K, V, O projections)
 * - Norm weights (input, post-attention, FFN norms)
 * - FFN weights (gate, up, down projections)
 * - MoE router weights
 *
 * @module loader/layer-loader
 */

import type { WeightBuffer } from '../gpu/weight-buffer.js';
import type { TensorLocation, LayerWeights } from './loader-types.js';

/** Tensor loading function signature */
export type TensorLoader = (
  name: string,
  toGPU?: boolean,
  silent?: boolean
) => Promise<GPUBuffer | WeightBuffer | Float32Array | Uint8Array | null>;

/**
 * Context required for layer loading.
 */
export interface LayerLoaderContext {
  /** Tensor locations map */
  tensorLocations: Map<string, TensorLocation>;
  /** Load a tensor by name */
  loadTensor: TensorLoader;
  /** Check if model needs norm weight offset */
  needsNormWeightOffset: () => boolean;
  /** GPU buffers to track for cleanup */
  gpuBuffers: Set<GPUBuffer>;
  /** Keep F32 weights (skip downcast) */
  keepF32Weights: boolean;
  /** Whether model is MoE */
  isMoE: boolean;
  /** Check if layer is an expert layer */
  isExpertLayer: (layerIdx: number) => boolean;
  /** Optional attention dimensions for fused QKV sizing */
  numHeads?: number;
  numKVHeads?: number;
  headDim?: number;
  hiddenSize?: number;
  /** Optional linear-attention dimensions for hybrid models (e.g. Qwen3.5) */
  linearNumKeyHeads?: number;
  linearNumValueHeads?: number;
  linearKeyHeadDim?: number;
  linearValueHeadDim?: number;
}

/**
 * Load all weights for a single transformer layer.
 *
 * @param ctx - Layer loader context
 * @param layerIdx - Layer index
 * @returns Loaded layer weights
 */
export declare function loadLayer(
  ctx: LayerLoaderContext,
  layerIdx: number
): Promise<LayerWeights>;
