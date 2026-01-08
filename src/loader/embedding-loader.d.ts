/**
 * Embedding Loader - Load embedding weights.
 *
 * Handles loading of token embedding weights with support for:
 * - GPU and CPU paths
 * - Large weight streaming
 * - F32 to F16 downcast
 * - WeightBuffer wrapping
 *
 * @module loader/embedding-loader
 */

import type {
  WeightBuffer,
  WeightLayout,
  CpuWeightBuffer,
} from '../gpu/weight-buffer.js';
import type { TensorLocation } from './loader-types.js';

/** Tensor loading function signature */
export type TensorLoader = (
  name: string,
  toGPU?: boolean,
  silent?: boolean
) => Promise<GPUBuffer | WeightBuffer | Float32Array | Uint8Array | null>;

/**
 * Context required for embedding loading.
 */
export interface EmbeddingLoaderContext {
  /** Tensor locations map */
  tensorLocations: Map<string, TensorLocation>;
  /** Load a tensor by name */
  loadTensor: TensorLoader;
  /** Check if large weight should stream to CPU */
  shouldStreamLargeWeight: (name: string, loc: TensorLocation, label: string) => boolean;
  /** Resolve weight layout from location */
  resolveWeightLayout: (loc: TensorLocation, name: string) => WeightLayout;
  /** GPU buffers to track for cleanup */
  gpuBuffers: Set<GPUBuffer>;
  /** Keep F32 weights (skip downcast) */
  keepF32Weights: boolean;
}

/** Result of embedding loading */
export type EmbeddingResult = GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null;

/**
 * Load embedding weights.
 *
 * @param ctx - Embedding loader context
 * @returns Loaded embeddings or null if not found
 */
export declare function loadEmbeddings(ctx: EmbeddingLoaderContext): Promise<EmbeddingResult>;
