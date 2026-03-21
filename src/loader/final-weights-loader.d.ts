/**
 * Final Weights Loader - Load final norm and LM head.
 *
 * Handles loading of:
 * - Final layer norm (offset applied at runtime when configured)
 * - LM head (output projection)
 * - Tied embeddings fallback
 *
 * @module loader/final-weights-loader
 */

import type {
  WeightBuffer,
  WeightLayout,
  CpuWeightBuffer,
} from '../gpu/weight-buffer.js';
import type {
  ManifestEmbeddingPostprocessorSchema,
  ManifestEmbeddingProjectionSchema,
} from '../config/schema/index.js';
import type { TensorLocation } from './loader-types.js';

/** Tensor loading function signature */
export type TensorLoader = (
  name: string,
  toGPU?: boolean,
  silent?: boolean
) => Promise<GPUBuffer | WeightBuffer | Float32Array | Uint8Array | null>;

export interface LoadedEmbeddingProjection extends ManifestEmbeddingProjectionSchema {
  weight: Float32Array;
  bias: Float32Array | null;
}

export interface LoadedEmbeddingPostprocessor
  extends Omit<ManifestEmbeddingPostprocessorSchema, 'projections'> {
  projections: LoadedEmbeddingProjection[];
}

/**
 * Context required for final weights loading.
 */
export interface FinalWeightsContext {
  /** Tensor locations map */
  tensorLocations: Map<string, TensorLocation>;
  /** Load a tensor by name */
  loadTensor: TensorLoader;
  /** Check if model needs norm weight offset */
  needsNormWeightOffset: () => boolean;
  /** Check if large weight should stream to CPU */
  shouldStreamLargeWeight: (name: string, loc: TensorLocation, label: string) => boolean;
  /** Resolve weight layout from location */
  resolveWeightLayout: (loc: TensorLocation) => WeightLayout;
  /** Current embeddings (for tied embeddings fallback) */
  embeddings: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null;
  /** Optional embedding postprocessor declared by manifest */
  embeddingPostprocessor?: ManifestEmbeddingPostprocessorSchema | null;
  /** Manifest model type */
  modelType?: string | null;
  /** Whether LM head should fall back to tied embeddings */
  tieWordEmbeddings: boolean;
  /** GPU buffers to track for cleanup */
  gpuBuffers: Set<GPUBuffer>;
  /** Keep F32 weights (skip downcast) */
  keepF32Weights: boolean;
  /** Whether debug log for norm offset has been done */
  normOffsetDebugLogged: boolean;
}

/** Result of final weights loading */
export interface FinalWeightsResult {
  /** Final layer norm tensor */
  finalNorm: GPUBuffer | Float32Array | null;
  /** LM head tensor */
  lmHead: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null;
  /** Optional embedding-only postprocessor weights */
  embeddingPostprocessor: LoadedEmbeddingPostprocessor | null;
  /** Whether norm offset debug was logged */
  normOffsetDebugLogged: boolean;
}

/**
 * Load final layer norm and LM head weights.
 *
 * @param ctx - Final weights loader context
 * @returns Loaded final weights
 */
export declare function loadFinalWeights(ctx: FinalWeightsContext): Promise<FinalWeightsResult>;
