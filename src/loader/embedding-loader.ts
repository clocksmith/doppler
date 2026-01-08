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

import {
  createWeightBuffer,
  createCpuWeightBuffer,
  isWeightBuffer,
  type WeightBuffer,
  type WeightDtype,
  type WeightLayout,
  type CpuWeightBuffer,
} from '../gpu/weight-buffer.js';
import type { TensorLocation } from './loader-types.js';
import { maybeDowncastToF16 } from './weight-downcast.js';
import { log } from '../debug/index.js';

// ============================================================================
// Types
// ============================================================================

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

// ============================================================================
// Constants
// ============================================================================

/** Known embedding tensor names in order of preference */
const EMBEDDING_NAMES = [
  'language_model.model.embed_tokens.weight',
  'model.embed_tokens.weight',
  'embed_tokens.weight',
  'token_embd.weight',
  'wte.weight',
  'transformer.wte.weight',
];

// ============================================================================
// Main Function
// ============================================================================

/**
 * Load embedding weights.
 *
 * @param ctx - Embedding loader context
 * @returns Loaded embeddings or null if not found
 */
export async function loadEmbeddings(ctx: EmbeddingLoaderContext): Promise<EmbeddingResult> {
  for (const name of EMBEDDING_NAMES) {
    const loc = ctx.tensorLocations.get(name);
    const shouldStream = loc ? ctx.shouldStreamLargeWeight(name, loc, 'Embedding') : false;

    // Load tensor (to CPU if streaming, to GPU otherwise)
    const tensor = await ctx.loadTensor(name, !shouldStream, true);

    // Skip if not found
    if (!tensor) continue;

    // Handle streaming path (CPU)
    if (shouldStream && !(tensor instanceof Float32Array)) {
      throw new Error(
        `[Loader] Embedding "${name}" too large for GPU and cannot be loaded on CPU (dtype=${loc?.dtype ?? 'unknown'}).`
      );
    }

    // Handle valid tensor types
    if (tensor instanceof GPUBuffer || isWeightBuffer(tensor) || tensor instanceof Float32Array) {
      const result = await processEmbeddingTensor(ctx, tensor, name, loc, shouldStream);
      if (result) {
        return result;
      }
    }
  }

  log.warn('Loader', 'Embeddings not found');
  return null;
}

// ============================================================================
// Internal Helpers
// ============================================================================

/**
 * Process a loaded embedding tensor.
 */
async function processEmbeddingTensor(
  ctx: EmbeddingLoaderContext,
  tensor: GPUBuffer | WeightBuffer | Float32Array,
  name: string,
  loc: TensorLocation | undefined,
  shouldStream: boolean
): Promise<EmbeddingResult> {
  log.info(
    'Loader',
    `Embeddings tensor loaded: name=${name}, hasShape=${!!loc?.shape}, ` +
    `shape=${loc?.shape ? `[${loc.shape.join(',')}]` : 'none'}, isWeightBuffer=${isWeightBuffer(tensor)}`
  );

  // WeightBuffer already has layout set correctly from _loadTensor
  if (isWeightBuffer(tensor)) {
    return maybeDowncastEmbeddings(ctx, tensor, name, loc);
  }

  // Float32Array streaming path
  if (tensor instanceof Float32Array && loc?.shape && shouldStream) {
    const layout = ctx.resolveWeightLayout(loc, name);
    const dtype: WeightDtype = loc.dtype === 'F16' ? 'f16' : 'f32';
    const result = createCpuWeightBuffer(tensor, dtype, layout, loc.shape, name);
    log.warn('Loader', `Embeddings stored on CPU for chunked gather (layout=${layout})`);
    return result;
  }

  // Raw GPUBuffer - wrap with dtype/layout metadata
  if (tensor instanceof GPUBuffer && loc?.shape && loc.shape.length === 2) {
    const layout = ctx.resolveWeightLayout(loc, name);
    const dtype: WeightDtype = loc.dtype === 'F16' ? 'f16' : 'f32';
    const wrapped = createWeightBuffer(tensor, dtype, layout, loc.shape, name);
    log.info('Loader', `Wrapped embeddings as WeightBuffer (layout=${layout}, dtype=${dtype})`);
    return maybeDowncastEmbeddings(ctx, wrapped, name, loc);
  }

  // Fall back to raw tensor
  return maybeDowncastEmbeddings(ctx, tensor, name, loc);
}

/**
 * Attempt to downcast embeddings from F32 to F16.
 */
async function maybeDowncastEmbeddings(
  ctx: EmbeddingLoaderContext,
  current: GPUBuffer | WeightBuffer | Float32Array,
  name: string,
  loc: TensorLocation | undefined
): Promise<EmbeddingResult> {
  // Can't downcast Float32Array or CpuWeightBuffer
  if (current instanceof Float32Array) {
    return current;
  }

  // Get current dtype
  const dtype = isWeightBuffer(current)
    ? current.dtype
    : (loc?.dtype === 'F16' ? 'f16' : 'f32');

  // Skip if not F32
  if (dtype !== 'f32') {
    return current;
  }

  // Get buffer for downcast
  const buffer = isWeightBuffer(current) ? current.buffer : current;
  const elems = buffer.size / 4;

  // Attempt downcast
  const result = await maybeDowncastToF16(current, {
    label: name,
    keepF32: ctx.keepF32Weights,
    shape: isWeightBuffer(current)
      ? Array.from(current.shape)
      : (loc?.shape ?? [elems]),
    layout: isWeightBuffer(current)
      ? current.layout
      : (loc ? ctx.resolveWeightLayout(loc, name) : 'row'),
  });

  if (result?.wasDowncast && result.newBuffer) {
    ctx.gpuBuffers.add(result.newBuffer);
    return result.buffer;
  }

  return current;
}
