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
} from '../gpu/weight-buffer.js';
import { maybeDowncastToF16 } from './weight-downcast.js';
import { log } from '../debug/index.js';

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
 * @param {import('./embedding-loader.js').EmbeddingLoaderContext} ctx - Embedding loader context
 * @returns {Promise<import('./embedding-loader.js').EmbeddingResult>} Loaded embeddings or null if not found
 */
export async function loadEmbeddings(ctx) {
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
 * @param {import('./embedding-loader.js').EmbeddingLoaderContext} ctx
 * @param {GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | Float32Array} tensor
 * @param {string} name
 * @param {import('./loader-types.js').TensorLocation | undefined} loc
 * @param {boolean} shouldStream
 * @returns {Promise<import('./embedding-loader.js').EmbeddingResult>}
 */
async function processEmbeddingTensor(ctx, tensor, name, loc, shouldStream) {
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
    /** @type {import('../gpu/weight-buffer.js').WeightDtype} */
    const dtype = loc.dtype === 'F16' ? 'f16' : 'f32';
    const result = createCpuWeightBuffer(tensor, dtype, layout, loc.shape, name);
    log.warn('Loader', `Embeddings stored on CPU for chunked gather (layout=${layout})`);
    return result;
  }

  // Raw GPUBuffer - wrap with dtype/layout metadata
  if (tensor instanceof GPUBuffer && loc?.shape && loc.shape.length === 2) {
    const layout = ctx.resolveWeightLayout(loc, name);
    /** @type {import('../gpu/weight-buffer.js').WeightDtype} */
    const dtype = loc.dtype === 'F16' ? 'f16' : 'f32';
    const wrapped = createWeightBuffer(tensor, dtype, layout, loc.shape, name);
    log.info('Loader', `Wrapped embeddings as WeightBuffer (layout=${layout}, dtype=${dtype})`);
    return maybeDowncastEmbeddings(ctx, wrapped, name, loc);
  }

  // Fall back to raw tensor
  return maybeDowncastEmbeddings(ctx, tensor, name, loc);
}

/**
 * Attempt to downcast embeddings from F32 to F16.
 * @param {import('./embedding-loader.js').EmbeddingLoaderContext} ctx
 * @param {GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | Float32Array} current
 * @param {string} name
 * @param {import('./loader-types.js').TensorLocation | undefined} loc
 * @returns {Promise<import('./embedding-loader.js').EmbeddingResult>}
 */
async function maybeDowncastEmbeddings(ctx, current, name, loc) {
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
