

import {
  createWeightBuffer,
  createCpuWeightBuffer,
  isWeightBuffer,
  getWeightDtype,
  getLayout,
} from '../gpu/weight-buffer.js';
import { maybeDowncastToF16 } from './weight-downcast.js';
import { getTensorNamesByRole } from './tensor-role.js';
import { log } from '../debug/index.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import { createTensor } from '../gpu/tensor.js';
import { castF16ToF32 } from '../gpu/kernel-selector.js';
import { releaseBuffer } from '../memory/buffer-pool.js';

// ============================================================================
// Constants
// ============================================================================


const EMBEDDING_ROLE = 'embedding';
const EMBEDDING_GROUP = 'embed';

// ============================================================================
// Main Function
// ============================================================================


export async function loadEmbeddings(ctx) {
  const embeddingNames = getTensorNamesByRole(ctx.tensorLocations, EMBEDDING_ROLE, EMBEDDING_GROUP);
  const candidates = embeddingNames.length > 0
    ? embeddingNames
    : getTensorNamesByRole(ctx.tensorLocations, EMBEDDING_ROLE);

  if (candidates.length === 0) {
    throw new Error(
      `[Loader] Embeddings not found. Expected tensor with role="${EMBEDDING_ROLE}"` +
      ` and group="${EMBEDDING_GROUP}". Re-convert the model with tensor roles.`
    );
  }

  for (const name of candidates) {
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

  throw new Error(
    `[Loader] Embeddings not found. Tried: ${candidates.join(', ')}`
  );
}

// ============================================================================
// Internal Helpers
// ============================================================================


async function processEmbeddingTensor(ctx, tensor, name, loc, shouldStream) {
  log.info(
    'Loader',
    `Embeddings tensor loaded: name=${name}, hasShape=${!!loc?.shape}, ` +
    `shape=${loc?.shape ? `[${loc.shape.join(',')}]` : 'none'}, isWeightBuffer=${isWeightBuffer(tensor)}`
  );

  // Preserve F32 embedding path when required by manifest/kernels.
  // This also repairs legacy manifests where embeddings were stored as F16
  // but loaded with an F32 gather kernel.
  const promoted = await maybePromoteEmbeddingsToF32(ctx, tensor, name, loc);

  // WeightBuffer already has layout set correctly from _loadTensor
  if (isWeightBuffer(promoted)) {
    return maybeDowncastEmbeddings(ctx, promoted, name, loc);
  }

  // Float32Array streaming path
  if (promoted instanceof Float32Array && loc?.shape && shouldStream) {
    const layout = ctx.resolveWeightLayout(loc);
    
    const dtype = selectRuleValue('loader', 'weights', 'floatLocationDtype', {
      locationDtype: loc.dtype,
    });
    const result = createCpuWeightBuffer(promoted, dtype, layout, loc.shape, name);
    log.warn('Loader', `Embeddings stored on CPU for chunked gather (layout=${layout})`);
    return result;
  }

  // Raw GPUBuffer - wrap with dtype/layout metadata
  if (promoted instanceof GPUBuffer && loc?.shape && loc.shape.length === 2) {
    const layout = ctx.resolveWeightLayout(loc);
    
    const dtype = selectRuleValue('loader', 'weights', 'floatLocationDtype', {
      locationDtype: loc.dtype,
    });
    const wrapped = createWeightBuffer(promoted, dtype, layout, loc.shape, name);
    log.info('Loader', `Wrapped embeddings as WeightBuffer (layout=${layout}, dtype=${dtype})`);
    return maybeDowncastEmbeddings(ctx, wrapped, name, loc);
  }

  // Fall back to raw tensor
  return maybeDowncastEmbeddings(ctx, promoted, name, loc);
}

async function maybePromoteEmbeddingsToF32(ctx, current, name, loc) {
  if (!ctx.preserveF32Embeddings) return current;
  if (current instanceof Float32Array) return current;

  if (isWeightBuffer(current)) {
    const dtype = getWeightDtype(current);
    if (dtype !== 'f16') return current;
    const elems = Math.floor(current.buffer.size / 2);
    const inputTensor = createTensor(current.buffer, 'f16', [elems], `${name}_f16`);
    const promoted = await castF16ToF32(inputTensor);
    const shape = Array.from(current.shape);
    const layout = getLayout(current) || 'row';
    const wrapped = createWeightBuffer(promoted.buffer, 'f32', layout, shape, name);
    ctx.gpuBuffers.add(promoted.buffer);
    releaseBuffer(current.buffer);
    return wrapped;
  }

  if (!(current instanceof GPUBuffer)) return current;

  const sourceDtype = selectRuleValue('loader', 'weights', 'floatLocationDtype', {
    locationDtype: loc?.dtype,
  });
  if (sourceDtype !== 'f16') return current;

  const elems = Math.floor(current.size / 2);
  const inputTensor = createTensor(current, 'f16', [elems], `${name}_f16`);
  const promoted = await castF16ToF32(inputTensor);
  ctx.gpuBuffers.add(promoted.buffer);
  releaseBuffer(current);

  if (loc?.shape && loc.shape.length === 2) {
    const layout = ctx.resolveWeightLayout(loc);
    return createWeightBuffer(promoted.buffer, 'f32', layout, loc.shape, name);
  }

  return promoted.buffer;
}


async function maybeDowncastEmbeddings(ctx, current, name, loc) {
  // Can't downcast Float32Array or CpuWeightBuffer
  if (current instanceof Float32Array) {
    return current;
  }

  // Get current dtype
  const dtype = isWeightBuffer(current)
    ? current.dtype
    : selectRuleValue('loader', 'weights', 'floatLocationDtype', {
      locationDtype: loc?.dtype,
    });

  // Skip if not F32
  if (dtype !== 'f32') {
    return current;
  }

  // Get buffer for downcast
  const buffer = isWeightBuffer(current) ? current.buffer : current;
  const elems = buffer.size / 4;

  // Attempt downcast
  const keepF32 = ctx.keepF32Weights || ctx.preserveF32Embeddings === true;
  const result = await maybeDowncastToF16(current, {
    label: name,
    keepF32,
    dtype,
    shape: isWeightBuffer(current)
      ? Array.from(current.shape)
      : (loc?.shape ?? [elems]),
    layout: isWeightBuffer(current)
      ? current.layout
      : (loc ? ctx.resolveWeightLayout(loc) : 'row'),
  });

  if (result?.wasDowncast && result.newBuffer) {
    ctx.gpuBuffers.add(result.newBuffer);
    return result.buffer;
  }

  return current;
}
