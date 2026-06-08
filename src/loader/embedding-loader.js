

import {
  createWeightBuffer,
  createCpuWeightBuffer,
  createSplitWeightBuffer,
  isWeightBuffer,
  isCpuWeightBuffer,
  isSplitWeightBuffer,
  getWeightDtype,
  getLayout,
} from '../gpu/weight-buffer.js';
import { getDevice } from '../gpu/device.js';
import { maybeDowncastToF16 } from './weight-downcast.js';
import { getTensorNamesByRole } from './tensors/tensor-role.js';
import { log } from '../debug/index.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import { DTYPE_SIZES } from '../config/schema/index.js';
import { createTensor } from '../gpu/tensor.js';
import { castF16ToF32 } from '../gpu/kernel-selector.js';
import { releaseBuffer } from '../memory/buffer-pool.js';
import { assembleShardData, loadTensorRange } from './tensors/tensor-reader.js';
import { hasSourceTransform } from './tensors/source-transform.js';
import { getLargeWeightMaxBytes } from './manifest-config.js';

// ============================================================================
// Constants
// ============================================================================


const EMBEDDING_ROLE = 'embedding';
const EMBEDDING_GROUP = 'embed';
const MAX_SPLIT_EMBEDDING_SECTIONS = 4;
const SPLIT_UPLOAD_CHUNK_BYTES = 64 * 1024 * 1024;

function isPerLayerEmbeddingTensor(name) {
  return String(name ?? '').toLowerCase().includes('embed_tokens_per_layer.weight');
}

function isGpuBufferInstance(value) {
  return typeof GPUBuffer !== 'undefined' && value instanceof GPUBuffer;
}

function normalizeLiteRTInt4StorageEncoding(transform, name) {
  const storageEncoding = String(transform?.storageEncoding ?? '').toLowerCase();
  if (storageEncoding !== 'signed' && storageEncoding !== 'offset_binary') {
    throw new Error(
      `[Loader] Embedding "${name}" LiteRT INT4 sourceTransform.storageEncoding must be ` +
      `"signed" or "offset_binary", got "${transform?.storageEncoding ?? 'missing'}".`
    );
  }
  return storageEncoding;
}

function createRangeBackedTensorSource(ctx, name, location) {
  if (typeof ctx.loadShardRange !== 'function') {
    return null;
  }
  const normalizedLocationDtype = typeof location?.dtype === 'string'
    ? location.dtype.toLowerCase()
    : 'f32';
  return {
    kind: 'tensor_range_source',
    sourceDtype: normalizedLocationDtype,
    async loadRange(byteOffset, byteLength) {
      return loadTensorRange(location, name, byteOffset, byteLength, ctx.loadShardRange);
    },
  };
}

function createRangeBackedWeightBuffer(ctx, name, location) {
  const source = createRangeBackedTensorSource(ctx, name, location);
  if (!source || !location?.shape || location.shape.length !== 2) {
    return null;
  }
  const layout = ctx.resolveWeightLayout(location);
  const dtype = selectRuleValue('loader', 'weights', 'floatLocationDtype', {
    locationDtype: location.dtype,
  });
  return createCpuWeightBuffer(source, dtype, layout, location.shape, name);
}

function isFixedLiteRTInt4AffineEmbedding(ctx, name, location) {
  if (ctx.hostHasShaderF16 === false) {
    return false;
  }
  if (!location?.shape || location.shape.length !== 2) {
    return false;
  }
  if (location.role !== EMBEDDING_ROLE) {
    return false;
  }
  if (ctx.resolveWeightLayout(location) !== 'row') {
    return false;
  }
  const transform = location.sourceTransform;
  if (!transform || transform.kind !== 'affine_dequant') {
    return false;
  }
  if (transform.scheme !== 'per_tensor_affine') {
    return false;
  }
  if (String(transform.sourceDtype ?? '').toUpperCase() !== 'INT4') {
    return false;
  }
  if (String(transform.targetDtype ?? '').toUpperCase() !== 'F16') {
    return false;
  }
  normalizeLiteRTInt4StorageEncoding(transform, name);
  if (Number(transform.scale) !== 0.0625 || Number(transform.zeroPoint) !== 0) {
    return false;
  }
  const [, hiddenSize] = location.shape;
  const expectedBytes = location.shape[0] * Math.ceil(hiddenSize / 2);
  if (location.size !== expectedBytes) {
    throw new Error(
      `[Loader] Embedding "${name}" LiteRT INT4 source size mismatch: ` +
      `manifest size=${location.size}, expected=${expectedBytes}.`
    );
  }
  return typeof ctx.loadShardRange === 'function';
}

async function createLiteRTInt4GpuEmbeddingWeightBuffer(ctx, name, location) {
  if (!isFixedLiteRTInt4AffineEmbedding(ctx, name, location)) {
    return null;
  }
  const device = getDevice();
  if (!device) {
    return null;
  }
  const rawBytes = await assembleShardData(location, name, null, ctx.loadShardRange, {
    materializeSourceTransform: false,
  });
  if (rawBytes.byteLength !== location.size) {
    throw new Error(
      `[Loader] Embedding "${name}" LiteRT INT4 raw read ${rawBytes.byteLength} bytes, ` +
      `expected ${location.size}.`
    );
  }
  const buffer = device.createBuffer({
    label: `${name}:litert_int4`,
    size: alignByteLength(rawBytes.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  try {
    writeBufferInChunks(device.queue, buffer, rawBytes);
  } catch (error) {
    buffer.destroy();
    throw error;
  }
  ctx.gpuBuffers.add(buffer);
  log.info(
    'Loader',
    `Embedding "${name}" stored as packed LiteRT INT4 GPU source ` +
    `(bytes=${rawBytes.byteLength}, layout=row).`
  );
  return createWeightBuffer(buffer, 'litert_int4', 'row', location.shape, name, null, {
    storageEncoding: normalizeLiteRTInt4StorageEncoding(location.sourceTransform, name),
  });
}

function getEmbeddingFloatDtype(location) {
  return selectRuleValue('loader', 'weights', 'floatLocationDtype', {
    locationDtype: location?.dtype,
  });
}

function alignByteLength(byteLength) {
  return Math.ceil(byteLength / 4) * 4;
}

function writeBufferInChunks(queue, buffer, bytes) {
  for (let offset = 0; offset < bytes.byteLength; offset += SPLIT_UPLOAD_CHUNK_BYTES) {
    const end = Math.min(offset + SPLIT_UPLOAD_CHUNK_BYTES, bytes.byteLength);
    queue.writeBuffer(buffer, offset, bytes, offset, end - offset);
  }
}

function expectsSplitGpuEmbedding(ctx) {
  return ctx.embeddingKernel?.kernel === 'gather_split4_f16_vec4_f16_out.wgsl'
    && ctx.embeddingKernel?.entry === 'gather_vec4_f16_out';
}

async function createSplitGpuEmbeddingWeightBuffer(ctx, name, location) {
  if (!expectsSplitGpuEmbedding(ctx)) {
    return null;
  }
  if (ctx.hostHasShaderF16 === false) {
    return null;
  }
  if (!location?.shape || location.shape.length !== 2) {
    return null;
  }
  if (hasSourceTransform(location)) {
    return null;
  }
  if (typeof ctx.loadShardRange !== 'function') {
    return null;
  }

  const layout = ctx.resolveWeightLayout(location);
  if (layout !== 'row') {
    return null;
  }

  const dtype = getEmbeddingFloatDtype(location);
  if (dtype !== 'f16') {
    return null;
  }

  const device = getDevice();
  const maxBytes = getLargeWeightMaxBytes();
  if (!device || !maxBytes) {
    return null;
  }
  if ((device.limits.maxStorageBuffersPerShaderStage ?? 0) < 6) {
    log.warn(
      'Loader',
      `Embedding "${name}" cannot use split GPU residency: maxStorageBuffersPerShaderStage is below 6.`
    );
    return null;
  }

  const [rows, hidden] = location.shape;
  const bytesPerElement = DTYPE_SIZES[dtype];
  const rowBytes = hidden * bytesPerElement;
  if (!Number.isFinite(rowBytes) || rowBytes <= 0 || rowBytes > maxBytes) {
    return null;
  }

  const totalBytes = rows * rowBytes;
  if (totalBytes <= maxBytes) {
    return null;
  }

  const rowsPerSection = Math.floor(maxBytes / rowBytes);
  if (rowsPerSection <= 0) {
    return null;
  }
  const sectionCount = Math.ceil(rows / rowsPerSection);
  if (sectionCount > MAX_SPLIT_EMBEDDING_SECTIONS) {
    log.warn(
      'Loader',
      `Embedding "${name}" needs ${sectionCount} GPU sections; split4 supports at most ${MAX_SPLIT_EMBEDDING_SECTIONS}. ` +
      'Using CPU-backed streaming.'
    );
    return null;
  }

  const createdBuffers = [];
  try {
    const sections = [];
    for (let rowStart = 0; rowStart < rows; rowStart += rowsPerSection) {
      const rowCount = Math.min(rowsPerSection, rows - rowStart);
      const byteOffset = rowStart * rowBytes;
      const byteLength = rowCount * rowBytes;
      const bytes = await loadTensorRange(location, name, byteOffset, byteLength, ctx.loadShardRange);
      if (bytes.byteLength !== byteLength) {
        throw new Error(
          `[Loader] Embedding "${name}" split section read ${bytes.byteLength} bytes, expected ${byteLength}.`
        );
      }
      const buffer = device.createBuffer({
        label: `${name}:split:${sections.length}`,
        size: alignByteLength(byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      createdBuffers.push(buffer);
      writeBufferInChunks(device.queue, buffer, bytes);
      sections.push({ buffer, rowStart, rowCount });
    }
    for (const buffer of createdBuffers) {
      ctx.gpuBuffers.add(buffer);
    }
    log.warn(
      'Loader',
      `Embedding "${name}" stored as split GPU sections (${sections.length} sections, dtype=${dtype}, layout=${layout}).`
    );
    return createSplitWeightBuffer(sections, dtype, layout, location.shape, name);
  } catch (error) {
    for (const buffer of createdBuffers) {
      try {
        buffer.destroy();
      } catch {
        // Ignore cleanup failure while preserving the original load error.
      }
    }
    throw error;
  }
}

function shouldUseRangeBackedEmbeddingSource(ctx, name, location) {
  if (!location) {
    return false;
  }
  if (String(location.dtype ?? '').toUpperCase() === 'F16' && ctx.hostHasShaderF16 === false) {
    log.info(
      'Loader',
      `Embedding "${name}" range-backed: F16 source on device without shader-f16 support.`
    );
    return true;
  }
  if (hasSourceTransform(location) && typeof ctx.loadShardRange === 'function') {
    log.info(
      'Loader',
      `Embedding "${name}" range-backed: sourceTransform.kind="${location.sourceTransform.kind}" defers full materialization.`
    );
    return true;
  }
  const stream = ctx.shouldStreamLargeWeight(name, location, 'Embedding');
  if (!stream) {
    log.info(
      'Loader',
      `Embedding "${name}" GPU-resident: shouldStreamLargeWeight returned false.`
    );
  }
  return stream;
}

// ============================================================================
// Main Function
// ============================================================================


export async function loadEmbeddings(ctx) {
  const embeddingNames = getTensorNamesByRole(ctx.tensorLocations, EMBEDDING_ROLE, EMBEDDING_GROUP);
  const groupedCandidates = embeddingNames.length > 0
    ? embeddingNames
    : getTensorNamesByRole(ctx.tensorLocations, EMBEDDING_ROLE);
  const filteredCandidates = groupedCandidates.filter((name) => !isPerLayerEmbeddingTensor(name));
  const candidates = filteredCandidates.length > 0 ? filteredCandidates : groupedCandidates;

  if (candidates.length === 0) {
    throw new Error(
      `[Loader] Embeddings not found. Expected tensor with role="${EMBEDDING_ROLE}"` +
      ` and group="${EMBEDDING_GROUP}". Re-convert the model with tensor roles.`
    );
  }

  for (const name of candidates) {
    const loc = ctx.tensorLocations.get(name);
    const packedLiteRTTensor = await createLiteRTInt4GpuEmbeddingWeightBuffer(ctx, name, loc);
    const shouldStream = packedLiteRTTensor ? false : shouldUseRangeBackedEmbeddingSource(ctx, name, loc);
    const splitGpuTensor = packedLiteRTTensor ? null : await createSplitGpuEmbeddingWeightBuffer(ctx, name, loc);

    // Load tensor (to CPU if streaming, to GPU otherwise)
    const tensor = packedLiteRTTensor ?? splitGpuTensor ?? (shouldStream
      ? (
        createRangeBackedWeightBuffer(ctx, name, loc)
        ?? await ctx.loadTensor(name, false, true)
      )
      : await ctx.loadTensor(name, true, true));
    const tensorShouldStream = splitGpuTensor ? false : shouldStream;

    // Skip if not found
    if (!tensor) continue;

    // Handle streaming path (CPU)
    if (tensorShouldStream && !(tensor instanceof Float32Array) && !isCpuWeightBuffer(tensor)) {
      throw new Error(
        `[Loader] Embedding "${name}" too large for GPU and cannot be loaded on CPU (dtype=${loc?.dtype ?? 'unknown'}).`
      );
    }

    // Handle valid tensor types
    if (isGpuBufferInstance(tensor) || isWeightBuffer(tensor) || isCpuWeightBuffer(tensor) || isSplitWeightBuffer(tensor) || tensor instanceof Float32Array) {
      const result = await processEmbeddingTensor(ctx, tensor, name, loc, tensorShouldStream);
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

  if (isSplitWeightBuffer(promoted)) {
    return promoted;
  }

  // WeightBuffer already has layout set correctly from _loadTensor
  if (isWeightBuffer(promoted)) {
    return maybeDowncastEmbeddings(ctx, promoted, name, loc);
  }

  if (isCpuWeightBuffer(promoted)) {
    log.warn('Loader', `Embeddings stored on CPU via range-backed source (layout=${promoted.layout})`);
    return promoted;
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
  if (isGpuBufferInstance(promoted) && loc?.shape && loc.shape.length === 2) {
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
  if (isCpuWeightBuffer(current)) return current;
  if (isSplitWeightBuffer(current)) return current;

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

  if (!isGpuBufferInstance(current)) return current;

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
  if (current instanceof Float32Array || isCpuWeightBuffer(current) || isSplitWeightBuffer(current)) {
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
