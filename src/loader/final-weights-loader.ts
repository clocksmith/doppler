/**
 * Final Weights Loader - Load final norm and LM head.
 *
 * Handles loading of:
 * - Final layer norm (with optional +1 offset for Gemma 3+)
 * - LM head (output projection)
 * - Tied embeddings fallback
 *
 * @module loader/final-weights-loader
 */

import {
  createWeightBuffer,
  createCpuWeightBuffer,
  isWeightBuffer,
  isCpuWeightBuffer,
  type WeightBuffer,
  type WeightDtype,
  type WeightLayout,
  type CpuWeightBuffer,
} from '../gpu/weight-buffer.js';
import type { TensorLocation } from './loader-types.js';
import { applyNormWeightOffset } from './norm-offset.js';
import { maybeDowncastToF16 } from './weight-downcast.js';
import { log, trace as debugTrace } from '../debug/index.js';

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
  resolveWeightLayout: (loc: TensorLocation, name: string) => WeightLayout;
  /** Current embeddings (for tied embeddings fallback) */
  embeddings: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null;
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
  /** Whether norm offset debug was logged */
  normOffsetDebugLogged: boolean;
}

// ============================================================================
// Constants
// ============================================================================

/** Known final norm tensor names in order of preference */
const FINAL_NORM_NAMES = [
  'language_model.model.norm.weight',
  'model.norm.weight',
  'norm.weight',
  'output_norm.weight',
  'transformer.ln_f.weight',
];

/** Known LM head tensor names in order of preference */
const LM_HEAD_NAMES = [
  'language_model.lm_head.weight',
  'lm_head.weight',
  'output.weight',
];

// ============================================================================
// Main Function
// ============================================================================

/**
 * Load final layer norm and LM head weights.
 *
 * @param ctx - Final weights loader context
 * @returns Loaded final weights
 */
export async function loadFinalWeights(ctx: FinalWeightsContext): Promise<FinalWeightsResult> {
  let normOffsetDebugLogged = ctx.normOffsetDebugLogged;

  // Load final norm
  const { finalNorm, debugLogged: normDebugLogged } = await loadFinalNorm(ctx);
  if (normDebugLogged) {
    normOffsetDebugLogged = true;
  }

  // Load LM head
  const lmHead = await loadLmHead(ctx);

  return {
    finalNorm,
    lmHead,
    normOffsetDebugLogged,
  };
}

// ============================================================================
// Final Norm Loading
// ============================================================================

interface FinalNormResult {
  finalNorm: GPUBuffer | Float32Array | null;
  debugLogged: boolean;
}

async function loadFinalNorm(ctx: FinalWeightsContext): Promise<FinalNormResult> {
  let finalNorm: GPUBuffer | Float32Array | null = null;
  let finalNormElements: number | undefined;
  let debugLogged = false;

  for (const name of FINAL_NORM_NAMES) {
    const location = ctx.tensorLocations.get(name);
    if (location) {
      finalNormElements = location.shape.reduce((a, b) => a * b, 1);
      finalNorm = await ctx.loadTensor(name, true, true) as GPUBuffer | Float32Array | null;
      break;
    }
  }

  if (finalNorm && ctx.needsNormWeightOffset()) {
    const result = await applyNormWeightOffset(finalNorm, {
      actualNumElements: finalNormElements,
      bufferDtype: 'f32',
      enableDebugLog: !ctx.normOffsetDebugLogged,
    });
    finalNorm = result.tensor;
    debugLogged = result.debugLogged;
  }

  if (!finalNorm) {
    log.warn('Loader', 'Final norm not found');
  }

  return { finalNorm, debugLogged };
}

// ============================================================================
// LM Head Loading
// ============================================================================

async function loadLmHead(
  ctx: FinalWeightsContext
): Promise<GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null> {
  let lmHead: GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array | null = null;
  let lmHeadName: string | null = null;
  let lmHeadLoc: TensorLocation | undefined;

  for (const name of LM_HEAD_NAMES) {
    const loc = ctx.tensorLocations.get(name);
    if (!loc) continue;

    const shouldStream = ctx.shouldStreamLargeWeight(name, loc, 'LM head');
    const tensor = await ctx.loadTensor(name, !shouldStream, true);

    if (shouldStream && tensor && !(tensor instanceof Float32Array)) {
      throw new Error(
        `[Loader] LM head "${name}" too large for GPU and cannot be loaded on CPU (dtype=${loc.dtype}).`
      );
    }

    if (tensor && (tensor instanceof GPUBuffer || isWeightBuffer(tensor) || tensor instanceof Float32Array)) {
      lmHeadName = name;
      lmHeadLoc = loc;
      lmHead = processLmHeadTensor(ctx, tensor, name, loc, shouldStream);
      break;
    }
  }

  // Use tied embeddings as fallback
  if (!lmHead && ctx.embeddings) {
    debugTrace.loader('Using tied embeddings as LM head');
    lmHead = ctx.embeddings;
  } else if (!lmHead) {
    log.warn('Loader', 'LM head not found');
  }

  // Downcast LM head to F16 if applicable
  if (lmHead && !isCpuWeightBuffer(lmHead)) {
    lmHead = await maybeDowncastLmHead(ctx, lmHead, lmHeadName, lmHeadLoc);
  }

  return lmHead;
}

/**
 * Process a loaded LM head tensor.
 */
function processLmHeadTensor(
  ctx: FinalWeightsContext,
  tensor: GPUBuffer | WeightBuffer | Float32Array,
  name: string,
  loc: TensorLocation,
  shouldStream: boolean
): GPUBuffer | WeightBuffer | CpuWeightBuffer | Float32Array {
  // Float32Array streaming path
  if (tensor instanceof Float32Array && shouldStream) {
    const layout = ctx.resolveWeightLayout(loc, name);
    const dtype: WeightDtype = loc.dtype === 'F16' ? 'f16' : 'f32';
    const result = createCpuWeightBuffer(tensor, dtype, layout, loc.shape, name);
    log.warn('Loader', `LM head stored on CPU for chunked matmul (layout=${layout})`);
    return result;
  }

  // Raw GPUBuffer - wrap with dtype/layout metadata
  if (tensor instanceof GPUBuffer && loc.shape && loc.shape.length === 2) {
    const layout = ctx.resolveWeightLayout(loc, name);
    const dtype: WeightDtype = loc.dtype === 'F16' ? 'f16' : 'f32';
    const wrapped = createWeightBuffer(tensor, dtype, layout, loc.shape, name);
    log.info('Loader', `Wrapped lm_head as WeightBuffer (layout=${layout}, dtype=${dtype})`);
    return wrapped;
  }

  return tensor;
}

/**
 * Attempt to downcast LM head from F32 to F16.
 */
async function maybeDowncastLmHead(
  ctx: FinalWeightsContext,
  lmHead: GPUBuffer | WeightBuffer | Float32Array,
  lmHeadName: string | null,
  lmHeadLoc: TensorLocation | undefined
): Promise<GPUBuffer | WeightBuffer | Float32Array> {
  // Check if tied to embeddings (skip downcast to avoid double-processing)
  const tiedToEmbeddings =
    lmHead === ctx.embeddings ||
    (isWeightBuffer(lmHead) && isWeightBuffer(ctx.embeddings) && lmHead.buffer === ctx.embeddings.buffer) ||
    (lmHead instanceof GPUBuffer && isWeightBuffer(ctx.embeddings) && lmHead === ctx.embeddings.buffer);

  if (tiedToEmbeddings) {
    return lmHead;
  }

  // Can't downcast Float32Array
  if (lmHead instanceof Float32Array) {
    return lmHead;
  }

  // Get current dtype
  const dtype = isWeightBuffer(lmHead)
    ? lmHead.dtype
    : (lmHeadLoc?.dtype === 'F16' ? 'f16' : 'f32');

  // Skip if not F32
  if (dtype !== 'f32') {
    return lmHead;
  }

  // Get buffer for downcast
  const buffer = isWeightBuffer(lmHead) ? lmHead.buffer : lmHead;
  if (!(buffer instanceof GPUBuffer)) {
    return lmHead;
  }

  const elems = buffer.size / 4;

  // Attempt downcast
  const result = await maybeDowncastToF16(lmHead, {
    label: lmHeadName ?? 'lm_head',
    keepF32: ctx.keepF32Weights,
    shape: isWeightBuffer(lmHead)
      ? Array.from(lmHead.shape)
      : (lmHeadLoc?.shape ?? [elems]),
    layout: isWeightBuffer(lmHead)
      ? lmHead.layout
      : (lmHeadLoc ? ctx.resolveWeightLayout(lmHeadLoc, lmHeadName ?? 'lm_head') : 'row'),
  });

  if (result?.wasDowncast && result.newBuffer) {
    ctx.gpuBuffers.add(result.newBuffer);
    return result.buffer as GPUBuffer | WeightBuffer;
  }

  return lmHead;
}
