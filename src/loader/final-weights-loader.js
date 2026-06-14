

import {
  createWeightBuffer,
  createCpuWeightBuffer,
  isWeightBuffer,
  isCpuWeightBuffer,
  isSplitWeightBuffer,
  getWeightDtype,
} from '../gpu/weight-buffer.js';
import { maybeDowncastToF16 } from './weight-downcast.js';
import { getTensorNamesByRole } from './tensors/tensor-role.js';
import { log, trace as debugTrace } from '../debug/index.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import { loadTensorRange } from './tensors/tensor-reader.js';

// ============================================================================
// Constants
// ============================================================================


const HEAD_GROUP = 'head';
const FINAL_NORM_ROLE = 'norm';
const LM_HEAD_ROLE = 'lm_head';
const EMBEDDING_MODEL_TYPE = 'embedding';
const DENSE_TIED_LM_HEAD_DTYPES = new Set(['F16', 'F32']);
const DIFFUSION_GEMMA_SELF_CONDITIONING_PREFIXES = [
  'model.decoder.self_conditioning',
  'decoder.self_conditioning',
  'self_conditioning',
];

function isGpuBufferInstance(value) {
  return typeof GPUBuffer !== 'undefined' && value instanceof GPUBuffer;
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

function normalizeLocationDtype(location) {
  return typeof location?.dtype === 'string'
    ? location.dtype.trim().toUpperCase()
    : '';
}

function isDenseTiedLmHeadLocation(location) {
  if (!location || !DENSE_TIED_LM_HEAD_DTYPES.has(normalizeLocationDtype(location))) {
    return false;
  }
  if (location.sourceTransform || location.storage) {
    return false;
  }
  return true;
}

function getLoadedWeightShape(weight) {
  if (isWeightBuffer(weight) || isCpuWeightBuffer(weight) || isSplitWeightBuffer(weight)) {
    return weight.shape;
  }
  return null;
}

function getLoadedWeightLayout(weight) {
  if (isWeightBuffer(weight) || isCpuWeightBuffer(weight) || isSplitWeightBuffer(weight)) {
    return weight.layout;
  }
  return null;
}

function shapesEqual(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
    return false;
  }
  for (let i = 0; i < left.length; i++) {
    if (left[i] !== right[i]) {
      return false;
    }
  }
  return true;
}

function shouldUseTiedEmbeddingsForLmHead(ctx, loc) {
  if (ctx.tieWordEmbeddings !== true || !ctx.embeddings) {
    return false;
  }
  if (ctx.embeddings instanceof Float32Array || isCpuWeightBuffer(ctx.embeddings)) {
    return false;
  }
  if (!isDenseTiedLmHeadLocation(loc)) {
    return false;
  }

  const embeddingShape = getLoadedWeightShape(ctx.embeddings);
  if (!shapesEqual(loc.shape, embeddingShape)) {
    return false;
  }

  const lmHeadDtype = selectRuleValue('loader', 'weights', 'floatLocationDtype', {
    locationDtype: loc.dtype,
  });
  const embeddingDtype = getWeightDtype(ctx.embeddings);
  if (embeddingDtype !== lmHeadDtype) {
    return false;
  }

  const lmHeadLayout = ctx.resolveWeightLayout(loc);
  const embeddingLayout = getLoadedWeightLayout(ctx.embeddings);
  return embeddingLayout === lmHeadLayout;
}

function isLikelyFinalNormName(name) {
  const lower = String(name || '').toLowerCase();
  if (!lower) return false;
  if (/layers?[._]\d+/.test(lower)) return false;
  if (lower.includes('input_layernorm')) return false;
  if (lower.includes('post_attention_layernorm')) return false;
  if (lower.includes('pre_feedforward_layernorm')) return false;
  if (lower.includes('post_feedforward_layernorm')) return false;

  return (
    lower === 'norm.weight' ||
    lower === 'model.decoder.norm.weight' ||
    lower === 'model.encoder.language_model.norm.weight' ||
    lower.includes('model.norm.weight') ||
    lower.includes('language_model.norm.weight') ||
    lower.includes('model.language_model.norm.weight') ||
    lower.includes('embedding_norm.weight') ||
    lower.includes('model.embedding_norm.weight') ||
    lower.includes('final_layernorm.weight') ||
    lower.includes('final_layer_norm.weight') ||
    lower.includes('norm_f.weight')
  );
}

async function loadFirstExistingTensor(ctx, prefixes, suffixes, options = {}) {
  for (const prefix of prefixes) {
    for (const suffix of suffixes) {
      const tensor = await ctx.loadTensor(`${prefix}.${suffix}`, options.toGPU !== false, true);
      if (tensor) return tensor;
    }
  }
  return null;
}

async function loadDiffusionGemmaSelfConditioning(ctx) {
  const preNorm = await loadFirstExistingTensor(
    ctx,
    DIFFUSION_GEMMA_SELF_CONDITIONING_PREFIXES,
    ['pre_norm.weight']
  );
  const gateProj = await loadFirstExistingTensor(
    ctx,
    DIFFUSION_GEMMA_SELF_CONDITIONING_PREFIXES,
    ['gate_proj.weight']
  );
  const upProj = await loadFirstExistingTensor(
    ctx,
    DIFFUSION_GEMMA_SELF_CONDITIONING_PREFIXES,
    ['up_proj.weight']
  );
  const downProj = await loadFirstExistingTensor(
    ctx,
    DIFFUSION_GEMMA_SELF_CONDITIONING_PREFIXES,
    ['down_proj.weight']
  );
  const postNorm = await loadFirstExistingTensor(
    ctx,
    DIFFUSION_GEMMA_SELF_CONDITIONING_PREFIXES,
    ['post_norm.weight']
  );

  const anyPresent = preNorm || gateProj || upProj || downProj || postNorm;
  if (!anyPresent) {
    return null;
  }
  const missing = [];
  if (!preNorm) missing.push('pre_norm.weight');
  if (!gateProj) missing.push('gate_proj.weight');
  if (!upProj) missing.push('up_proj.weight');
  if (!downProj) missing.push('down_proj.weight');
  if (missing.length > 0) {
    throw new Error(
      `[Loader] DiffusionGemma self-conditioning weights are incomplete. Missing: ${missing.join(', ')}.`
    );
  }

  return {
    preNorm,
    postNorm,
    gateProj,
    upProj,
    downProj,
  };
}

// ============================================================================
// Main Function
// ============================================================================


export async function loadFinalWeights(ctx) {
  let normOffsetDebugLogged = ctx.normOffsetDebugLogged;

  // Load final norm
  const { finalNorm, debugLogged: normDebugLogged } = await loadFinalNorm(ctx);
  if (normDebugLogged) {
    normOffsetDebugLogged = true;
  }

  // Load LM head
  const lmHead = await loadLmHead(ctx);
  const embeddingPostprocessor = await loadEmbeddingPostprocessor(ctx);
  const diffusionGemmaSelfConditioning = await loadDiffusionGemmaSelfConditioning(ctx);

  return {
    finalNorm,
    lmHead,
    embeddingPostprocessor,
    diffusionGemmaSelfConditioning,
    normOffsetDebugLogged,
  };
}

// ============================================================================
// Final Norm Loading
// ============================================================================


async function loadFinalNorm(ctx) {
  
  let finalNorm = null;
  let debugLogged = false;

  let finalNormNames = getTensorNamesByRole(ctx.tensorLocations, FINAL_NORM_ROLE, HEAD_GROUP);
  if (finalNormNames.length === 0) {
    const legacyCandidates = getTensorNamesByRole(ctx.tensorLocations, FINAL_NORM_ROLE).filter(
      (name) => isLikelyFinalNormName(name)
    );
    if (legacyCandidates.length > 0) {
      finalNormNames = legacyCandidates;
      log.info(
        'Loader',
        '[FinalNorm] Falling back to role-only final norm selection because tensor groups are missing in manifest.'
      );
    }
  }
  if (finalNormNames.length === 0) {
    throw new Error(
      `[Loader] Final norm not found. Expected tensor with role="${FINAL_NORM_ROLE}" and group="${HEAD_GROUP}".`
    );
  }

  for (const name of finalNormNames) {
    const location = ctx.tensorLocations.get(name);
    if (location) {
      finalNorm =  (await ctx.loadTensor(name, true, true));
      break;
    }
  }

  if (finalNorm && ctx.needsNormWeightOffset() && !ctx.normOffsetDebugLogged) {
    debugTrace.loader('Final norm uses RMSNorm weight offset (applied at runtime)');
    debugLogged = true;
  }

  if (!finalNorm) {
    throw new Error(
      `[Loader] Final norm not found. Tried: ${finalNormNames.join(', ')}`
    );
  }

  return { finalNorm, debugLogged };
}

// ============================================================================
// LM Head Loading
// ============================================================================


async function loadLmHead(ctx) {
  
  let lmHead = null;
  
  let lmHeadName = null;
  
  let lmHeadLoc;
  const modelType = typeof ctx.modelType === 'string'
    ? ctx.modelType.trim().toLowerCase()
    : '';
  const allowMissingLmHead = modelType === EMBEDDING_MODEL_TYPE;

  const lmHeadNames = getTensorNamesByRole(ctx.tensorLocations, LM_HEAD_ROLE, HEAD_GROUP);
  if (lmHeadNames.length === 0 && allowMissingLmHead) {
    debugTrace.loader('Embedding model has no LM head tensor; skipping LM head load.');
    return null;
  }
  if (lmHeadNames.length === 0 && !ctx.tieWordEmbeddings) {
    throw new Error(
      `[Loader] LM head not found. Expected tensor with role="${LM_HEAD_ROLE}" and group="${HEAD_GROUP}".`
    );
  }

  for (const name of lmHeadNames) {
    const loc = ctx.tensorLocations.get(name);
    if (!loc) continue;

    if (shouldUseTiedEmbeddingsForLmHead(ctx, loc)) {
      debugTrace.loader(`Using tied embeddings as dense LM head "${name}" (manifest.tieWordEmbeddings=true)`);
      lmHeadName = name;
      lmHeadLoc = loc;
      lmHead = ctx.embeddings;
      break;
    }

    const shouldStream = ctx.shouldStreamLargeWeight(name, loc, 'LM head');
    const tensor = shouldStream
      ? (
        createRangeBackedWeightBuffer(ctx, name, loc)
        ?? await ctx.loadTensor(name, false, true)
      )
      : await ctx.loadTensor(name, true, true);
    const tensorShouldStream = shouldStream;

    if (tensorShouldStream && tensor && !(tensor instanceof Float32Array) && !isCpuWeightBuffer(tensor)) {
      throw new Error(
        `[Loader] LM head "${name}" too large for GPU and cannot be loaded on CPU (dtype=${loc.dtype}).`
      );
    }

    if (tensor && (isGpuBufferInstance(tensor) || isWeightBuffer(tensor) || isCpuWeightBuffer(tensor) || isSplitWeightBuffer(tensor) || tensor instanceof Float32Array)) {
      lmHeadName = name;
      lmHeadLoc = loc;
      lmHead = processLmHeadTensor(ctx, tensor, name, loc, tensorShouldStream);
      break;
    }
  }

  // Use tied embeddings as fallback
  if (!lmHead && ctx.embeddings && ctx.tieWordEmbeddings) {
    debugTrace.loader('Using tied embeddings as LM head (manifest.tieWordEmbeddings=true)');
    lmHead = ctx.embeddings;
  } else if (!lmHead && allowMissingLmHead) {
    debugTrace.loader('Embedding model completed without LM head tensor.');
    return null;
  } else if (!lmHead) {
    throw new Error(
      `[Loader] LM head not found. Tried: ${lmHeadNames.join(', ')}`
    );
  }

  // Downcast LM head to F16 if applicable
  if (lmHead && !isCpuWeightBuffer(lmHead)) {
    lmHead = await maybeDowncastLmHead(ctx, lmHead, lmHeadName, lmHeadLoc);
  }

  return lmHead;
}

async function loadEmbeddingPostprocessor(ctx) {
  const config = ctx.embeddingPostprocessor;
  if (config == null) {
    return null;
  }

  const projections = [];
  for (const projection of config.projections) {
    const weightTensor = String(projection?.weightTensor || '').trim();
    if (!weightTensor) {
      throw new Error('[Loader] Embedding postprocessor projection is missing weightTensor.');
    }
    const weight = await loadCpuFloatTensor(
      ctx,
      weightTensor,
      `embedding postprocessor weight "${weightTensor}"`
    );
    const biasTensor = projection?.biasTensor == null
      ? null
      : String(projection.biasTensor).trim();
    const bias = biasTensor
      ? await loadCpuFloatTensor(
        ctx,
        biasTensor,
        `embedding postprocessor bias "${biasTensor}"`
      )
      : null;
    projections.push({
      weightTensor,
      biasTensor: biasTensor || null,
      inputSize: projection.inputSize,
      outputSize: projection.outputSize,
      activation: projection.activation,
      weight,
      bias,
    });
  }

  return {
    poolingMode: config.poolingMode,
    includePrompt: config.includePrompt,
    projections,
    normalize: config.normalize,
  };
}

async function loadCpuFloatTensor(ctx, tensorName, label) {
  const tensor = await ctx.loadTensor(tensorName, false, true);
  if (tensor == null) {
    throw new Error(`[Loader] Missing ${label}.`);
  }
  if (tensor instanceof Float32Array) {
    return tensor;
  }
  throw new Error(`[Loader] ${label} must decode to Float32Array on CPU.`);
}


function processLmHeadTensor(ctx, tensor, name, loc, shouldStream) {
  if (isSplitWeightBuffer(tensor)) {
    return tensor;
  }

  if (isCpuWeightBuffer(tensor)) {
    log.warn('Loader', `LM head stored on CPU via range-backed source (layout=${tensor.layout})`);
    return tensor;
  }

  // Float32Array streaming path
  if (tensor instanceof Float32Array && shouldStream) {
    const layout = ctx.resolveWeightLayout(loc);
    
    const dtype = selectRuleValue('loader', 'weights', 'floatLocationDtype', {
      locationDtype: loc.dtype,
    });
    const result = createCpuWeightBuffer(tensor, dtype, layout, loc.shape, name);
    log.warn('Loader', `LM head stored on CPU for chunked matmul (layout=${layout})`);
    return result;
  }

  // Raw GPUBuffer - wrap with dtype/layout metadata
  if (isGpuBufferInstance(tensor) && loc.shape && loc.shape.length === 2) {
    const layout = ctx.resolveWeightLayout(loc);
    
    const dtype = selectRuleValue('loader', 'weights', 'floatLocationDtype', {
      locationDtype: loc.dtype,
    });
    const wrapped = createWeightBuffer(tensor, dtype, layout, loc.shape, name);
    log.info('Loader', `Wrapped lm_head as WeightBuffer (layout=${layout}, dtype=${dtype})`);
    return wrapped;
  }

  return tensor;
}


async function maybeDowncastLmHead(ctx, lmHead, lmHeadName, lmHeadLoc) {
  // Check if tied to embeddings (skip downcast to avoid double-processing)
  const tiedToEmbeddings =
    lmHead === ctx.embeddings ||
    (isWeightBuffer(lmHead) && isWeightBuffer(ctx.embeddings) && lmHead.buffer === ctx.embeddings.buffer) ||
    (isGpuBufferInstance(lmHead) && isWeightBuffer(ctx.embeddings) && lmHead === ctx.embeddings.buffer);

  if (tiedToEmbeddings) {
    return lmHead;
  }

  // Can't downcast Float32Array or CpuWeightBuffer
  if (lmHead instanceof Float32Array || isCpuWeightBuffer(lmHead) || isSplitWeightBuffer(lmHead)) {
    return lmHead;
  }

  // Get current dtype
  const dtype = isWeightBuffer(lmHead)
    ? lmHead.dtype
    : selectRuleValue('loader', 'weights', 'floatLocationDtype', {
      locationDtype: lmHeadLoc?.dtype,
    });

  // Skip if not F32
  if (dtype !== 'f32') {
    return lmHead;
  }

  // Get buffer for downcast
  const buffer = isWeightBuffer(lmHead) ? lmHead.buffer : lmHead;
  if (!isGpuBufferInstance(buffer)) {
    return lmHead;
  }

  const elems = buffer.size / 4;

  // Attempt downcast
  const result = await maybeDowncastToF16(lmHead, {
    label: lmHeadName ?? 'lm_head',
    keepF32: ctx.keepF32Weights,
    dtype,
    shape: isWeightBuffer(lmHead)
      ? Array.from(lmHead.shape)
      : (lmHeadLoc?.shape ?? [elems]),
    layout: isWeightBuffer(lmHead)
      ? lmHead.layout
      : (lmHeadLoc ? ctx.resolveWeightLayout(lmHeadLoc) : 'row'),
  });

  if (result?.wasDowncast && result.newBuffer) {
    ctx.gpuBuffers.add(result.newBuffer);
    return  (result.buffer);
  }

  return lmHead;
}
