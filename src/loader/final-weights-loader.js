

import {
  createWeightBuffer,
  createCpuWeightBuffer,
  isWeightBuffer,
  isCpuWeightBuffer,
} from '../gpu/weight-buffer.js';
import { maybeDowncastToF16 } from './weight-downcast.js';
import { getTensorNamesByRole } from './tensor-role.js';
import { log, trace as debugTrace } from '../debug/index.js';
import { selectRuleValue } from '../rules/rule-registry.js';

// ============================================================================
// Constants
// ============================================================================


const HEAD_GROUP = 'head';
const FINAL_NORM_ROLE = 'norm';
const LM_HEAD_ROLE = 'lm_head';

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

  return {
    finalNorm,
    lmHead,
    normOffsetDebugLogged,
  };
}

// ============================================================================
// Final Norm Loading
// ============================================================================


async function loadFinalNorm(ctx) {
  
  let finalNorm = null;
  let debugLogged = false;

  const finalNormNames = getTensorNamesByRole(ctx.tensorLocations, FINAL_NORM_ROLE, HEAD_GROUP);
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

  const lmHeadNames = getTensorNamesByRole(ctx.tensorLocations, LM_HEAD_ROLE, HEAD_GROUP);
  if (lmHeadNames.length === 0 && !ctx.tieWordEmbeddings) {
    throw new Error(
      `[Loader] LM head not found. Expected tensor with role="${LM_HEAD_ROLE}" and group="${HEAD_GROUP}".`
    );
  }

  for (const name of lmHeadNames) {
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
  if (!lmHead && ctx.embeddings && ctx.tieWordEmbeddings) {
    debugTrace.loader('Using tied embeddings as LM head (manifest.tieWordEmbeddings=true)');
    lmHead = ctx.embeddings;
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


function processLmHeadTensor(ctx, tensor, name, loc, shouldStream) {
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
  if (tensor instanceof GPUBuffer && loc.shape && loc.shape.length === 2) {
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
    : selectRuleValue('loader', 'weights', 'floatLocationDtype', {
      locationDtype: lmHeadLoc?.dtype,
    });

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
      : (lmHeadLoc ? ctx.resolveWeightLayout(lmHeadLoc) : 'row'),
  });

  if (result?.wasDowncast && result.newBuffer) {
    ctx.gpuBuffers.add(result.newBuffer);
    return  (result.buffer);
  }

  return lmHead;
}
