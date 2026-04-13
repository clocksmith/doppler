

import { getKernelCapabilities } from '../gpu/device.js';
import {
  isWeightBuffer,
  createWeightBuffer,
  getWeightDtype,
  resolveWeightBufferMaterialization,
} from '../gpu/weight-buffer.js';
import { dequantize, dequantizeRowwise } from '../gpu/kernel-selector.js';
import { releaseBuffer, acquireBuffer, uploadData } from '../memory/buffer-pool.js';
import { batchDowncastWeights } from './weight-downcast.js';
import { QK_K, Q4K_BLOCK_BYTES } from './quantization-constants.js';
import { log, trace as debugTrace } from '../debug/index.js';
import { createTensor } from '../gpu/tensor.js';
import { castF16ToF32 } from '../gpu/kernels/cast.js';
import { dequantizeQ4KM, dequantizeQ4KMRowWise } from '../converter/quantizer.js';

// ============================================================================
// Constants
// ============================================================================


const LAYER_PREFIXES = (layerIdx) => [
  `model.language_model.layers.${layerIdx}`,
  `language_model.layers.${layerIdx}`,
  `language_model.model.layers.${layerIdx}`,
  `model.layers.${layerIdx}`,
  `layers.${layerIdx}`,
  `blk.${layerIdx}`,
];


const ATTN_SUFFIXES = {
  inputNorm: ['input_layernorm.weight', 'attn_norm.weight', 'operator_norm.weight'],
  qProj: ['self_attn.q_proj.weight', 'attention.wq.weight', 'attn_q.weight'],
  kProj: ['self_attn.k_proj.weight', 'attention.wk.weight', 'attn_k.weight'],
  vProj: ['self_attn.v_proj.weight', 'attention.wv.weight', 'attn_v.weight'],
  oProj: ['self_attn.o_proj.weight', 'self_attn.out_proj.weight', 'attention.wo.weight', 'attn_output.weight'],
  qNorm: ['self_attn.q_norm.weight', 'self_attn.q_layernorm.weight', 'attn_q_norm.weight'],
  kNorm: ['self_attn.k_norm.weight', 'self_attn.k_layernorm.weight', 'attn_k_norm.weight'],
  postAttentionNorm: ['post_attention_layernorm.weight', 'post_attention_norm.weight', 'ffn_norm.weight'],
  preFeedforwardNorm: ['pre_feedforward_layernorm.weight'],
  postFeedforwardNorm: ['post_feedforward_layernorm.weight', 'post_ffw_norm.weight'],
  postPerLayerInputNorm: ['post_per_layer_input_norm.weight'],
  layerScalar: ['layer_scalar'],
};

const LINEAR_ATTN_SUFFIXES = {
  qkvProj: ['linear_attn.in_proj_qkv.weight'],
  outProj: ['linear_attn.out_proj.weight'],
  inProjZ: ['linear_attn.in_proj_z.weight'],
  inProjA: ['linear_attn.in_proj_a.weight'],
  inProjB: ['linear_attn.in_proj_b.weight'],
  conv1D: ['linear_attn.conv1d.weight'],
  dtBias: ['linear_attn.dt_bias'],
  aLog: ['linear_attn.A_log'],
  norm: ['linear_attn.norm.weight'],
};

const CONV_SUFFIXES = {
  convInProj: ['conv.in_proj.weight', 'convolution.in_proj.weight'],
  convKernel: ['conv.conv.weight', 'convolution.conv.weight', 'conv.weight'],
  convOutProj: ['conv.out_proj.weight', 'convolution.out_proj.weight'],
};


const FFN_SUFFIXES = {
  ffnGateUp: ['mlp.gate_up_proj.weight', 'ffn_gate_up.weight', 'feed_forward.w1_w3.weight'],
  ffnGate: ['mlp.gate_proj.weight', 'feed_forward.w1.weight', 'ffn_gate.weight'],
  ffnUp: ['mlp.up_proj.weight', 'feed_forward.w3.weight', 'ffn_up.weight'],
  ffnDown: ['mlp.down_proj.weight', 'feed_forward.w2.weight', 'ffn_down.weight'],
  perLayerInputGate: ['per_layer_input_gate.weight'],
  perLayerProjection: ['per_layer_projection.weight'],
};


const ROUTER_SUFFIXES = {
  routerWeight: ['mlp.router.weight', 'block_sparse_moe.gate.weight'],
  routerBias: ['mlp.router.bias'],
};


const SINK_SUFFIXES = ['self_attn.sinks'];


const MATMUL_KEYS = [
  'qProj', 'kProj', 'vProj', 'oProj',
  'qkvProj',
  'linearInProjZ', 'linearInProjA', 'linearInProjB',
  'ffnGate', 'ffnUp', 'ffnDown', 'ffnGateUp',
  'perLayerInputGate', 'perLayerProjection',
  'convInProj', 'convOutProj',
  'routerWeight',
];

function toPositiveInt(value, label) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) {
    if (value != null && label) {
      log.debug('LayerLoader', `toPositiveInt: invalid value for ${label}: ${String(value)}`);
    }
    return null;
  }
  return Math.trunc(num);
}

function getWeightShape(value) {
  if (!isWeightBuffer(value)) return null;
  if (!Array.isArray(value.shape)) {
    log.warn('LayerLoader', `getWeightShape: expected value.shape to be an array, got ${typeof value.shape}`);
    return null;
  }
  if (value.shape.length < 2) return null;
  const dim0 = toPositiveInt(value.shape[0], 'shape[0]');
  const dim1 = toPositiveInt(value.shape[1], 'shape[1]');
  if (dim0 === null || dim1 === null) {
    return null;
  }
  return [dim0, dim1];
}

function inferLinearQKVSizes(ctx, linearQkvProj, linearOutProj) {
  const qkvShape = getWeightShape(linearQkvProj);
  if (!qkvShape) return null;

  const hiddenSize = toPositiveInt(ctx.hiddenSize, 'hiddenSize');
  if (hiddenSize === null) {
    log.warn('LayerLoader', 'inferLinearQKVSizes: hiddenSize is null; QKV size inference may be unreliable');
  }
  const total = (
    hiddenSize !== null && qkvShape[0] === hiddenSize ? qkvShape[1]
      : hiddenSize !== null && qkvShape[1] === hiddenSize ? qkvShape[0]
        : Math.max(qkvShape[0], qkvShape[1])
  );

  const linearNumKeyHeads = toPositiveInt(ctx.linearNumKeyHeads, 'linearNumKeyHeads');
  const linearNumValueHeads = toPositiveInt(ctx.linearNumValueHeads, 'linearNumValueHeads');
  const linearKeyHeadDim = toPositiveInt(ctx.linearKeyHeadDim, 'linearKeyHeadDim');
  const linearValueHeadDim = toPositiveInt(ctx.linearValueHeadDim, 'linearValueHeadDim');
  if (linearNumKeyHeads !== null && linearNumValueHeads !== null && linearKeyHeadDim !== null && linearValueHeadDim !== null) {
    const qSize = linearNumKeyHeads * linearKeyHeadDim;
    const kSize = qSize;
    const vSize = linearNumValueHeads * linearValueHeadDim;
    if ((qSize + kSize + vSize) === total) {
      return [qSize, kSize, vSize];
    }
  }

  const outShape = getWeightShape(linearOutProj);
  if (outShape) {
    const outInput = (
      hiddenSize !== null && outShape[0] === hiddenSize ? outShape[1]
        : hiddenSize !== null && outShape[1] === hiddenSize ? outShape[0]
          : Math.max(outShape[0], outShape[1])
    );
    const remainder = total - outInput;
    if (outInput > 0 && remainder > 0 && remainder % 2 === 0) {
      const qSize = remainder / 2;
      return [qSize, qSize, outInput];
    }
  }

  const numHeads = toPositiveInt(ctx.numHeads, 'numHeads');
  const numKVHeads = toPositiveInt(ctx.numKVHeads, 'numKVHeads');
  const headDim = toPositiveInt(ctx.headDim, 'headDim');
  if (numHeads !== null && numKVHeads !== null && headDim !== null) {
    const qSize = numHeads * headDim;
    const kvSize = numKVHeads * headDim;
    if ((qSize + kvSize + kvSize) <= total) {
      return [qSize, kvSize, kvSize];
    }
  }

  log.debug(
    'LayerLoader',
    `inferLinearQKVSizes: could not infer QKV split from shape [${qkvShape.join(',')}] ` +
    `(hiddenSize=${hiddenSize}, numHeads=${numHeads}, numKVHeads=${numKVHeads}, headDim=${headDim})`
  );
  return null;
}

function cloneWeightMaterializations(weight, excludeDtypes = []) {
  if (!isWeightBuffer(weight) || !weight.materializations || typeof weight.materializations !== 'object') {
    return null;
  }
  const excluded = new Set(excludeDtypes);
  const cloned = {};
  for (const [dtype, descriptor] of Object.entries(weight.materializations)) {
    if (excluded.has(dtype) || !descriptor?.buffer) {
      continue;
    }
    cloned[dtype] = {
      buffer: descriptor.buffer,
      layout: descriptor.layout ?? weight.layout,
    };
  }
  return Object.keys(cloned).length > 0 ? cloned : null;
}

function isQ4KLocationDtype(dtype) {
  return dtype === 'Q4_K_M' || dtype === 'Q4_K';
}

function toUint8Array(data) {
  if (data instanceof Uint8Array) {
    return data;
  }
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }
  return null;
}

async function loadPleReferenceQ4K(ctx, name, location, layerIdx, label) {
  if (!location || !isQ4KLocationDtype(String(location.dtype ?? '').toUpperCase())) {
    return null;
  }
  const shape = Array.isArray(location.shape) ? location.shape : null;
  if (!shape || shape.length !== 2) {
    return null;
  }

  const quantizedBytes = toUint8Array(await ctx.loadTensor(name, false, true));
  if (!quantizedBytes) {
    return null;
  }

  const rows = Number(shape[0]);
  const cols = Number(shape[1]);
  if (!Number.isFinite(rows) || rows <= 0 || !Number.isFinite(cols) || cols <= 0) {
    return null;
  }

  const f32Weights = cols % QK_K === 0
    ? dequantizeQ4KM(quantizedBytes, Math.ceil(location.size / Q4K_BLOCK_BYTES), shape)
    : dequantizeQ4KMRowWise(quantizedBytes, shape);

  const buffer = acquireBuffer(f32Weights.byteLength, undefined, `${name}_f32_reference`);
  uploadData(buffer, f32Weights);
  ctx.gpuBuffers.add(buffer);

  debugTrace.loader(
    `Layer ${layerIdx}: loaded ${label} via CPU reference q4k dequant`
  );

  return createWeightBuffer(buffer, 'f32', location.layout ?? 'row', shape, name);
}

async function loadPerLayerProjectionReferenceQ4K(ctx, name, location, layerIdx) {
  return loadPleReferenceQ4K(ctx, name, location, layerIdx, 'perLayerProjection');
}

async function loadPerLayerInputGateReferenceQ4K(ctx, name, location, layerIdx) {
  return loadPleReferenceQ4K(ctx, name, location, layerIdx, 'perLayerInputGate');
}

async function stabilizePleWeight(ctx, weights, key, layerIdx, label) {
  const originalWeight = weights[key];
  if (!isWeightBuffer(originalWeight)) {
    return;
  }

  const denseWeight = resolveWeightBufferMaterialization(originalWeight, 'f16');
  const denseDtype = String(getWeightDtype(denseWeight) ?? '').toLowerCase();
  if (denseDtype === 'f32') {
    return;
  }
  if (denseDtype !== 'f16') {
    return;
  }

  const denseShape = Array.isArray(denseWeight.shape) ? denseWeight.shape : null;
  if (!denseShape || denseShape.length !== 2) {
    return;
  }

  const f32Tensor = await castF16ToF32(
    createTensor(
      denseWeight.buffer,
      'f16',
      denseShape,
      denseWeight.label ?? `layer_${layerIdx}_per_layer_projection_f16`
    )
  );
  ctx.gpuBuffers.add(f32Tensor.buffer);

  weights[key] = createWeightBuffer(
    f32Tensor.buffer,
    'f32',
    denseWeight.layout ?? originalWeight.layout,
    denseShape,
    denseWeight.label ?? `layer_${layerIdx}_${label}_f32`,
    {
      ...(cloneWeightMaterializations(originalWeight, ['f32']) ?? {}),
      f16: {
        buffer: denseWeight.buffer,
        layout: denseWeight.layout ?? originalWeight.layout,
      },
    }
  );

  debugTrace.loader(
    `Layer ${layerIdx}: promoted ${label} to f32 for stable per-layer input execution`
  );
}

async function stabilizePerLayerProjectionWeight(ctx, weights, layerIdx) {
  await stabilizePleWeight(ctx, weights, 'perLayerProjection', layerIdx, 'per_layer_projection');
}

async function stabilizePerLayerInputGateWeight(ctx, weights, layerIdx) {
  await stabilizePleWeight(ctx, weights, 'perLayerInputGate', layerIdx, 'per_layer_input_gate');
}

// ============================================================================
// Main Function
// ============================================================================


export async function loadLayer(ctx, layerIdx) {
  const prefixes = LAYER_PREFIXES(layerIdx);

  
  const weights = {
    inputNorm: null,
    qProj: null,
    kProj: null,
    vProj: null,
    oProj: null,
    qkvProj: null,
    qkvSizes: null,
    qkvDtype: null,
    linearInProjZ: null,
    linearInProjA: null,
    linearInProjB: null,
    linearConv1D: null,
    linearDtBias: null,
    linearALog: null,
    linearNorm: null,
    qNorm: null,
    kNorm: null,
    postAttentionNorm: null,
    preFeedforwardNorm: null,
    postFeedforwardNorm: null,
    postNorm: null,
    postAttnNorm: null,
    convInProj: null,
    convKernel: null,
    convOutProj: null,
    ffnGate: null,
    ffnUp: null,
    ffnDown: null,
    ffnGateUp: null,
    perLayerInputGate: null,
    perLayerProjection: null,
    postPerLayerInputNorm: null,
    layerScalar: null,
  };

  // Create helper functions bound to this context
  const tryLoad = createTryLoad(ctx, prefixes);
  const tryLoadNorm = createTryLoadNorm(ctx, prefixes, tryLoad);

  // Load attention weights in parallel
  await loadAttentionWeights(ctx, weights, layerIdx, tryLoad, tryLoadNorm);

  // Load FFN weights (unless MoE expert layer)
  if (!ctx.isMoE || !ctx.isExpertLayer(layerIdx)) {
    await loadFfnWeights(ctx, weights, layerIdx, tryLoad, prefixes);
  }

  // Load MoE router weights
  if (ctx.isMoE && ctx.isExpertLayer(layerIdx)) {
    await loadRouterWeights(weights, tryLoad);
  }

  // Load attention sinks
  weights.attentionSinks =  (await tryLoad(SINK_SUFFIXES));

  // Downcast matmul weights to F16
  await downcastLayerWeights(ctx, weights, layerIdx);

  return weights;
}

// ============================================================================
// Helper Factories
// ============================================================================


function createTryLoad(ctx, prefixes) {
  return async (suffixes) => {
    for (const prefix of prefixes) {
      for (const suffix of suffixes) {
        const tensor = await ctx.loadTensor(`${prefix}.${suffix}`, true, true);
        const isGpuBuffer = typeof GPUBuffer !== 'undefined' && tensor instanceof GPUBuffer;
        if (tensor && (isGpuBuffer || tensor instanceof Float32Array || isWeightBuffer(tensor))) {
          return tensor;
        }
      }
    }
    return null;
  };
}


function createTryLoadNorm(ctx, prefixes, tryLoad) {
  return async (suffixes) => {
    const tensor = await tryLoad(suffixes);
    if (!tensor) return null;

    // Norm weights are never WeightBuffer (non-matmul weights)
    // Cast is safe because _loadTensor only returns WeightBuffer for matmul weights
    const normTensor =  (tensor);
    return normTensor;
  };
}

// ============================================================================
// Weight Loading Functions
// ============================================================================


async function loadAttentionWeights(ctx, weights, layerIdx, tryLoad, tryLoadNorm) {
  const tryLoadCpu = async (suffixes) => {
    for (const prefix of LAYER_PREFIXES(layerIdx)) {
      for (const suffix of suffixes) {
        const tensor = await ctx.loadTensor(`${prefix}.${suffix}`, false, true);
        if (tensor instanceof Float32Array) {
          return tensor;
        }
      }
    }
    return null;
  };

  const [
    inputNorm,
    qProj,
    kProj,
    vProj,
    oProj,
    qNorm,
    kNorm,
    postAttentionNorm,
    preFeedforwardNorm,
    postFeedforwardNorm,
    postPerLayerInputNorm,
    layerScalar,
    convInProj,
    convKernel,
    convOutProj,
    linearQkvProj,
    linearOutProj,
    linearInProjZ,
    linearInProjA,
    linearInProjB,
    linearConv1D,
    linearDtBias,
    linearALog,
    linearNorm,
  ] = await Promise.all([
    tryLoadNorm(ATTN_SUFFIXES.inputNorm),
    tryLoad(ATTN_SUFFIXES.qProj),
    tryLoad(ATTN_SUFFIXES.kProj),
    tryLoad(ATTN_SUFFIXES.vProj),
    tryLoad(ATTN_SUFFIXES.oProj),
    // Gemma 3: q_norm and k_norm use Gemma3RMSNorm with (1+weight) formula
    tryLoadNorm(ATTN_SUFFIXES.qNorm),
    tryLoadNorm(ATTN_SUFFIXES.kNorm),
    tryLoadNorm(ATTN_SUFFIXES.postAttentionNorm),
    tryLoadNorm(ATTN_SUFFIXES.preFeedforwardNorm),
    tryLoadNorm(ATTN_SUFFIXES.postFeedforwardNorm),
    tryLoadNorm(ATTN_SUFFIXES.postPerLayerInputNorm),
    tryLoadCpu(ATTN_SUFFIXES.layerScalar),
    tryLoad(CONV_SUFFIXES.convInProj),
    tryLoad(CONV_SUFFIXES.convKernel),
    tryLoad(CONV_SUFFIXES.convOutProj),
    tryLoad(LINEAR_ATTN_SUFFIXES.qkvProj),
    tryLoad(LINEAR_ATTN_SUFFIXES.outProj),
    tryLoad(LINEAR_ATTN_SUFFIXES.inProjZ),
    tryLoad(LINEAR_ATTN_SUFFIXES.inProjA),
    tryLoad(LINEAR_ATTN_SUFFIXES.inProjB),
    tryLoad(LINEAR_ATTN_SUFFIXES.conv1D),
    tryLoadNorm(LINEAR_ATTN_SUFFIXES.dtBias),
    tryLoadNorm(LINEAR_ATTN_SUFFIXES.aLog),
    tryLoadNorm(LINEAR_ATTN_SUFFIXES.norm),
  ]);

  weights.inputNorm = inputNorm;
  weights.qProj = qProj;
  weights.kProj = kProj;
  weights.vProj = vProj;
  weights.oProj = oProj;
  weights.qNorm = qNorm;
  weights.kNorm = kNorm;

  // Log q_norm/k_norm loading status for layer 0 only
  if (layerIdx === 0) {
    const hasOffset = ctx.needsNormWeightOffset();
    debugTrace.loader(
      `Layer 0 norm weights: qNorm=${qNorm ? 'found' : 'null'}, ` +
      `kNorm=${kNorm ? 'found' : 'null'}, offset=${hasOffset ? 'runtime' : 'none'}`
    );
  }

  weights.postAttentionNorm = postAttentionNorm;
  weights.preFeedforwardNorm = preFeedforwardNorm;
  weights.postFeedforwardNorm = postFeedforwardNorm;
  weights.postPerLayerInputNorm = postPerLayerInputNorm;
  weights.layerScalar = layerScalar;
  weights.postNorm = weights.postAttentionNorm || weights.preFeedforwardNorm;
  weights.postAttnNorm = weights.postNorm;
  weights.convInProj = convInProj;
  weights.convKernel = convKernel;
  weights.convOutProj = convOutProj;
  weights.linearInProjZ = linearInProjZ;
  weights.linearInProjA = linearInProjA;
  weights.linearInProjB = linearInProjB;
  weights.linearConv1D = linearConv1D;
  weights.linearDtBias = linearDtBias;
  weights.linearALog = linearALog;
  weights.linearNorm = linearNorm;

  // Qwen3.5 linear-attention layers expose fused in_proj_qkv + out_proj
  // instead of self_attn.{q,k,v}_proj. Route into shared fused-QKV path.
  const hasDenseQkv = Boolean(weights.qProj && weights.kProj && weights.vProj);
  if (!hasDenseQkv && linearQkvProj) {
    weights.qkvProj = linearQkvProj;
    if (!weights.oProj && linearOutProj) {
      weights.oProj = linearOutProj;
    }

    const inferredSizes = inferLinearQKVSizes(ctx, linearQkvProj, linearOutProj ?? weights.oProj);
    if (inferredSizes) {
      weights.qkvSizes = inferredSizes;
    }

    if (isWeightBuffer(linearQkvProj)) {
      const dtype = String(linearQkvProj.dtype ?? '').toLowerCase();
      weights.qkvDtype = dtype === 'f32' ? 'f32' : 'f16';
    }
  }
}


async function loadFfnWeights(ctx, weights, layerIdx, tryLoad, prefixes) {
  const perLayerProjection = await loadStablePerLayerProjection(ctx, layerIdx, prefixes, tryLoad);
  const stablePerLayerInputGate = await loadStablePerLayerInputGate(ctx, layerIdx, prefixes, tryLoad);
  const [ffnGateUp, ffnGate, ffnUp, ffnDown] = await Promise.all([
    tryLoad(FFN_SUFFIXES.ffnGateUp),
    tryLoad(FFN_SUFFIXES.ffnGate),
    tryLoad(FFN_SUFFIXES.ffnUp),
    tryLoad(FFN_SUFFIXES.ffnDown),
  ]);

  if (ffnGateUp) {
    weights.ffnGateUp = ffnGateUp;
    weights.ffnGate = null;
    weights.ffnUp = null;
    debugTrace.loader(`Layer ${layerIdx}: Using fused gate_up_proj for 2-pass FFN`);
  } else {
    weights.ffnGate = ffnGate;
    weights.ffnUp = ffnUp;
  }

  weights.ffnDown = ffnDown;
  weights.perLayerInputGate = stablePerLayerInputGate;
  weights.perLayerProjection = perLayerProjection;

  weights.gate = weights.ffnGate;
  weights.up = weights.ffnUp;
  weights.down = weights.ffnDown;
  weights.gateUp = weights.ffnGateUp;
}

async function loadStablePerLayerProjection(ctx, layerIdx, prefixes, tryLoad) {
  for (const prefix of prefixes) {
    const name = `${prefix}.${FFN_SUFFIXES.perLayerProjection[0]}`;
    const location = ctx.tensorLocations.get(name) ?? null;
    if (location) {
      const referenceWeight = await loadPerLayerProjectionReferenceQ4K(ctx, name, location, layerIdx);
      if (referenceWeight) {
        return referenceWeight;
      }
    }
  }
  return tryLoad(FFN_SUFFIXES.perLayerProjection);
}

async function loadStablePerLayerInputGate(ctx, layerIdx, prefixes, tryLoad) {
  for (const prefix of prefixes) {
    const name = `${prefix}.${FFN_SUFFIXES.perLayerInputGate[0]}`;
    const location = ctx.tensorLocations.get(name) ?? null;
    if (location) {
      const referenceWeight = await loadPerLayerInputGateReferenceQ4K(ctx, name, location, layerIdx);
      if (referenceWeight) {
        return referenceWeight;
      }
    }
  }
  return tryLoad(FFN_SUFFIXES.perLayerInputGate);
}


async function loadRouterWeights(weights, tryLoad) {
  const [routerWeight, routerBias] = await Promise.all([
    tryLoad(ROUTER_SUFFIXES.routerWeight),
    tryLoad(ROUTER_SUFFIXES.routerBias),
  ]);
  // Router weights follow matmul dtype/layout rules when present
  weights.routerWeight =  (routerWeight);
  weights.routerBias =  (routerBias);
}

// ============================================================================
// Weight Downcast
// ============================================================================


async function downcastLayerWeights(ctx, weights, layerIdx) {
  const caps = getKernelCapabilities();
  if (!caps.hasF16) return;

  await batchDowncastWeights(
     ( (weights)),
     (MATMUL_KEYS),
    {
      keepF32: ctx.keepF32Weights,
      layerIdx,
    },
    ctx.gpuBuffers
  );

  await dequantConvQ4KWeights(ctx, weights, layerIdx);
  await stabilizePerLayerInputGateWeight(ctx, weights, layerIdx);
  await stabilizePerLayerProjectionWeight(ctx, weights, layerIdx);
}


const CONV_Q4K_DEQUANT_KEYS = ['convInProj', 'convOutProj', 'convKernel'];

async function dequantConvQ4KWeights(ctx, weights, layerIdx) {
  for (const key of CONV_Q4K_DEQUANT_KEYS) {
    const buf = weights[key];
    if (!buf || !isWeightBuffer(buf)) continue;
    if (getWeightDtype(buf) !== 'q4k') continue;

    const shape = buf.shape;
    if (!Array.isArray(shape) || shape.length < 2) continue;

    const is2D = shape.length === 2;
    const totalElements = shape.reduce((a, b) => a * b, 1);

    let dequantizedTensor;
    const outputDtype = 'f32';
    if (is2D && shape[1] % QK_K !== 0) {
      dequantizedTensor = await dequantizeRowwise(buf.buffer, shape[0], shape[1], { outputDtype });
    } else {
      if (totalElements === 0 || totalElements % QK_K !== 0) continue;
      const numBlocks = totalElements / QK_K;
      dequantizedTensor = await dequantize(buf.buffer, numBlocks, { outputDtype });
    }

    releaseBuffer(buf.buffer);
    const dequantizedBuffer = dequantizedTensor.buffer;
    weights[key] = createWeightBuffer(dequantizedBuffer, outputDtype, 'row', shape, buf.label ?? key);
    ctx.gpuBuffers.add(dequantizedBuffer);

    debugTrace.loader(`Layer ${layerIdx} dequantized conv ${key} Q4K→${outputDtype.toUpperCase()}: [${shape.join(',')}]`);
  }
}
