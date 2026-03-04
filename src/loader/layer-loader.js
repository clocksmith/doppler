

import { getKernelCapabilities } from '../gpu/device.js';
import { isWeightBuffer } from '../gpu/weight-buffer.js';
import { batchDowncastWeights } from './weight-downcast.js';
import { trace as debugTrace } from '../debug/index.js';

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
  qNorm: ['self_attn.q_norm.weight', 'attn_q_norm.weight'],
  kNorm: ['self_attn.k_norm.weight', 'attn_k_norm.weight'],
  postAttentionNorm: ['post_attention_layernorm.weight', 'post_attention_norm.weight', 'ffn_norm.weight'],
  preFeedforwardNorm: ['pre_feedforward_layernorm.weight'],
  postFeedforwardNorm: ['post_feedforward_layernorm.weight', 'post_ffw_norm.weight'],
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
  'convInProj', 'convOutProj',
  'routerWeight',
];

function toPositiveInt(value) {
  const num = Number(value);
  if (!Number.isFinite(num) || num <= 0) return null;
  return Math.trunc(num);
}

function getWeightShape(value) {
  if (isWeightBuffer(value) && Array.isArray(value.shape) && value.shape.length >= 2) {
    const dim0 = toPositiveInt(value.shape[0]);
    const dim1 = toPositiveInt(value.shape[1]);
    if (dim0 && dim1) {
      return [dim0, dim1];
    }
  }
  return null;
}

function inferLinearQKVSizes(ctx, linearQkvProj, linearOutProj) {
  const qkvShape = getWeightShape(linearQkvProj);
  if (!qkvShape) return null;

  const hiddenSize = toPositiveInt(ctx.hiddenSize);
  const total = (
    hiddenSize && qkvShape[0] === hiddenSize ? qkvShape[1]
      : hiddenSize && qkvShape[1] === hiddenSize ? qkvShape[0]
        : Math.max(qkvShape[0], qkvShape[1])
  );

  const linearNumKeyHeads = toPositiveInt(ctx.linearNumKeyHeads);
  const linearNumValueHeads = toPositiveInt(ctx.linearNumValueHeads);
  const linearKeyHeadDim = toPositiveInt(ctx.linearKeyHeadDim);
  const linearValueHeadDim = toPositiveInt(ctx.linearValueHeadDim);
  if (linearNumKeyHeads && linearNumValueHeads && linearKeyHeadDim && linearValueHeadDim) {
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
      hiddenSize && outShape[0] === hiddenSize ? outShape[1]
        : hiddenSize && outShape[1] === hiddenSize ? outShape[0]
          : Math.max(outShape[0], outShape[1])
    );
    const remainder = total - outInput;
    if (outInput > 0 && remainder > 0 && remainder % 2 === 0) {
      const qSize = remainder / 2;
      return [qSize, qSize, outInput];
    }
  }

  const numHeads = toPositiveInt(ctx.numHeads);
  const numKVHeads = toPositiveInt(ctx.numKVHeads);
  const headDim = toPositiveInt(ctx.headDim);
  if (numHeads && numKVHeads && headDim) {
    const qSize = numHeads * headDim;
    const kvSize = numKVHeads * headDim;
    if ((qSize + kvSize + kvSize) <= total) {
      return [qSize, kvSize, kvSize];
    }
  }

  return null;
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
  };

  // Create helper functions bound to this context
  const tryLoad = createTryLoad(ctx, prefixes);
  const tryLoadNorm = createTryLoadNorm(ctx, prefixes, tryLoad);

  // Load attention weights in parallel
  await loadAttentionWeights(ctx, weights, layerIdx, tryLoad, tryLoadNorm);

  // Load FFN weights (unless MoE expert layer)
  if (!ctx.isMoE || !ctx.isExpertLayer(layerIdx)) {
    await loadFfnWeights(weights, layerIdx, tryLoad);
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
        if (tensor && (tensor instanceof GPUBuffer || tensor instanceof Float32Array || isWeightBuffer(tensor))) {
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


async function loadFfnWeights(weights, layerIdx, tryLoad) {
  const [ffnGateUp, ffnGate, ffnUp, ffnDown] = await Promise.all([
    tryLoad(FFN_SUFFIXES.ffnGateUp),
    tryLoad(FFN_SUFFIXES.ffnGate),
    tryLoad(FFN_SUFFIXES.ffnUp),
    tryLoad(FFN_SUFFIXES.ffnDown),
  ]);

  if (ffnGateUp) {
    // Fused path: no separate gate/up weights
    weights.ffnGateUp = ffnGateUp;
    weights.ffnGate = null;
    weights.ffnUp = null;
    debugTrace.loader(`Layer ${layerIdx}: Using fused gate_up_proj for 2-pass FFN`);
  } else {
    // Separate path: use gate and up individually (3-pass FFN)
    weights.ffnGate = ffnGate;
    weights.ffnUp = ffnUp;
  }

  weights.ffnDown = ffnDown;

  // Set aliases for pipeline compatibility
  weights.gate = weights.ffnGate;
  weights.up = weights.ffnUp;
  weights.down = weights.ffnDown;
  weights.gateUp = weights.ffnGateUp;
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
}
