/**
 * Layer Loader - Load transformer layer weights.
 *
 * Handles loading of all weights for a single transformer layer:
 * - Attention weights (Q, K, V, O projections)
 * - Norm weights (input, post-attention, FFN norms)
 * - FFN weights (gate, up, down projections)
 * - MoE router weights
 *
 * @module loader/layer-loader
 */

import { getKernelCapabilities } from '../gpu/device.js';
import { isWeightBuffer } from '../gpu/weight-buffer.js';
import { batchDowncastWeights } from './weight-downcast.js';
import { trace as debugTrace } from '../debug/index.js';

// ============================================================================
// Constants
// ============================================================================

/**
 * Layer name prefixes in order of preference
 * @param {number} layerIdx
 * @returns {string[]}
 */
const LAYER_PREFIXES = (layerIdx) => [
  `language_model.model.layers.${layerIdx}`,
  `model.layers.${layerIdx}`,
  `layers.${layerIdx}`,
  `blk.${layerIdx}`,
];

/** Attention weight suffixes */
const ATTN_SUFFIXES = {
  inputNorm: ['input_layernorm.weight', 'attn_norm.weight'],
  qProj: ['self_attn.q_proj.weight', 'attention.wq.weight', 'attn_q.weight'],
  kProj: ['self_attn.k_proj.weight', 'attention.wk.weight', 'attn_k.weight'],
  vProj: ['self_attn.v_proj.weight', 'attention.wv.weight', 'attn_v.weight'],
  oProj: ['self_attn.o_proj.weight', 'attention.wo.weight', 'attn_output.weight'],
  qNorm: ['self_attn.q_norm.weight', 'attn_q_norm.weight'],
  kNorm: ['self_attn.k_norm.weight', 'attn_k_norm.weight'],
  postAttentionNorm: ['post_attention_layernorm.weight', 'post_attention_norm.weight', 'ffn_norm.weight'],
  preFeedforwardNorm: ['pre_feedforward_layernorm.weight'],
  postFeedforwardNorm: ['post_feedforward_layernorm.weight', 'post_ffw_norm.weight'],
};

/** FFN weight suffixes */
const FFN_SUFFIXES = {
  ffnGateUp: ['mlp.gate_up_proj.weight', 'ffn_gate_up.weight', 'feed_forward.w1_w3.weight'],
  ffnGate: ['mlp.gate_proj.weight', 'feed_forward.w1.weight', 'ffn_gate.weight'],
  ffnUp: ['mlp.up_proj.weight', 'feed_forward.w3.weight', 'ffn_up.weight'],
  ffnDown: ['mlp.down_proj.weight', 'feed_forward.w2.weight', 'ffn_down.weight'],
};

/** MoE router weight suffixes */
const ROUTER_SUFFIXES = {
  routerWeight: ['mlp.router.weight', 'block_sparse_moe.gate.weight'],
  routerBias: ['mlp.router.bias'],
};

/** Attention sink suffixes */
const SINK_SUFFIXES = ['self_attn.sinks'];

/** Matmul weight keys for downcast
 * @type {(keyof import('./loader-types.js').LayerWeights)[]}
 */
const MATMUL_KEYS = [
  'qProj', 'kProj', 'vProj', 'oProj',
  'ffnGate', 'ffnUp', 'ffnDown', 'ffnGateUp',
];

// ============================================================================
// Main Function
// ============================================================================

/**
 * Load all weights for a single transformer layer.
 *
 * @param {import('./layer-loader.js').LayerLoaderContext} ctx - Layer loader context
 * @param {number} layerIdx - Layer index
 * @returns {Promise<import('./loader-types.js').LayerWeights>} Loaded layer weights
 */
export async function loadLayer(ctx, layerIdx) {
  const prefixes = LAYER_PREFIXES(layerIdx);

  /** @type {import('./loader-types.js').LayerWeights} */
  const weights = {
    inputNorm: null,
    qProj: null,
    kProj: null,
    vProj: null,
    oProj: null,
    qNorm: null,
    kNorm: null,
    postAttentionNorm: null,
    preFeedforwardNorm: null,
    postFeedforwardNorm: null,
    postNorm: null,
    postAttnNorm: null,
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
  weights.attentionSinks = /** @type {GPUBuffer | Float32Array | null} */ (await tryLoad(SINK_SUFFIXES));

  // Downcast matmul weights to F16
  await downcastLayerWeights(ctx, weights, layerIdx);

  return weights;
}

// ============================================================================
// Helper Factories
// ============================================================================

/**
 * Create tryLoad helper bound to context and prefixes.
 * @param {import('./layer-loader.js').LayerLoaderContext} ctx
 * @param {string[]} prefixes
 * @returns {(suffixes: string[]) => Promise<GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | Float32Array | null>}
 */
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

/**
 * Create tryLoadNorm helper that applies norm offset when needed.
 * @param {import('./layer-loader.js').LayerLoaderContext} ctx
 * @param {string[]} prefixes
 * @param {(suffixes: string[]) => Promise<GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | Float32Array | null>} tryLoad
 * @returns {(suffixes: string[]) => Promise<GPUBuffer | Float32Array | null>}
 */
function createTryLoadNorm(ctx, prefixes, tryLoad) {
  return async (suffixes) => {
    const tensor = await tryLoad(suffixes);
    if (!tensor) return null;

    // Norm weights are never WeightBuffer (they're f32 and not matmul weights)
    // Cast is safe because _loadTensor only returns WeightBuffer for matmul weights
    const normTensor = /** @type {GPUBuffer | Float32Array} */ (tensor);
    return normTensor;
  };
}

// ============================================================================
// Weight Loading Functions
// ============================================================================

/**
 * Load attention weights in parallel.
 * @param {import('./layer-loader.js').LayerLoaderContext} ctx
 * @param {import('./loader-types.js').LayerWeights} weights
 * @param {number} layerIdx
 * @param {(suffixes: string[]) => Promise<GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | Float32Array | null>} tryLoad
 * @param {(suffixes: string[]) => Promise<GPUBuffer | Float32Array | null>} tryLoadNorm
 */
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
}

/**
 * Load FFN weights in parallel.
 * @param {import('./loader-types.js').LayerWeights} weights
 * @param {number} layerIdx
 * @param {(suffixes: string[]) => Promise<GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | Float32Array | null>} tryLoad
 */
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

/**
 * Load MoE router weights.
 * @param {import('./loader-types.js').LayerWeights} weights
 * @param {(suffixes: string[]) => Promise<GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | Float32Array | null>} tryLoad
 */
async function loadRouterWeights(weights, tryLoad) {
  const [routerWeight, routerBias] = await Promise.all([
    tryLoad(ROUTER_SUFFIXES.routerWeight),
    tryLoad(ROUTER_SUFFIXES.routerBias),
  ]);
  // Router weights are not matmul weights, so they're GPUBuffer | Float32Array (not WeightBuffer)
  weights.routerWeight = /** @type {GPUBuffer | Float32Array | null} */ (routerWeight);
  weights.routerBias = /** @type {GPUBuffer | Float32Array | null} */ (routerBias);
}

// ============================================================================
// Weight Downcast
// ============================================================================

/**
 * Downcast matmul weights to F16 when supported.
 * @param {import('./layer-loader.js').LayerLoaderContext} ctx
 * @param {import('./loader-types.js').LayerWeights} weights
 * @param {number} layerIdx
 */
async function downcastLayerWeights(ctx, weights, layerIdx) {
  const caps = getKernelCapabilities();
  if (!caps.hasF16) return;

  await batchDowncastWeights(
    /** @type {Record<string, GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | null>} */ (/** @type {unknown} */ (weights)),
    /** @type {string[]} */ (MATMUL_KEYS),
    {
      keepF32: ctx.keepF32Weights,
      layerIdx,
    },
    ctx.gpuBuffers
  );
}
