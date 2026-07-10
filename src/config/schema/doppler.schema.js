import { cloneJsonValue as cloneConfigTree } from '../../utils/clone-json.js';
import { DEFAULT_LOADING_CONFIG } from './loading.schema.js';
import { DEFAULT_SHARED_RUNTIME_CONFIG } from './shared-runtime.schema.js';
import { DEFAULT_EMULATION_CONFIG, createEmulationConfig } from './emulation.schema.js';
import { DEFAULT_KVCACHE_CONFIG } from './kvcache.schema.js';
import { DEFAULT_MOE_RUNTIME_CONFIG } from './moe.schema.js';
import { DEFAULT_SPECULATIVE_CONFIG } from './speculative.schema.js';
import { DEFAULT_SELF_SPECULATION_CONFIG } from './speculation-self.schema.js';
import { mergeRuntimeValues } from '../runtime-merge.js';
import { validateRuntimeOverrides } from '../param-validator.js';
import { isPlainObject } from '../../utils/plain-object.js';
import {
  chooseDefined,
  mergeExecutionPatchLists,
  mergeKernelPathPolicy,
  mergeShallowObject,
  replaceSubtree,
} from '../merge-helpers.js';

// =============================================================================
// Runtime Config (all non-model-specific settings)
// =============================================================================

export const DEFAULT_CHAT_TEMPLATE_CONFIG = Object.freeze({
  enabled: null,
  type: null,
  thinking: false,
});

export const DEFAULT_LARGE_WEIGHT_CONFIG = Object.freeze({
  enabled: true,
  safetyRatio: 0.9,
  preferF16: true,
  lmHeadChunkRows: null,
  gpuResidentOverrides: null,
});

export const DEFAULT_RUNTIME_CONFIG = {
  shared: DEFAULT_SHARED_RUNTIME_CONFIG,
  loading: DEFAULT_LOADING_CONFIG,
  inference: {
    batching: {},
    sampling: {
      temperature: 1.0,
      topP: 0.95,
      topK: 50,
      repetitionPenalty: 1.1,
      greedyThreshold: 0.01,
      repetitionPenaltyWindow: 100,
      suppressSpecialTokens: false,
      suppressSpecialLikeTokens: false,
      suppressTokenIds: [],
    },
    compute: {},
    tokenizer: {},
    largeWeights: DEFAULT_LARGE_WEIGHT_CONFIG,
    kvcache: DEFAULT_KVCACHE_CONFIG,
    diffusion: {},
    diffusionGemma: {},
    energy: {},
    moe: DEFAULT_MOE_RUNTIME_CONFIG,
    speculative: DEFAULT_SPECULATIVE_CONFIG,
    generation: {
      maxTokens: 256,
      multimodalMaxTokens: 512,
      disableMultiTokenDecode: false,
    },
    kernelPathPolicy: mergeKernelPathPolicy(undefined, {
      mode: 'capability-aware',
      sourceScope: ['model', 'manifest', 'config'],
      onIncompatible: 'remap',
    }),
    chatTemplate: DEFAULT_CHAT_TEMPLATE_CONFIG,
    session: {
      kvcache: DEFAULT_KVCACHE_CONFIG,
      speculation: { ...DEFAULT_SELF_SPECULATION_CONFIG },
      // "sync" (default): wait for each prefill chunk to finish on GPU before
      //   recording the next. Required when profile timings are being
      //   resolved and when low-memory constraints demand deterministic
      //   buffer release per chunk.
      // "async": queue each prefill chunk without waiting — GPU queue ordering
      //   keeps commands correct and deferred cleanup releases tracked
      //   buffers when work completes. Reduces CPU-GPU roundtrips; profile
      //   paths override to sync automatically.
      prefillChunkSubmitMode: 'sync',
      // Number of transformer layers recorded per prefill command recorder
      // before a chunk boundary fires. Larger values → fewer chunk
      // transitions but longer intermediate-buffer lifetime. Default 4 is
      // memory-conservative; profiles with known-safe memory can raise it.
      prefillChunkLayers: 4,
      prefillTokenChunkSize: null,
      skipEmbeddingKVCacheWrites: false,
      // Opt into the flash-attention prefill kernel (head_dim=256, f16 KV,
      // contiguous layout). Two-pass dispatch raises RDNA3 workgroup
      // occupancy by KV-axis splitting. The kernel itself enforces head_dim
      // and layout preconditions at runtime. Default false until parity is
      // validated on target hardware.
      useFlashPrefillAttention: false,
      // Opt into the fused gate + up projection + GeGLU activation prefill
      // kernel (replaces gate_proj + up_proj + gelu activation dispatches
      // with a single fused compute pass). Requires f16 weights + gelu
      // activation + no LoRA on gate/up. Default false until parity is
      // validated on target hardware.
      useFusedGateUpGelu: false,
      // Opt into large-batch f16-weight/f32-activation fused gate/up prefill.
      // This keeps the AF32 lane while replacing split gate/up/activation
      // dispatches for explicitly profiled browser/Metal rerank profiles.
      useLargeBatchF16F32FusedGateUp: false,
      // Opt into the register-tiled Q4_K prefill matmul kernel (64x64 outputs
      // per workgroup via 4x4 register tile per thread). Amortizes Q4_K
      // dequantization across 16x the outputs of q4_fused_batched_f16a,
      // replacing ~1024x too many launched workgroups on prefill shapes.
      // Applied only to Q4_K-weight matmuls with f16 activations, f16 output,
      // hasSubgroups, M>=16. Default false until parity is validated.
      useTiledQ4KPrefill: false,
      // Force the loader to retain Q4_K packed weights alongside the
      // dequantized dense buffer ("mixed" materialization mode). Without a
      // fused projection kernel declared in the execution graph, the default
      // is "dense" which drops the Q4_K buffer after dequant — preventing the
      // FFN fusion rule from firing its `hasQ4KMaterialization=true` branch.
      // Memory cost: ~50% extra weight storage for Q4_K matmul weights.
      retainQ4KMaterialization: false,
      // Opt into the WideTile Q4_K prefill matmul (adapted from ORT
      // MatMulNBitsWideTile). Register-tiled accumulation, 1 thread per
      // output column, TILE_M rows accumulated simultaneously. Dramatically
      // fewer workgroups launched vs q4_fused_batched_f16 at prefill shapes.
      // Requires f32 activations + Q4_K weights + shader-f16 + M >= TILE_M.
      // Default false until correctness + perf parity validated on target
      // hardware.
      useWideTileQ4KPrefill: false,
      // Opt into the WideTile Q4_K decode matmul. This reuses the WideTile
      // register kernel for one-row decode matmuls when explicitly enabled.
      // Default false until correctness + perf parity validated on target
      // hardware.
      useWideTileQ4KDecode: false,
      // Opt into a two-output sandwich RMSNorm decode kernel:
      // post_attn_norm and pre_ffn_norm execute in one dispatch while still
      // materializing both tensors for the downstream residual and FFN paths.
      // Default false until correctness + perf validated.
      useSandwichRMSNormPairFusion: false,
      // Opt into a decode-only cross-layer RMSNorm pair:
      // post_ffn_norm for layer N and input_norm for layer N+1 execute in one
      // dispatch. The next layer consumes the precomputed input-norm tensor at
      // its normal observation point. Default false until correctness + perf
      // validated.
      usePostFfnNextInputRMSNormPairFusion: false,
      // Opt into a decode-only post-attention norm stats prelude consumed by
      // Q4_K fused gate/up. It materializes the pre-norm residual sum and one
      // inverse-RMS scalar instead of a full normalized FFN input tensor.
      // Default false until correctness + perf validated on target hardware.
      usePostAttnNormFusedGateUp: false,
      // Optional Q4_K fused FFN pipeline constants keyed by phase. Profiles can
      // tune the gate/up workgroup shape per platform without changing the
      // global kernel default or manifest math contract.
      fusedFfnQ4K: null,
      lmHeadArgmaxQ4K: null,
      attentionDecodeOnline: null,
      // Opt into a decode-only linear-attention A+B input projection fusion.
      // Requires dense f16 row weights for linear_attn.in_proj_a/b and falls
      // back to separate projections when the weight contract is not met.
      useLinearAttentionABProjectionFusion: false,
      // Opt into a decode-only linear-attention QKV+Z projection fusion.
      // Requires row-wise Q4_K linear_attn.in_proj_qkv/z weights with matching
      // input width and falls back to separate projections otherwise.
      useLinearAttentionQKVZProjectionFusion: false,
      // Opt into a decode-only linear-attention core fusion. Combines the
      // conv1d state update and recurrent scan when qRep=1 and head dims fit
      // one workgroup, eliminating the conv_out intermediate dispatch.
      useLinearAttentionFusedDecodeCore: false,
      // Opt into the single-pass flash attention prefill kernel adapted from
      // ORT's flash_attention.wgsl.template. 64 threads = 64 queries per WG;
      // private Q/O tiles, shared K/V tiles, online softmax, no reduce pass.
      // At Gemma 4 E2B prefill=64 this dispatches 8 WGs total (vs ~32 for the
      // split+reduce Doppler flash path). Requires head_dim=256, f16 KV,
      // contiguous KV layout, seqLen > 1. Supersedes useFlashPrefillAttention
      // when both flags are set (this path is picked first). Default false.
      useOrtFlashPrefillAttention: false,
      // Opt into the WideTile Q4_K matmul + residual fusion at the ffn_down
      // + ffn_residual call site (dense.js). Eliminates one separate residual
      // dispatch per layer. Requires f32 activations + retained Q4_K weights
      // plus an enabled WideTile phase.
      // Default false until correctness + perf validated.
      useWideTileResidualFusion: false,
      // Opt into the RMSNorm + WideTile Q4_K matmul fusion at pre-matmul
      // norm sites (input_norm→q/k/v_proj, pre_feedforward_norm→gate/up).
      // Each fused call recomputes RMS internally (redundant across q/k/v
      // but negligible) and skips the standalone rmsnorm dispatch upstream.
      // Requires f32 activations + Q4_K weights retained + prefill. Default
      // false until wiring lands and correctness validates.
      useFusedRmsnormWideTile: false,
      // Opt into fused packed-QKV split plus weighted Q/K RMSNorm. Replaces
      // split_qkv + rmsnorm_qk when diagnostics do not need raw Q/K projection
      // readbacks. Requires f32 QKV output and weighted Q/K norm.
      useFusedQKVSplitQKNorm: false,
      // Opt into fused packed-QKV split plus weighted Q/K RMSNorm and full-head
      // non-interleaved RoPE. Replaces split_qkv + rmsnorm_qk + rope_qk when
      // diagnostics do not need intermediate Q/K stage readbacks.
      useFusedQKVSplitQKNormRoPE: false,
      useGreedyLmHeadArgmaxFusion: false,
    },
    executionPatch: {},
  },
  emulation: DEFAULT_EMULATION_CONFIG,
};

// =============================================================================
// Master Doppler Config
// =============================================================================

export const DEFAULT_DOPPLER_CONFIG = {
  model: undefined,
  runtime: DEFAULT_RUNTIME_CONFIG,
};

// =============================================================================
// Factory Function
// =============================================================================

export function createDopplerConfig(
  overrides
) {
  const runtimeBase = cloneConfigTree(DEFAULT_RUNTIME_CONFIG);

  if (!overrides) {
    return {
      model: DEFAULT_DOPPLER_CONFIG.model,
      runtime: runtimeBase,
    };
  }

  const runtimeOverrides = overrides.runtime ?? {};
  const runtime = overrides.runtime
    ? createRuntimeConfig(runtimeBase, runtimeOverrides)
    : runtimeBase;
  const config = {
    model: overrides.model ?? DEFAULT_DOPPLER_CONFIG.model,
    runtime,
  };
  return config;
}

function createRuntimeConfig(base, overrides) {
  validateRuntimeOverrides(overrides);
  const runtime = mergeRuntimeValues(base, overrides);
  normalizeRuntimeContracts(runtime, base, overrides);
  return runtime;
}

function normalizeRuntimeContracts(runtime, base, overrides) {
  if (isPlainObject(overrides.inference)) {
    normalizeInferenceContracts(runtime.inference, base.inference, overrides.inference);
  }
  if (isPlainObject(overrides.emulation)) {
    runtime.emulation = createEmulationConfig(overrides.emulation);
  }
}

function normalizeInferenceContracts(inference, base, overrides) {
  if (hasOwn(overrides, 'kernelPath')) {
    inference.kernelPath = mergeRuntimeKernelPath(base.kernelPath, overrides.kernelPath);
  }
  if (hasOwn(overrides, 'kernelPathPolicy')) {
    inference.kernelPathPolicy = mergeKernelPathPolicy(
      base.kernelPathPolicy ?? {},
      overrides.kernelPathPolicy ?? {}
    );
  }
  if (hasOwn(overrides, 'chatTemplate')) {
    inference.chatTemplate = mergeShallowObject(base.chatTemplate, overrides.chatTemplate);
  }
  if (hasOwn(overrides, 'executionPatch')) {
    inference.executionPatch = mergeExecutionPatchLists(
      base.executionPatch ?? {},
      overrides.executionPatch ?? {}
    );
  }
  if (hasOwn(overrides, 'modelOverrides')) {
    inference.modelOverrides = chooseDefined(overrides.modelOverrides, base.modelOverrides);
  }
  normalizeSessionContracts(inference, base, overrides);
}

function normalizeSessionContracts(inference, base, overrides) {
  if (!isPlainObject(overrides.session) || !isPlainObject(inference.session)) {
    return;
  }

  const baseSession = base.session ?? {};
  const overrideSession = overrides.session;
  if (hasOwn(overrideSession, 'kvcache')) {
    inference.session.kvcache = replaceSubtree(overrideSession.kvcache, baseSession.kvcache);
  }
  if (hasOwn(overrideSession, 'decodeLoop')) {
    inference.session.decodeLoop = replaceSubtree(overrideSession.decodeLoop, baseSession.decodeLoop);
  }
  if (hasOwn(overrideSession, 'perLayerInputs')) {
    inference.session.perLayerInputs = replaceSubtree(
      overrideSession.perLayerInputs,
      baseSession.perLayerInputs
    );
  }
  if (hasOwn(overrideSession, 'speculation')) {
    inference.session.speculation = replaceSubtree(
      overrideSession.speculation,
      baseSession.speculation
    );
  }

  const overrideSessionCompute = overrideSession.compute;
  if (isPlainObject(overrideSessionCompute)) {
    const compute = isPlainObject(inference.session.compute) ? inference.session.compute : {};
    const nextCompute = { ...compute };
    if (hasOwn(overrideSessionCompute, 'defaults')) {
      nextCompute.defaults = replaceSubtree(
        overrideSessionCompute.defaults,
        baseSession.compute?.defaults
      );
    }
    if (hasOwn(overrideSessionCompute, 'kernelProfiles')) {
      nextCompute.kernelProfiles = overrideSessionCompute.kernelProfiles;
    }
    inference.session.compute = nextCompute;
  }
}

function hasOwn(container, key) {
  return Object.prototype.hasOwnProperty.call(container, key);
}

function mergeRuntimeKernelPath(
  baseKernelPath,
  overrideKernelPath
) {
  const kernelPath = chooseDefined(overrideKernelPath, baseKernelPath);
  assertRuntimeKernelPath(kernelPath);
  return kernelPath;
}

function assertRuntimeKernelPath(kernelPath) {
  if (kernelPath === undefined || kernelPath === null) {
    return;
  }
  if (typeof kernelPath === 'string') {
    throw new Error(
      'DopplerConfigError: runtime.inference.kernelPath no longer accepts string registry IDs. ' +
      'Use an inline kernel path object generated from execution-v1, or leave kernelPath null.'
    );
  }
  if (typeof kernelPath !== 'object' || Array.isArray(kernelPath)) {
    throw new Error(
      'DopplerConfigError: runtime.inference.kernelPath must be an inline kernel path object or null.'
    );
  }
}
