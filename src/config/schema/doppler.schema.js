import { cloneJsonValue as cloneConfigTree } from '../../utils/clone-json.js';
import { DEFAULT_LOADING_CONFIG } from './loading.schema.js';
import { DEFAULT_SHARED_RUNTIME_CONFIG } from './shared-runtime.schema.js';
import { DEFAULT_EMULATION_CONFIG, createEmulationConfig } from './emulation.schema.js';
import { DEFAULT_KVCACHE_CONFIG } from './kvcache.schema.js';
import { DEFAULT_MOE_RUNTIME_CONFIG } from './moe.schema.js';
import { DEFAULT_SPECULATIVE_CONFIG } from './speculative.schema.js';
import { mergeEcosystemConfig } from './ecosystem.schema.js';
import {
  chooseNullish,
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
    },
    compute: {},
    tokenizer: {},
    largeWeights: DEFAULT_LARGE_WEIGHT_CONFIG,
    kvcache: DEFAULT_KVCACHE_CONFIG,
    diffusion: {},
    energy: {},
    moe: DEFAULT_MOE_RUNTIME_CONFIG,
    speculative: DEFAULT_SPECULATIVE_CONFIG,
    generation: {
      disableMultiTokenDecode: false,
    },
    chatTemplate: DEFAULT_CHAT_TEMPLATE_CONFIG,
    session: {
      kvcache: DEFAULT_KVCACHE_CONFIG,
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
      // dispatch per layer (~0.85 ms bubble × 35 layers ≈ 30 ms prefill
      // savings). Requires f32 activations + Q4_K weights retained + prefill.
      // Default false until correctness + perf validated.
      useWideTileResidualFusion: false,
      // Opt into the RMSNorm + WideTile Q4_K matmul fusion at pre-matmul
      // norm sites (input_norm→q/k/v_proj, pre_feedforward_norm→gate/up).
      // Each fused call recomputes RMS internally (redundant across q/k/v
      // but negligible) and skips the standalone rmsnorm dispatch upstream.
      // Saves ~2 dispatches/layer × 35 layers = 70 dispatches × ~0.85 ms
      // bubble ≈ 60 ms prefill savings. Requires f32 activations + Q4_K
      // weights retained + prefill. Default false until wiring lands and
      // correctness validates.
      useFusedRmsnormWideTile: false,
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
  if (!overrides) {
    return {
      model: DEFAULT_DOPPLER_CONFIG.model,
      runtime: cloneConfigTree(DEFAULT_RUNTIME_CONFIG),
    };
  }

  const runtimeOverrides = overrides.runtime ?? {};
  const runtimeBase = cloneConfigTree(DEFAULT_RUNTIME_CONFIG);
  const runtime = overrides.runtime
    ? mergeRuntimeConfig(runtimeBase, runtimeOverrides)
    : runtimeBase;
  const config = {
    model: overrides.model ?? DEFAULT_DOPPLER_CONFIG.model,
    runtime,
  };
  return config;
}

function mergeRuntimeConfig(
  base,
  overrides
) {
  return {
    shared: overrides.shared
      ? mergeSharedRuntimeConfig(base.shared, overrides.shared)
      : { ...base.shared },
    loading: overrides.loading
      ? mergeLoadingConfig(base.loading, overrides.loading)
      : { ...base.loading },
    inference: overrides.inference
      ? mergeInferenceConfig(base.inference, overrides.inference)
      : { ...base.inference },
    emulation: overrides.emulation
      ? mergeEmulationConfig(base.emulation, overrides.emulation)
      : { ...base.emulation },
  };
}

function mergeSharedRuntimeConfig(
  base,
  overrides
) {
  return {
    debug: overrides.debug
      ? mergeDebugConfig(base.debug, overrides.debug)
      : { ...base.debug },
    benchmark: overrides.benchmark
      ? mergeBenchmarkConfig(base.benchmark, overrides.benchmark)
      : { ...base.benchmark },
    harness: overrides.harness
      ? { ...base.harness, ...overrides.harness }
      : { ...base.harness },
    tooling: overrides.tooling
      ? { ...base.tooling, ...overrides.tooling }
      : { ...base.tooling },
    ecosystem: overrides.ecosystem
      ? mergeEcosystemConfig(base.ecosystem, overrides.ecosystem)
      : mergeEcosystemConfig(base.ecosystem, {}),
    platform: overrides.platform ?? base.platform,
    kernelRegistry: { ...base.kernelRegistry, ...overrides.kernelRegistry },
    kernelThresholds: overrides.kernelThresholds
      ? mergeKernelThresholds(base.kernelThresholds, overrides.kernelThresholds)
      : { ...base.kernelThresholds },
    kernelWarmup: overrides.kernelWarmup
      ? { ...base.kernelWarmup, ...overrides.kernelWarmup }
      : { ...base.kernelWarmup },
    bufferPool: overrides.bufferPool
      ? {
          bucket: { ...base.bufferPool.bucket, ...overrides.bufferPool.bucket },
          limits: { ...base.bufferPool.limits, ...overrides.bufferPool.limits },
          alignment: { ...base.bufferPool.alignment, ...overrides.bufferPool.alignment },
        }
      : { ...base.bufferPool },
    gpuCache: { ...base.gpuCache, ...overrides.gpuCache },
    tuner: { ...base.tuner, ...overrides.tuner },
    memory: overrides.memory
      ? {
          heapTesting: { ...base.memory.heapTesting, ...overrides.memory.heapTesting },
          segmentTesting: { ...base.memory.segmentTesting, ...overrides.memory.segmentTesting },
          addressSpace: { ...base.memory.addressSpace, ...overrides.memory.addressSpace },
          segmentAllocation: { ...base.memory.segmentAllocation, ...overrides.memory.segmentAllocation },
        }
      : { ...base.memory },
    hotSwap: overrides.hotSwap
      ? {
          ...base.hotSwap,
          ...overrides.hotSwap,
          trustedSigners: overrides.hotSwap.trustedSigners ?? base.hotSwap.trustedSigners,
        }
      : { ...base.hotSwap },
    intentBundle: overrides.intentBundle
      ? { ...base.intentBundle, ...overrides.intentBundle }
      : { ...base.intentBundle },
    bridge: { ...base.bridge, ...overrides.bridge },
  };
}

function mergeLoadingConfig(
  base,
  overrides
) {
  return {
    storage: overrides.storage
      ? {
          quota: { ...base.storage.quota, ...overrides.storage.quota },
          vramEstimation: { ...base.storage.vramEstimation, ...overrides.storage.vramEstimation },
          alignment: { ...base.storage.alignment, ...overrides.storage.alignment },
          backend: overrides.storage.backend
            ? {
                backend: overrides.storage.backend.backend ?? base.storage.backend.backend,
                opfs: { ...base.storage.backend.opfs, ...overrides.storage.backend.opfs },
                indexeddb: { ...base.storage.backend.indexeddb, ...overrides.storage.backend.indexeddb },
                memory: { ...base.storage.backend.memory, ...overrides.storage.backend.memory },
                streaming: { ...base.storage.backend.streaming, ...overrides.storage.backend.streaming },
              }
            : { ...base.storage.backend },
        }
      : { ...base.storage },
    distribution: { ...base.distribution, ...overrides.distribution },
    shardCache: { ...base.shardCache, ...overrides.shardCache },
    memoryManagement: overrides.memoryManagement
      ? {
          ...base.memoryManagement,
          ...overrides.memoryManagement,
          budget: {
            ...base.memoryManagement.budget,
            ...overrides.memoryManagement.budget,
          },
        }
      : { ...base.memoryManagement },
    prefetch: { ...base.prefetch, ...overrides.prefetch },
    opfsPath: { ...base.opfsPath, ...overrides.opfsPath },
    expertCache: { ...base.expertCache, ...overrides.expertCache },
    allowF32UpcastNonMatmul: overrides.allowF32UpcastNonMatmul ?? base.allowF32UpcastNonMatmul,
  };
}

function mergeInferenceConfig(
  base,
  overrides
) {
  const baseSession = base.session ?? {};
  const overrideSession = overrides.session ?? {};
  const baseSessionCompute = baseSession.compute ?? {};
  const overrideSessionCompute = overrideSession.compute ?? {};
  const baseSessionComputeDefaults = baseSessionCompute.defaults ?? {};
  const overrideSessionComputeDefaults = overrideSessionCompute.defaults ?? {};
  const baseExecutionPatch = base.executionPatch ?? {};
  const overrideExecutionPatch = overrides.executionPatch ?? {};
  const baseKernelPathPolicy = base.kernelPathPolicy ?? {};
  const overrideKernelPathPolicy = overrides.kernelPathPolicy ?? {};
  const baseDiffusion = base.diffusion ?? {};
  const baseDiffusionDecode = baseDiffusion.decode ?? {};
  const baseEnergy = base.energy ?? {};
  const baseEnergyQuintel = baseEnergy.quintel ?? {};
  const baseMoe = base.moe ?? {};
  const hasRuntimeKernelProfiles = Object.prototype.hasOwnProperty.call(
    overrideSessionCompute,
    'kernelProfiles'
  );

  return {
    prompt: overrides.prompt ?? base.prompt,
    debugTokens: overrides.debugTokens ?? base.debugTokens,
    batching: { ...base.batching, ...overrides.batching },
    sampling: { ...base.sampling, ...overrides.sampling },
    compute: { ...base.compute, ...overrides.compute },
    tokenizer: { ...base.tokenizer, ...overrides.tokenizer },
    largeWeights: { ...base.largeWeights, ...overrides.largeWeights },
    kvcache: { ...base.kvcache, ...overrides.kvcache },
    diffusion: overrides.diffusion
      ? {
          ...baseDiffusion,
          ...overrides.diffusion,
          scheduler: { ...baseDiffusion.scheduler, ...overrides.diffusion.scheduler },
          latent: { ...baseDiffusion.latent, ...overrides.diffusion.latent },
          textEncoder: { ...baseDiffusion.textEncoder, ...overrides.diffusion.textEncoder },
          decode: {
            ...baseDiffusionDecode,
            ...overrides.diffusion.decode,
            tiling: { ...baseDiffusionDecode.tiling, ...overrides.diffusion.decode?.tiling },
          },
          swapper: { ...baseDiffusion.swapper, ...overrides.diffusion.swapper },
          quantization: { ...baseDiffusion.quantization, ...overrides.diffusion.quantization },
        }
      : { ...baseDiffusion },
    energy: overrides.energy
      ? {
          ...baseEnergy,
          ...overrides.energy,
          problem: overrides.energy.problem ?? baseEnergy.problem,
          state: { ...baseEnergy.state, ...overrides.energy.state },
          init: { ...baseEnergy.init, ...overrides.energy.init },
          target: { ...baseEnergy.target, ...overrides.energy.target },
          loop: { ...baseEnergy.loop, ...overrides.energy.loop },
          diagnostics: { ...baseEnergy.diagnostics, ...overrides.energy.diagnostics },
          quintel: overrides.energy.quintel
            ? {
                ...baseEnergyQuintel,
                ...overrides.energy.quintel,
                rules: { ...baseEnergyQuintel.rules, ...overrides.energy.quintel.rules },
                weights: { ...baseEnergyQuintel.weights, ...overrides.energy.quintel.weights },
                clamp: { ...baseEnergyQuintel.clamp, ...overrides.energy.quintel.clamp },
              }
            : { ...baseEnergyQuintel },
        }
      : { ...baseEnergy },
    moe: overrides.moe
      ? {
          routing: { ...baseMoe.routing, ...overrides.moe.routing },
          cache: { ...baseMoe.cache, ...overrides.moe.cache },
        }
      : { ...baseMoe },
    speculative: { ...base.speculative, ...overrides.speculative },
    generation: { ...base.generation, ...overrides.generation },
    pipeline: overrides.pipeline ?? base.pipeline,
    kernelPath: mergeRuntimeKernelPath(base.kernelPath, overrides.kernelPath),
    kernelPathSource: overrides.kernelPathSource ?? base.kernelPathSource,
    kernelPathPolicy: mergeKernelPathPolicy(baseKernelPathPolicy, overrideKernelPathPolicy),
    chatTemplate: mergeShallowObject(base.chatTemplate, overrides.chatTemplate),
    session: {
      ...baseSession,
      ...overrideSession,
      compute: {
        ...baseSessionCompute,
        ...overrideSessionCompute,
        defaults: {
          ...baseSessionComputeDefaults,
          ...overrideSessionComputeDefaults,
        },
        ...(hasRuntimeKernelProfiles
          ? { kernelProfiles: overrideSessionCompute.kernelProfiles }
          : { kernelProfiles: baseSessionCompute.kernelProfiles }),
      },
      kvcache: replaceSubtree(overrideSession.kvcache, baseSession.kvcache),
      decodeLoop: replaceSubtree(overrideSession.decodeLoop, baseSession.decodeLoop),
      perLayerInputs: replaceSubtree(overrideSession.perLayerInputs, baseSession.perLayerInputs),
    },
    executionPatch: mergeExecutionPatchLists(baseExecutionPatch, overrideExecutionPatch),
    // Model-specific inference overrides (merged with manifest.inference at load time)
    modelOverrides: overrides.modelOverrides ?? base.modelOverrides,
  };
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

function mergeKernelThresholds(
  base,
  overrides
) {
  return {
    ...base,
    ...overrides,
    matmul: { ...base.matmul, ...overrides.matmul },
    rmsnorm: { ...base.rmsnorm, ...overrides.rmsnorm },
    rope: { ...base.rope, ...overrides.rope },
    attention: { ...base.attention, ...overrides.attention },
    fusedMatmul: { ...base.fusedMatmul, ...overrides.fusedMatmul },
    cast: { ...base.cast, ...overrides.cast },
  };
}

function mergeDebugConfig(
  base,
  overrides
) {
  if (!overrides) {
    return { ...base };
  }

  return {
    logOutput: { ...base.logOutput, ...overrides.logOutput },
    logHistory: { ...base.logHistory, ...overrides.logHistory },
    logLevel: { ...base.logLevel, ...overrides.logLevel },
    trace: { ...base.trace, ...overrides.trace },
    pipeline: { ...base.pipeline, ...overrides.pipeline },
    loader: { ...base.loader, ...overrides.loader },
    matmul: { ...base.matmul, ...overrides.matmul },
    kernelTrace: { ...base.kernelTrace, ...overrides.kernelTrace },
    probes: overrides.probes ?? base.probes,
    profiler: { ...base.profiler, ...overrides.profiler },
    perfGuards: { ...base.perfGuards, ...overrides.perfGuards },
  };
}

function mergeBenchmarkConfig(
  base,
  overrides
) {
  if (!overrides) {
    return { ...base };
  }

  return {
    output: { ...base.output, ...overrides.output },
    run: { ...base.run, ...overrides.run },
    stats: { ...base.stats, ...overrides.stats },
    comparison: { ...base.comparison, ...overrides.comparison },
    baselines: { ...base.baselines, ...overrides.baselines },
  };
}

function mergeEmulationConfig(
  base,
  overrides
) {
  if (!overrides) {
    return { ...base };
  }

  return createEmulationConfig(overrides);
}
