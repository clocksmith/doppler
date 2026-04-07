import { DEFAULT_LOADING_CONFIG } from './loading.schema.js';
import { DEFAULT_SHARED_RUNTIME_CONFIG } from './shared-runtime.schema.js';
import { DEFAULT_EMULATION_CONFIG, createEmulationConfig } from './emulation.schema.js';
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
    largeWeights: {},
    kvcache: {},
    diffusion: {},
    energy: {},
    moe: {},
    speculative: {},
    generation: {
      disableMultiTokenDecode: false,
    },
    chatTemplate: {},
    session: {},
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

function cloneConfigTree(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

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
    memoryManagement: { ...base.memoryManagement, ...overrides.memoryManagement },
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
    kernelPath: chooseDefined(overrides.kernelPath, base.kernelPath),
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
