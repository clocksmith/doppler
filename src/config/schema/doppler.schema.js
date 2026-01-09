import { DEFAULT_LOADING_CONFIG } from './loading.schema.js';
import { DEFAULT_INFERENCE_DEFAULTS_CONFIG } from './inference-defaults.schema.js';
import { DEFAULT_SHARED_RUNTIME_CONFIG } from './shared-runtime.schema.js';

// =============================================================================
// Runtime Config (all non-model-specific settings)
// =============================================================================

export const DEFAULT_RUNTIME_CONFIG = {
  shared: DEFAULT_SHARED_RUNTIME_CONFIG,
  loading: DEFAULT_LOADING_CONFIG,
  inference: DEFAULT_INFERENCE_DEFAULTS_CONFIG,
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
    return { ...DEFAULT_DOPPLER_CONFIG };
  }

  const runtimeOverrides = overrides.runtime ?? {};
  return {
    model: overrides.model ?? DEFAULT_DOPPLER_CONFIG.model,
    runtime: overrides.runtime
      ? mergeRuntimeConfig(DEFAULT_RUNTIME_CONFIG, runtimeOverrides)
      : { ...DEFAULT_RUNTIME_CONFIG },
  };
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
    platform: overrides.platform ?? base.platform,
    kernelRegistry: { ...base.kernelRegistry, ...overrides.kernelRegistry },
    kernelThresholds: overrides.kernelThresholds
      ? mergeKernelThresholds(base.kernelThresholds, overrides.kernelThresholds)
      : { ...base.kernelThresholds },
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
        }
      : { ...base.storage },
    distribution: { ...base.distribution, ...overrides.distribution },
    shardCache: { ...base.shardCache, ...overrides.shardCache },
    memoryManagement: { ...base.memoryManagement, ...overrides.memoryManagement },
    opfsPath: { ...base.opfsPath, ...overrides.opfsPath },
    expertCache: { ...base.expertCache, ...overrides.expertCache },
  };
}

function mergeInferenceConfig(
  base,
  overrides
) {
  return {
    batching: { ...base.batching, ...overrides.batching },
    sampling: { ...base.sampling, ...overrides.sampling },
    compute: { ...base.compute, ...overrides.compute },
    tokenizer: { ...base.tokenizer, ...overrides.tokenizer },
    largeWeights: { ...base.largeWeights, ...overrides.largeWeights },
    kvcache: { ...base.kvcache, ...overrides.kvcache },
    moe: overrides.moe
      ? {
          routing: { ...base.moe.routing, ...overrides.moe.routing },
          cache: { ...base.moe.cache, ...overrides.moe.cache },
        }
      : { ...base.moe },
    prompt: overrides.prompt ?? base.prompt,
    pipeline: overrides.pipeline ?? base.pipeline,
    kernelPath: overrides.kernelPath ?? base.kernelPath,
    chatTemplate: overrides.chatTemplate
      ? { ...base.chatTemplate, ...overrides.chatTemplate }
      : base.chatTemplate,
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
  };
}
