var __defProp = Object.defineProperty;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __esm = (fn, res) => function __init() {
  return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
};
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};

// src/config/schema/manifest.schema.ts
var SHARD_SIZE;
var init_manifest_schema = __esm({
  "src/config/schema/manifest.schema.ts"() {
    SHARD_SIZE = 64 * 1024 * 1024;
  }
});

// src/config/schema/kernel-path.schema.ts
var DEFAULT_ENTRY;
var init_kernel_path_schema = __esm({
  "src/config/schema/kernel-path.schema.ts"() {
    DEFAULT_ENTRY = "main";
  }
});

// src/config/schema/inference.schema.ts
var init_inference_schema = __esm({
  "src/config/schema/inference.schema.ts"() {
  }
});

// src/config/schema/conversion.schema.ts
var init_conversion_schema = __esm({
  "src/config/schema/conversion.schema.ts"() {
  }
});

// src/config/schema/loading.schema.ts
var DEFAULT_SHARD_CACHE_CONFIG, DEFAULT_MEMORY_MANAGEMENT_CONFIG, DEFAULT_OPFS_PATH_CONFIG, DEFAULT_EXPERT_CACHE_CONFIG, DEFAULT_LOADING_CONFIG;
var init_loading_schema = __esm({
  "src/config/schema/loading.schema.ts"() {
    DEFAULT_SHARD_CACHE_CONFIG = {
      opfsEntries: 2,
      networkEntries: 16,
      moeMaxEntries: 16
    };
    DEFAULT_MEMORY_MANAGEMENT_CONFIG = {
      flushIntervalLayers: 4,
      flushThresholdBytes: 256 * 1024 * 1024,
      // 256MB
      gpuQueueFlushLayers: 4,
      logIntervalMs: 3e4
      // 30 seconds
    };
    DEFAULT_OPFS_PATH_CONFIG = {
      opfsRootDir: "doppler-models"
    };
    DEFAULT_EXPERT_CACHE_CONFIG = {
      defaultSizeBytes: 2 * 1024 * 1024 * 1024,
      // 2GB
      maxBufferPercentage: 0.25
      // 25% of max buffer
    };
    DEFAULT_LOADING_CONFIG = {
      shardCache: DEFAULT_SHARD_CACHE_CONFIG,
      memoryManagement: DEFAULT_MEMORY_MANAGEMENT_CONFIG,
      opfsPath: DEFAULT_OPFS_PATH_CONFIG,
      expertCache: DEFAULT_EXPERT_CACHE_CONFIG
    };
  }
});

// src/config/schema/kernel-registry.schema.ts
var init_kernel_registry_schema = __esm({
  "src/config/schema/kernel-registry.schema.ts"() {
  }
});

// src/config/schema/storage.schema.ts
var DEFAULT_QUOTA_CONFIG, DEFAULT_VRAM_ESTIMATION_CONFIG, DEFAULT_STORAGE_ALIGNMENT_CONFIG, DEFAULT_STORAGE_FULL_CONFIG;
var init_storage_schema = __esm({
  "src/config/schema/storage.schema.ts"() {
    DEFAULT_QUOTA_CONFIG = {
      lowSpaceThresholdBytes: 500 * 1024 * 1024,
      // 500MB
      criticalSpaceThresholdBytes: 100 * 1024 * 1024,
      // 100MB
      monitorIntervalMs: 3e4
      // 30 seconds
    };
    DEFAULT_VRAM_ESTIMATION_CONFIG = {
      unifiedMemoryRatio: 0.5,
      // 50% of system RAM
      fallbackVramBytes: 2 * 1024 * 1024 * 1024,
      // 2GB
      lowVramHeadroomBytes: 500 * 1024 * 1024
      // 500MB
    };
    DEFAULT_STORAGE_ALIGNMENT_CONFIG = {
      bufferAlignmentBytes: 4096
      // 4KB alignment (typical page size)
    };
    DEFAULT_STORAGE_FULL_CONFIG = {
      quota: DEFAULT_QUOTA_CONFIG,
      vramEstimation: DEFAULT_VRAM_ESTIMATION_CONFIG,
      alignment: DEFAULT_STORAGE_ALIGNMENT_CONFIG
    };
  }
});

// src/config/schema/inference-defaults.schema.ts
var DEFAULT_BATCHING_DEFAULTS, DEFAULT_COMPUTE_DEFAULTS, DEFAULT_LARGE_WEIGHT_CONFIG, DEFAULT_SAMPLING_DEFAULTS, DEFAULT_TOKENIZER_DEFAULTS, DEFAULT_INFERENCE_DEFAULTS_CONFIG;
var init_inference_defaults_schema = __esm({
  "src/config/schema/inference-defaults.schema.ts"() {
    DEFAULT_BATCHING_DEFAULTS = {
      batchSize: 1,
      // Compare single-token
      maxTokens: 512,
      stopCheckMode: "per-token"
    };
    DEFAULT_COMPUTE_DEFAULTS = {
      activationDtype: "f32",
      // Safe default, F16 is experimental
      largeModelParamThreshold: 4e9,
      // 4B parameters
      paramEstimationMultiplier: 12
      // Rough approximation: 12 * hidden^2 * layers
    };
    DEFAULT_LARGE_WEIGHT_CONFIG = {
      enabled: true,
      safetyRatio: 0.9,
      preferF16: true,
      lmHeadChunkRows: null
    };
    DEFAULT_SAMPLING_DEFAULTS = {
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      repetitionPenalty: 1.1,
      greedyThreshold: 0.01,
      repetitionPenaltyWindow: 100
    };
    DEFAULT_TOKENIZER_DEFAULTS = {
      addBosToken: true,
      addEosToken: false
    };
    DEFAULT_INFERENCE_DEFAULTS_CONFIG = {
      batching: DEFAULT_BATCHING_DEFAULTS,
      sampling: DEFAULT_SAMPLING_DEFAULTS,
      compute: DEFAULT_COMPUTE_DEFAULTS,
      tokenizer: DEFAULT_TOKENIZER_DEFAULTS,
      largeWeights: DEFAULT_LARGE_WEIGHT_CONFIG,
      prompt: null,
      pipeline: null,
      kernelPath: void 0
    };
  }
});

// src/config/schema/distribution.schema.ts
var DEFAULT_DISTRIBUTION_CONFIG;
var init_distribution_schema = __esm({
  "src/config/schema/distribution.schema.ts"() {
    DEFAULT_DISTRIBUTION_CONFIG = {
      concurrentDownloads: 3,
      maxRetries: 3,
      initialRetryDelayMs: 1e3,
      maxRetryDelayMs: 3e4,
      maxChunkSizeBytes: 8 * 1024 * 1024,
      // 8MB
      cdnBasePath: null,
      progressUpdateIntervalMs: 100
    };
  }
});

// src/config/schema/moe.schema.ts
var DEFAULT_MOE_ROUTING_CONFIG, DEFAULT_MOE_CACHE_CONFIG, DEFAULT_MOE_RUNTIME_CONFIG;
var init_moe_schema = __esm({
  "src/config/schema/moe.schema.ts"() {
    DEFAULT_MOE_ROUTING_CONFIG = {
      numExperts: 8,
      topK: 2,
      normalizeWeights: true,
      routerDtype: "f32"
    };
    DEFAULT_MOE_CACHE_CONFIG = {
      dequantCacheMaxEntries: 128
    };
    DEFAULT_MOE_RUNTIME_CONFIG = {
      routing: DEFAULT_MOE_ROUTING_CONFIG,
      cache: DEFAULT_MOE_CACHE_CONFIG
    };
  }
});

// src/config/schema/kvcache.schema.ts
var DEFAULT_KVCACHE_CONFIG;
var init_kvcache_schema = __esm({
  "src/config/schema/kvcache.schema.ts"() {
    DEFAULT_KVCACHE_CONFIG = {
      maxSeqLen: 4096,
      kvDtype: "f16",
      layout: "contiguous",
      pageSize: 256,
      windowSize: 1024
    };
  }
});

// src/config/schema/gpu-cache.schema.ts
var DEFAULT_GPU_CACHE_CONFIG;
var init_gpu_cache_schema = __esm({
  "src/config/schema/gpu-cache.schema.ts"() {
    DEFAULT_GPU_CACHE_CONFIG = {
      uniformCacheMaxEntries: 256,
      uniformCacheMaxAgeMs: 6e4
      // 60 seconds
    };
  }
});

// src/config/schema/tuner.schema.ts
var DEFAULT_TUNER_CONFIG;
var init_tuner_schema = __esm({
  "src/config/schema/tuner.schema.ts"() {
    DEFAULT_TUNER_CONFIG = {
      cacheKeyPrefix: "doppler_kernel_tune_",
      defaultWarmupIterations: 3,
      defaultTimedIterations: 10
    };
  }
});

// src/config/schema/debug.schema.ts
var DEFAULT_LOG_OUTPUT_CONFIG, DEFAULT_LOG_HISTORY_CONFIG, DEFAULT_LOG_LEVEL_CONFIG, DEFAULT_TRACE_CONFIG, DEFAULT_PIPELINE_DEBUG_CONFIG, DEFAULT_DEBUG_CONFIG;
var init_debug_schema = __esm({
  "src/config/schema/debug.schema.ts"() {
    DEFAULT_LOG_OUTPUT_CONFIG = {
      stdout: true,
      file: null,
      append: true
    };
    DEFAULT_LOG_HISTORY_CONFIG = {
      maxLogHistoryEntries: 1e3
    };
    DEFAULT_LOG_LEVEL_CONFIG = {
      defaultLogLevel: "info"
    };
    DEFAULT_TRACE_CONFIG = {
      enabled: false,
      categories: ["all"],
      layers: null,
      maxDecodeSteps: 0,
      file: null
    };
    DEFAULT_PIPELINE_DEBUG_CONFIG = {
      enabled: false,
      categories: [],
      layers: null,
      maxDecodeSteps: 0,
      maxAbsThreshold: 1e4,
      bufferStats: false,
      readbackSampleSize: 512
    };
    DEFAULT_DEBUG_CONFIG = {
      logOutput: DEFAULT_LOG_OUTPUT_CONFIG,
      logHistory: DEFAULT_LOG_HISTORY_CONFIG,
      logLevel: DEFAULT_LOG_LEVEL_CONFIG,
      trace: DEFAULT_TRACE_CONFIG,
      pipeline: DEFAULT_PIPELINE_DEBUG_CONFIG,
      probes: []
    };
  }
});

// src/config/schema/hotswap.schema.ts
var DEFAULT_HOTSWAP_CONFIG;
var init_hotswap_schema = __esm({
  "src/config/schema/hotswap.schema.ts"() {
    DEFAULT_HOTSWAP_CONFIG = {
      enabled: false,
      localOnly: false,
      allowUnsignedLocal: false,
      trustedSigners: [],
      manifestUrl: null
    };
  }
});

// src/config/schema/buffer-pool.schema.ts
var DEFAULT_BUFFER_POOL_BUCKET_CONFIG, DEFAULT_BUFFER_POOL_LIMITS_CONFIG, DEFAULT_BUFFER_POOL_ALIGNMENT_CONFIG, DEFAULT_BUFFER_POOL_CONFIG;
var init_buffer_pool_schema = __esm({
  "src/config/schema/buffer-pool.schema.ts"() {
    DEFAULT_BUFFER_POOL_BUCKET_CONFIG = {
      minBucketSizeBytes: 256,
      // 256 bytes
      largeBufferThresholdBytes: 32 * 1024 * 1024,
      // 32MB
      largeBufferStepBytes: 16 * 1024 * 1024
      // 16MB
    };
    DEFAULT_BUFFER_POOL_LIMITS_CONFIG = {
      maxBuffersPerBucket: 8,
      maxTotalPooledBuffers: 64
    };
    DEFAULT_BUFFER_POOL_ALIGNMENT_CONFIG = {
      alignmentBytes: 256
      // WebGPU buffer alignment
    };
    DEFAULT_BUFFER_POOL_CONFIG = {
      bucket: DEFAULT_BUFFER_POOL_BUCKET_CONFIG,
      limits: DEFAULT_BUFFER_POOL_LIMITS_CONFIG,
      alignment: DEFAULT_BUFFER_POOL_ALIGNMENT_CONFIG
    };
  }
});

// src/config/schema/memory-limits.schema.ts
var MB, GB, DEFAULT_HEAP_TESTING_CONFIG, DEFAULT_SEGMENT_TESTING_CONFIG, DEFAULT_ADDRESS_SPACE_CONFIG, DEFAULT_SEGMENT_ALLOCATION_CONFIG, DEFAULT_MEMORY_LIMITS_CONFIG;
var init_memory_limits_schema = __esm({
  "src/config/schema/memory-limits.schema.ts"() {
    MB = 1024 * 1024;
    GB = 1024 * MB;
    DEFAULT_HEAP_TESTING_CONFIG = {
      heapTestSizes: [16 * GB, 8 * GB, 4 * GB, 2 * GB, 1 * GB],
      fallbackMaxHeapBytes: 1 * GB
    };
    DEFAULT_SEGMENT_TESTING_CONFIG = {
      segmentTestSizes: [1 * GB, 512 * MB, 256 * MB, 128 * MB],
      safeSegmentSizeBytes: 256 * MB
    };
    DEFAULT_ADDRESS_SPACE_CONFIG = {
      targetAddressSpaceBytes: 8 * GB
    };
    DEFAULT_SEGMENT_ALLOCATION_CONFIG = {
      fallbackSegmentSizeBytes: 4 * GB,
      segmentFallbackSizes: [512 * MB, 256 * MB, 128 * MB]
    };
    DEFAULT_MEMORY_LIMITS_CONFIG = {
      heapTesting: DEFAULT_HEAP_TESTING_CONFIG,
      segmentTesting: DEFAULT_SEGMENT_TESTING_CONFIG,
      addressSpace: DEFAULT_ADDRESS_SPACE_CONFIG,
      segmentAllocation: DEFAULT_SEGMENT_ALLOCATION_CONFIG
    };
  }
});

// src/config/schema/bridge.schema.ts
var DEFAULT_BRIDGE_CONFIG;
var init_bridge_schema = __esm({
  "src/config/schema/bridge.schema.ts"() {
    DEFAULT_BRIDGE_CONFIG = {
      maxReadSizeBytes: 100 * 1024 * 1024,
      // 100MB
      allowedDirectories: "/Users:/home:/tmp:/var/tmp"
    };
  }
});

// src/config/schema/quantization-defaults.schema.ts
var init_quantization_defaults_schema = __esm({
  "src/config/schema/quantization-defaults.schema.ts"() {
  }
});

// src/config/schema/kernel-thresholds.schema.ts
function getKernelThresholds() {
  return currentThresholds;
}
var DEFAULT_MATMUL_THRESHOLDS, DEFAULT_RMSNORM_THRESHOLDS, DEFAULT_ROPE_DEFAULTS, DEFAULT_ATTENTION_THRESHOLDS, DEFAULT_FUSED_MATMUL_THRESHOLDS, DEFAULT_CAST_THRESHOLDS, DTYPE_SIZES, DEFAULT_KERNEL_THRESHOLDS, currentThresholds;
var init_kernel_thresholds_schema = __esm({
  "src/config/schema/kernel-thresholds.schema.ts"() {
    DEFAULT_MATMUL_THRESHOLDS = {
      multicolThreshold: 256
    };
    DEFAULT_RMSNORM_THRESHOLDS = {
      smallThreshold: 256
    };
    DEFAULT_ROPE_DEFAULTS = {
      defaultTheta: 1e4,
      uniformSize: 32
    };
    DEFAULT_ATTENTION_THRESHOLDS = {
      chunkedMaxKVLen: 2048,
      minHeadDimForChunked: 128,
      tierHeadDimLimits: {
        tier3: 64,
        tier2: 128,
        tier1: 256
      },
      tierMinSharedMemory: {
        tier3: 16384,
        // 16KB for small models
        tier2: 32768,
        // 32KB for medium models
        tier1: 65536
        // 64KB for large models
      }
    };
    DEFAULT_FUSED_MATMUL_THRESHOLDS = {
      maxMediumN: 4096,
      colsPerWg: 4
    };
    DEFAULT_CAST_THRESHOLDS = {
      maxWorkgroupsPerDim: 65535
    };
    DTYPE_SIZES = {
      f32: 4,
      f16: 2,
      bf16: 2,
      i32: 4,
      u32: 4,
      i16: 2,
      u16: 2,
      i8: 1,
      u8: 1
    };
    DEFAULT_KERNEL_THRESHOLDS = {
      matmul: DEFAULT_MATMUL_THRESHOLDS,
      rmsnorm: DEFAULT_RMSNORM_THRESHOLDS,
      rope: DEFAULT_ROPE_DEFAULTS,
      attention: DEFAULT_ATTENTION_THRESHOLDS,
      fusedMatmul: DEFAULT_FUSED_MATMUL_THRESHOLDS,
      cast: DEFAULT_CAST_THRESHOLDS
    };
    currentThresholds = { ...DEFAULT_KERNEL_THRESHOLDS };
  }
});

// src/config/schema/doppler.schema.ts
function createDopplerConfig(overrides) {
  if (!overrides) {
    return { ...DEFAULT_DOPPLER_CONFIG };
  }
  return {
    model: overrides.model ?? DEFAULT_DOPPLER_CONFIG.model,
    runtime: overrides.runtime ? mergeRuntimeConfig(DEFAULT_RUNTIME_CONFIG, overrides.runtime) : { ...DEFAULT_RUNTIME_CONFIG },
    platform: overrides.platform
  };
}
function mergeRuntimeConfig(base, overrides) {
  return {
    distribution: { ...base.distribution, ...overrides.distribution },
    storage: overrides.storage ? {
      quota: { ...base.storage.quota, ...overrides.storage.quota },
      vramEstimation: { ...base.storage.vramEstimation, ...overrides.storage.vramEstimation },
      alignment: { ...base.storage.alignment, ...overrides.storage.alignment }
    } : { ...base.storage },
    loading: overrides.loading ? {
      shardCache: { ...base.loading.shardCache, ...overrides.loading.shardCache },
      memoryManagement: { ...base.loading.memoryManagement, ...overrides.loading.memoryManagement },
      opfsPath: { ...base.loading.opfsPath, ...overrides.loading.opfsPath },
      expertCache: { ...base.loading.expertCache, ...overrides.loading.expertCache }
    } : { ...base.loading },
    inference: overrides.inference ? {
      batching: { ...base.inference.batching, ...overrides.inference.batching },
      sampling: { ...base.inference.sampling, ...overrides.inference.sampling },
      compute: { ...base.inference.compute, ...overrides.inference.compute },
      tokenizer: { ...base.inference.tokenizer, ...overrides.inference.tokenizer },
      largeWeights: { ...base.inference.largeWeights, ...overrides.inference.largeWeights },
      prompt: overrides.inference.prompt ?? base.inference.prompt,
      pipeline: overrides.inference.pipeline ?? base.inference.pipeline,
      kernelPath: overrides.inference.kernelPath ?? base.inference.kernelPath,
      chatTemplate: overrides.inference.chatTemplate ? { ...base.inference.chatTemplate, ...overrides.inference.chatTemplate } : base.inference.chatTemplate,
      // Model-specific inference overrides (merged with manifest.inference at load time)
      modelOverrides: overrides.inference.modelOverrides ?? base.inference.modelOverrides
    } : { ...base.inference },
    kvcache: { ...base.kvcache, ...overrides.kvcache },
    moe: overrides.moe ? {
      routing: { ...base.moe.routing, ...overrides.moe.routing },
      cache: { ...base.moe.cache, ...overrides.moe.cache }
    } : { ...base.moe },
    bufferPool: overrides.bufferPool ? {
      bucket: { ...base.bufferPool.bucket, ...overrides.bufferPool.bucket },
      limits: { ...base.bufferPool.limits, ...overrides.bufferPool.limits },
      alignment: { ...base.bufferPool.alignment, ...overrides.bufferPool.alignment }
    } : { ...base.bufferPool },
    gpuCache: { ...base.gpuCache, ...overrides.gpuCache },
    tuner: { ...base.tuner, ...overrides.tuner },
    memory: overrides.memory ? {
      heapTesting: { ...base.memory.heapTesting, ...overrides.memory.heapTesting },
      segmentTesting: { ...base.memory.segmentTesting, ...overrides.memory.segmentTesting },
      addressSpace: { ...base.memory.addressSpace, ...overrides.memory.addressSpace },
      segmentAllocation: { ...base.memory.segmentAllocation, ...overrides.memory.segmentAllocation }
    } : { ...base.memory },
    debug: overrides.debug ? {
      logOutput: { ...base.debug.logOutput, ...overrides.debug.logOutput },
      logHistory: { ...base.debug.logHistory, ...overrides.debug.logHistory },
      logLevel: { ...base.debug.logLevel, ...overrides.debug.logLevel },
      trace: { ...base.debug.trace, ...overrides.debug.trace },
      pipeline: { ...base.debug.pipeline, ...overrides.debug.pipeline },
      probes: overrides.debug.probes ?? base.debug.probes
    } : { ...base.debug },
    hotSwap: overrides.hotSwap ? {
      ...base.hotSwap,
      ...overrides.hotSwap,
      trustedSigners: overrides.hotSwap.trustedSigners ?? base.hotSwap.trustedSigners
    } : { ...base.hotSwap },
    bridge: { ...base.bridge, ...overrides.bridge }
  };
}
var DEFAULT_RUNTIME_CONFIG, DEFAULT_DOPPLER_CONFIG;
var init_doppler_schema = __esm({
  "src/config/schema/doppler.schema.ts"() {
    init_distribution_schema();
    init_storage_schema();
    init_loading_schema();
    init_inference_defaults_schema();
    init_kvcache_schema();
    init_moe_schema();
    init_buffer_pool_schema();
    init_gpu_cache_schema();
    init_tuner_schema();
    init_memory_limits_schema();
    init_debug_schema();
    init_hotswap_schema();
    init_bridge_schema();
    DEFAULT_RUNTIME_CONFIG = {
      distribution: DEFAULT_DISTRIBUTION_CONFIG,
      storage: DEFAULT_STORAGE_FULL_CONFIG,
      loading: DEFAULT_LOADING_CONFIG,
      inference: DEFAULT_INFERENCE_DEFAULTS_CONFIG,
      kvcache: DEFAULT_KVCACHE_CONFIG,
      moe: DEFAULT_MOE_RUNTIME_CONFIG,
      bufferPool: DEFAULT_BUFFER_POOL_CONFIG,
      gpuCache: DEFAULT_GPU_CACHE_CONFIG,
      tuner: DEFAULT_TUNER_CONFIG,
      memory: DEFAULT_MEMORY_LIMITS_CONFIG,
      debug: DEFAULT_DEBUG_CONFIG,
      hotSwap: DEFAULT_HOTSWAP_CONFIG,
      bridge: DEFAULT_BRIDGE_CONFIG
    };
    DEFAULT_DOPPLER_CONFIG = {
      model: void 0,
      runtime: DEFAULT_RUNTIME_CONFIG,
      platform: void 0
    };
  }
});

// src/config/schema/index.ts
var init_schema = __esm({
  "src/config/schema/index.ts"() {
    init_manifest_schema();
    init_kernel_path_schema();
    init_inference_schema();
    init_conversion_schema();
    init_loading_schema();
    init_kernel_registry_schema();
    init_storage_schema();
    init_inference_defaults_schema();
    init_distribution_schema();
    init_moe_schema();
    init_kvcache_schema();
    init_gpu_cache_schema();
    init_tuner_schema();
    init_debug_schema();
    init_hotswap_schema();
    init_buffer_pool_schema();
    init_memory_limits_schema();
    init_bridge_schema();
    init_quantization_defaults_schema();
    init_kernel_thresholds_schema();
    init_doppler_schema();
  }
});

// src/config/runtime.ts
function getRuntimeConfig() {
  return runtimeConfig;
}
var runtimeConfig;
var init_runtime = __esm({
  "src/config/runtime.ts"() {
    init_schema();
    init_debug();
    runtimeConfig = createDopplerConfig().runtime;
  }
});

// src/debug/index.ts
function signalDone(payload) {
  console.log(`${SIGNALS.DONE} ${JSON.stringify(payload)}`);
}
function signalResult(data) {
  console.log(`${SIGNALS.RESULT} ${JSON.stringify(data)}`);
}
function signalError(error, details) {
  console.log(`${SIGNALS.ERROR} ${JSON.stringify({ error, ...details })}`);
}
function signalProgress(percent, message) {
  console.log(`${SIGNALS.PROGRESS} ${JSON.stringify({ percent, message })}`);
}
function setLogLevel(level) {
  const levelMap = {
    debug: LOG_LEVELS2.DEBUG,
    verbose: LOG_LEVELS2.VERBOSE,
    info: LOG_LEVELS2.INFO,
    warn: LOG_LEVELS2.WARN,
    error: LOG_LEVELS2.ERROR,
    silent: LOG_LEVELS2.SILENT
  };
  currentLogLevel = levelMap[level.toLowerCase()] ?? LOG_LEVELS2.INFO;
  console.log(`[Doppler] Log level set to: ${level.toUpperCase()}`);
}
function getLogLevel() {
  for (const [name, value] of Object.entries(LOG_LEVELS2)) {
    if (value === currentLogLevel)
      return name.toLowerCase();
  }
  return "info";
}
function setTrace(categories, options) {
  if (categories === false) {
    enabledTraceCategories.clear();
    console.log("[Doppler] Trace disabled");
    return;
  }
  const catArray = typeof categories === "string" ? categories.split(",").map((s) => s.trim()) : categories;
  enabledTraceCategories.clear();
  const hasAll = catArray.includes("all");
  if (hasAll) {
    for (const cat of TRACE_CATEGORIES) {
      enabledTraceCategories.add(cat);
    }
  }
  for (const cat of catArray) {
    if (cat === "all")
      continue;
    if (cat.startsWith("-")) {
      const exclude = cat.slice(1);
      enabledTraceCategories.delete(exclude);
    } else if (TRACE_CATEGORIES.includes(cat)) {
      enabledTraceCategories.add(cat);
    }
  }
  if (options?.layers) {
    traceLayerFilter = options.layers;
  }
  if (options?.maxDecodeSteps !== void 0) {
    traceMaxDecodeSteps = options.maxDecodeSteps;
  }
  if (options?.breakOnAnomaly !== void 0) {
    traceBreakOnAnomaly = options.breakOnAnomaly;
  }
  const enabled = [...enabledTraceCategories].join(",") || "none";
  console.log(`[Doppler] Trace categories: ${enabled}`);
}
function getTrace() {
  return [...enabledTraceCategories];
}
function isTraceEnabled(category, layerIdx) {
  if (!enabledTraceCategories.has(category))
    return false;
  if (layerIdx !== void 0 && traceLayerFilter.length > 0) {
    if (!traceLayerFilter.includes(layerIdx))
      return false;
  }
  if (traceMaxDecodeSteps > 0 && traceDecodeStep > traceMaxDecodeSteps) {
    return false;
  }
  return true;
}
function setBenchmarkMode(enabled) {
  benchmarkMode = enabled;
  if (enabled) {
    const noop = () => {
    };
    console.log = noop;
    console.debug = noop;
    console.info = noop;
    originalConsoleLog("[Doppler] Benchmark mode enabled - logging silenced");
  } else {
    console.log = originalConsoleLog;
    console.debug = originalConsoleDebug;
    console.info = originalConsoleInfo;
    console.log("[Doppler] Benchmark mode disabled - logging restored");
  }
}
function isBenchmarkMode() {
  return benchmarkMode;
}
function initFromUrlParams() {
  if (typeof window === "undefined")
    return;
  const params = new URLSearchParams(window.location.search);
  const logLevel = params.get("log");
  if (logLevel) {
    setLogLevel(logLevel);
  }
  const traceParam = params.get("trace");
  if (traceParam) {
    const layers = params.get("layers")?.split(",").map(Number).filter((n) => !isNaN(n));
    const breakOn = params.get("break") === "1";
    setTrace(traceParam, { layers, breakOnAnomaly: breakOn });
  }
  const debugParam = params.get("debug");
  if (debugParam === "1" && !traceParam) {
    setTrace("all");
    setLogLevel("verbose");
  }
}
function shouldLog(module, level) {
  if (level < currentLogLevel)
    return false;
  const moduleLower = module.toLowerCase();
  if (enabledModules.size > 0 && !enabledModules.has(moduleLower)) {
    return false;
  }
  if (disabledModules.has(moduleLower)) {
    return false;
  }
  return true;
}
function formatMessage(module, message) {
  const timestamp = performance.now().toFixed(1);
  return `[${timestamp}ms][${module}] ${message}`;
}
function formatTraceMessage(category, message, layerIdx) {
  const timestamp = performance.now().toFixed(1);
  const layerTag = layerIdx !== void 0 ? `L${layerIdx}:` : "";
  return `[${timestamp}ms][TRACE:${category}] ${layerTag}${message}`;
}
function storeLog(level, module, message, data) {
  logHistory.push({
    time: Date.now(),
    perfTime: performance.now(),
    level,
    module,
    message,
    data
  });
  const maxHistory = getRuntimeConfig().debug.logHistory.maxLogHistoryEntries;
  if (logHistory.length > maxHistory) {
    logHistory.shift();
  }
}
function f16ToF32(h) {
  const sign = h >> 15 & 1;
  const exp = h >> 10 & 31;
  const mant = h & 1023;
  if (exp === 0) {
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024);
  } else if (exp === 31) {
    return mant === 0 ? sign ? -Infinity : Infinity : NaN;
  }
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
}
function getLogHistory(filter = {}) {
  let history = [...logHistory];
  if (filter.level) {
    history = history.filter((h) => h.level === filter.level.toUpperCase());
  }
  if (filter.module) {
    const m = filter.module.toLowerCase();
    history = history.filter((h) => h.module.toLowerCase().includes(m));
  }
  if (filter.last) {
    history = history.slice(-filter.last);
  }
  return history;
}
function printLogSummary(count = 20) {
  const recent = logHistory.slice(-count);
  console.log("=== Recent Logs ===");
  for (const entry of recent) {
    const time = entry.perfTime.toFixed(1);
    console.log(`[${time}ms][${entry.level}][${entry.module}] ${entry.message}`);
  }
  console.log("===================");
}
function getDebugSnapshot() {
  return {
    timestamp: (/* @__PURE__ */ new Date()).toISOString(),
    logLevel: Object.keys(LOG_LEVELS2).find(
      (k) => LOG_LEVELS2[k] === currentLogLevel
    ),
    traceCategories: [...enabledTraceCategories],
    enabledModules: [...enabledModules],
    disabledModules: [...disabledModules],
    recentLogs: logHistory.slice(-50).map((e) => ({
      time: e.perfTime.toFixed(1),
      level: e.level,
      module: e.module,
      message: e.message
    })),
    errorCount: logHistory.filter((e) => e.level === "ERROR").length,
    warnCount: logHistory.filter((e) => e.level === "WARN").length
  };
}
var SIGNALS, LOG_LEVELS2, TRACE_CATEGORIES, currentLogLevel, enabledModules, disabledModules, logHistory, gpuDevice, enabledTraceCategories, traceLayerFilter, traceDecodeStep, traceMaxDecodeSteps, traceBreakOnAnomaly, benchmarkMode, originalConsoleLog, originalConsoleDebug, originalConsoleInfo, log, trace, tensor, perf, DOPPLER_API;
var init_debug = __esm({
  "src/debug/index.ts"() {
    init_runtime();
    SIGNALS = {
      /** Task completed (success or error) - always emitted at end */
      DONE: "[DOPPLER:DONE]",
      /** Full result payload (JSON) - emitted before DONE for data extraction */
      RESULT: "[DOPPLER:RESULT]",
      /** Error occurred - can be emitted before DONE */
      ERROR: "[DOPPLER:ERROR]",
      /** Progress update (optional) */
      PROGRESS: "[DOPPLER:PROGRESS]"
    };
    LOG_LEVELS2 = {
      DEBUG: 0,
      VERBOSE: 1,
      INFO: 2,
      WARN: 3,
      ERROR: 4,
      SILENT: 5
    };
    TRACE_CATEGORIES = [
      "loader",
      // Model loading (shards, weights)
      "kernels",
      // GPU kernel execution
      "logits",
      // Logit computation
      "embed",
      // Embedding layer
      "attn",
      // Attention
      "ffn",
      // Feed-forward
      "kv",
      // KV cache
      "sample",
      // Token sampling
      "buffers",
      // GPU buffer stats (expensive!)
      "perf"
      // Timing
    ];
    currentLogLevel = LOG_LEVELS2.INFO;
    enabledModules = /* @__PURE__ */ new Set();
    disabledModules = /* @__PURE__ */ new Set();
    logHistory = [];
    gpuDevice = null;
    enabledTraceCategories = /* @__PURE__ */ new Set();
    traceLayerFilter = [];
    traceDecodeStep = 0;
    traceMaxDecodeSteps = 0;
    traceBreakOnAnomaly = false;
    benchmarkMode = false;
    originalConsoleLog = console.log;
    originalConsoleDebug = console.debug;
    originalConsoleInfo = console.info;
    log = {
      /**
       * Debug level logging (most verbose).
       */
      debug(module, message, data) {
        if (!shouldLog(module, LOG_LEVELS2.DEBUG))
          return;
        const formatted = formatMessage(module, message);
        storeLog("DEBUG", module, message, data);
        if (data !== void 0) {
          console.debug(formatted, data);
        } else {
          console.debug(formatted);
        }
      },
      /**
       * Verbose level logging (detailed operational info).
       */
      verbose(module, message, data) {
        if (!shouldLog(module, LOG_LEVELS2.VERBOSE))
          return;
        const formatted = formatMessage(module, message);
        storeLog("VERBOSE", module, message, data);
        if (data !== void 0) {
          console.log(formatted, data);
        } else {
          console.log(formatted);
        }
      },
      /**
       * Info level logging (normal operations).
       */
      info(module, message, data) {
        if (!shouldLog(module, LOG_LEVELS2.INFO))
          return;
        const formatted = formatMessage(module, message);
        storeLog("INFO", module, message, data);
        if (data !== void 0) {
          console.log(formatted, data);
        } else {
          console.log(formatted);
        }
      },
      /**
       * Warning level logging.
       */
      warn(module, message, data) {
        if (!shouldLog(module, LOG_LEVELS2.WARN))
          return;
        const formatted = formatMessage(module, message);
        storeLog("WARN", module, message, data);
        if (data !== void 0) {
          console.warn(formatted, data);
        } else {
          console.warn(formatted);
        }
      },
      /**
       * Error level logging.
       */
      error(module, message, data) {
        if (!shouldLog(module, LOG_LEVELS2.ERROR))
          return;
        const formatted = formatMessage(module, message);
        storeLog("ERROR", module, message, data);
        if (data !== void 0) {
          console.error(formatted, data);
        } else {
          console.error(formatted);
        }
      },
      /**
       * Always log regardless of level (for critical messages).
       */
      always(module, message, data) {
        const formatted = formatMessage(module, message);
        storeLog("ALWAYS", module, message, data);
        if (data !== void 0) {
          console.log(formatted, data);
        } else {
          console.log(formatted);
        }
      }
    };
    trace = {
      /**
       * Trace model loading operations.
       */
      loader(message, data) {
        if (!isTraceEnabled("loader"))
          return;
        const formatted = formatTraceMessage("loader", message);
        storeLog("TRACE:loader", "Loader", message, data);
        if (data !== void 0) {
          console.log(formatted, data);
        } else {
          console.log(formatted);
        }
      },
      /**
       * Trace kernel execution.
       */
      kernels(message, data) {
        if (!isTraceEnabled("kernels"))
          return;
        const formatted = formatTraceMessage("kernels", message);
        storeLog("TRACE:kernels", "Kernels", message, data);
        if (data !== void 0) {
          console.log(formatted, data);
        } else {
          console.log(formatted);
        }
      },
      /**
       * Trace logit computation.
       */
      logits(message, data) {
        if (!isTraceEnabled("logits"))
          return;
        const formatted = formatTraceMessage("logits", message);
        storeLog("TRACE:logits", "Logits", message, data);
        if (data !== void 0) {
          console.log(formatted, data);
        } else {
          console.log(formatted);
        }
      },
      /**
       * Trace embedding layer.
       */
      embed(message, data) {
        if (!isTraceEnabled("embed"))
          return;
        const formatted = formatTraceMessage("embed", message);
        storeLog("TRACE:embed", "Embed", message, data);
        if (data !== void 0) {
          console.log(formatted, data);
        } else {
          console.log(formatted);
        }
      },
      /**
       * Trace attention computation.
       */
      attn(layerIdx, message, data) {
        if (!isTraceEnabled("attn", layerIdx))
          return;
        const formatted = formatTraceMessage("attn", message, layerIdx);
        storeLog("TRACE:attn", `Attn:L${layerIdx}`, message, data);
        if (data !== void 0) {
          console.log(formatted, data);
        } else {
          console.log(formatted);
        }
      },
      /**
       * Trace feed-forward network.
       */
      ffn(layerIdx, message, data) {
        if (!isTraceEnabled("ffn", layerIdx))
          return;
        const formatted = formatTraceMessage("ffn", message, layerIdx);
        storeLog("TRACE:ffn", `FFN:L${layerIdx}`, message, data);
        if (data !== void 0) {
          console.log(formatted, data);
        } else {
          console.log(formatted);
        }
      },
      /**
       * Trace KV cache operations.
       */
      kv(layerIdx, message, data) {
        if (!isTraceEnabled("kv", layerIdx))
          return;
        const formatted = formatTraceMessage("kv", message, layerIdx);
        storeLog("TRACE:kv", `KV:L${layerIdx}`, message, data);
        if (data !== void 0) {
          console.log(formatted, data);
        } else {
          console.log(formatted);
        }
      },
      /**
       * Trace token sampling.
       */
      sample(message, data) {
        if (!isTraceEnabled("sample"))
          return;
        const formatted = formatTraceMessage("sample", message);
        storeLog("TRACE:sample", "Sample", message, data);
        if (data !== void 0) {
          console.log(formatted, data);
        } else {
          console.log(formatted);
        }
      },
      /**
       * Trace buffer stats (expensive - requires GPU readback).
       */
      buffers(message, data) {
        if (!isTraceEnabled("buffers"))
          return;
        const formatted = formatTraceMessage("buffers", message);
        storeLog("TRACE:buffers", "Buffers", message, data);
        if (data !== void 0) {
          console.log(formatted, data);
        } else {
          console.log(formatted);
        }
      },
      /**
       * Trace performance timing.
       */
      perf(message, data) {
        if (!isTraceEnabled("perf"))
          return;
        const formatted = formatTraceMessage("perf", message);
        storeLog("TRACE:perf", "Perf", message, data);
        if (data !== void 0) {
          console.log(formatted, data);
        } else {
          console.log(formatted);
        }
      }
    };
    tensor = {
      /**
       * Inspect a GPU or CPU tensor and log statistics.
       */
      async inspect(buffer, label, options = {}) {
        const { shape = [], maxPrint = 8, checkNaN = true } = options;
        let data;
        let isGPU = false;
        if (buffer && typeof buffer.mapAsync === "function") {
          const gpuBuffer = buffer;
          await gpuBuffer.mapAsync(GPUMapMode.READ);
          data = new Float32Array(gpuBuffer.getMappedRange().slice(0));
          gpuBuffer.unmap();
        } else if (buffer && buffer.size !== void 0 && gpuDevice) {
          isGPU = true;
          const gpuBuffer = buffer;
          const readSize = Math.min(gpuBuffer.size, 4096);
          const staging = gpuDevice.createBuffer({
            label: `debug_staging_${label}`,
            size: readSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
          });
          const encoder = gpuDevice.createCommandEncoder();
          encoder.copyBufferToBuffer(gpuBuffer, 0, staging, 0, readSize);
          gpuDevice.queue.submit([encoder.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          data = new Float32Array(staging.getMappedRange().slice(0));
          staging.unmap();
          staging.destroy();
        } else if (buffer instanceof Float32Array || buffer instanceof Float64Array) {
          data = buffer instanceof Float32Array ? buffer : new Float32Array(buffer);
        } else if (buffer instanceof Uint16Array) {
          data = new Float32Array(buffer.length);
          for (let i = 0; i < buffer.length; i++) {
            data[i] = f16ToF32(buffer[i]);
          }
        } else {
          log.warn("Debug", `Cannot inspect tensor "${label}": unknown type`);
          return null;
        }
        let min = Infinity, max = -Infinity, sum = 0, sumSq = 0;
        let nanCount = 0, infCount = 0, zeroCount = 0;
        for (let i = 0; i < data.length; i++) {
          const v = data[i];
          if (Number.isNaN(v)) {
            nanCount++;
            continue;
          }
          if (!Number.isFinite(v)) {
            infCount++;
            continue;
          }
          if (v === 0)
            zeroCount++;
          min = Math.min(min, v);
          max = Math.max(max, v);
          sum += v;
          sumSq += v * v;
        }
        const validCount = data.length - nanCount - infCount;
        const mean = validCount > 0 ? sum / validCount : 0;
        const variance = validCount > 0 ? sumSq / validCount - mean * mean : 0;
        const std = Math.sqrt(Math.max(0, variance));
        const stats = {
          label,
          shape,
          size: data.length,
          isGPU,
          min,
          max,
          mean,
          std,
          nanCount,
          infCount,
          zeroCount,
          zeroPercent: (zeroCount / data.length * 100).toFixed(1),
          first: Array.from(data.slice(0, maxPrint)).map((v) => v.toFixed(4)),
          last: Array.from(data.slice(-maxPrint)).map((v) => v.toFixed(4))
        };
        const shapeStr = shape.length > 0 ? `[${shape.join("x")}]` : `[${data.length}]`;
        log.debug(
          "Tensor",
          `${label} ${shapeStr}: min=${min.toFixed(4)}, max=${max.toFixed(4)}, mean=${mean.toFixed(4)}, std=${std.toFixed(4)}`
        );
        if (checkNaN && (nanCount > 0 || infCount > 0)) {
          log.warn("Tensor", `${label} has ${nanCount} NaN and ${infCount} Inf values!`);
        }
        return stats;
      },
      /**
       * Compare two tensors element-wise.
       */
      compare(a, b, label, tolerance = 1e-5) {
        if (a.length !== b.length) {
          log.error("Tensor", `${label}: size mismatch ${a.length} vs ${b.length}`);
          return { label, match: false, error: "size_mismatch", maxDiff: 0, maxDiffIdx: 0, avgDiff: 0, mismatchCount: 0, mismatchPercent: "0" };
        }
        let maxDiff = 0, maxDiffIdx = 0;
        let sumDiff = 0;
        let mismatchCount = 0;
        for (let i = 0; i < a.length; i++) {
          const diff = Math.abs(a[i] - b[i]);
          sumDiff += diff;
          if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIdx = i;
          }
          if (diff > tolerance) {
            mismatchCount++;
          }
        }
        const avgDiff = sumDiff / a.length;
        const match = mismatchCount === 0;
        const result = {
          label,
          match,
          maxDiff,
          maxDiffIdx,
          avgDiff,
          mismatchCount,
          mismatchPercent: (mismatchCount / a.length * 100).toFixed(2)
        };
        if (match) {
          log.debug("Tensor", `${label}: MATCH (maxDiff=${maxDiff.toExponential(2)})`);
        } else {
          log.warn(
            "Tensor",
            `${label}: MISMATCH ${mismatchCount}/${a.length} (${result.mismatchPercent}%) maxDiff=${maxDiff.toFixed(6)} at idx=${maxDiffIdx}`
          );
        }
        return result;
      },
      /**
       * Check tensor for common issues.
       */
      healthCheck(data, label) {
        const issues = [];
        const allZero = data.every((v) => v === 0);
        if (allZero) {
          issues.push("ALL_ZEROS");
        }
        const hasNaN = data.some((v) => Number.isNaN(v));
        const hasInf = data.some((v) => !Number.isFinite(v) && !Number.isNaN(v));
        if (hasNaN)
          issues.push("HAS_NAN");
        if (hasInf)
          issues.push("HAS_INF");
        const maxAbs = Math.max(...Array.from(data).map(Math.abs).filter(Number.isFinite));
        if (maxAbs > 1e6)
          issues.push(`EXTREME_VALUES (max=${maxAbs.toExponential(2)})`);
        const tinyCount = data.filter((v) => Math.abs(v) > 0 && Math.abs(v) < 1e-30).length;
        if (tinyCount > data.length * 0.1) {
          issues.push(`POTENTIAL_UNDERFLOW (${tinyCount} tiny values)`);
        }
        const healthy = issues.length === 0;
        if (healthy) {
          log.debug("Tensor", `${label}: healthy`);
        } else {
          log.warn("Tensor", `${label}: issues found - ${issues.join(", ")}`);
        }
        return { label, healthy, issues };
      }
    };
    perf = {
      marks: /* @__PURE__ */ new Map(),
      /**
       * Start a timing mark.
       */
      mark(label) {
        this.marks.set(label, performance.now());
      },
      /**
       * End a timing mark and log duration.
       */
      measure(label, module = "Perf") {
        const start = this.marks.get(label);
        if (start === void 0) {
          log.warn(module, `No mark found for "${label}"`);
          return 0;
        }
        const duration = performance.now() - start;
        this.marks.delete(label);
        log.debug(module, `${label}: ${duration.toFixed(2)}ms`);
        return duration;
      },
      /**
       * Time an async operation.
       */
      async time(label, fn) {
        const start = performance.now();
        const result = await fn();
        const durationMs = performance.now() - start;
        log.debug("Perf", `${label}: ${durationMs.toFixed(2)}ms`);
        return { result, durationMs };
      }
    };
    DOPPLER_API = {
      // Trace categories
      trace,
      setTrace,
      getTrace,
      // Log levels
      log,
      setLogLevel,
      getLogLevel,
      // Tensor inspection
      tensor,
      inspect: tensor.inspect.bind(tensor),
      // Performance
      perf,
      // Other
      setBenchmarkMode,
      isBenchmarkMode,
      // History
      getLogHistory,
      printLogSummary,
      getDebugSnapshot,
      // Completion signals
      SIGNALS,
      signalDone,
      signalResult,
      signalError,
      signalProgress
    };
    if (typeof window !== "undefined") {
      window.DOPPLER = {
        ...window.DOPPLER || {},
        ...DOPPLER_API
      };
      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initFromUrlParams);
      } else {
        initFromUrlParams();
      }
    }
  }
});

// src/gpu/perf-guards.ts
function trackSubmit() {
  if (config.trackSubmitCount) {
    counters.submits++;
    if (config.logExpensiveOps) {
      trace.perf(`PerfGuard: Submit #${counters.submits}`);
    }
  }
}
function trackAllocation(size, label) {
  if (config.trackAllocations) {
    counters.allocations++;
    if (config.logExpensiveOps) {
      trace.buffers(`PerfGuard: Allocation #${counters.allocations}: ${size} bytes (${label || "unlabeled"})`);
    }
  }
}
function allowReadback(reason) {
  if (!config.allowGPUReadback) {
    const message = `PerfGuard: GPU readback blocked: ${reason || "unknown reason"}`;
    if (config.strictMode) {
      throw new Error(message);
    }
    if (config.logExpensiveOps) {
      log.warn("PerfGuard", message);
    }
    return false;
  }
  if (config.trackSubmitCount) {
    counters.readbacks++;
    if (config.logExpensiveOps) {
      trace.perf(`PerfGuard: Readback #${counters.readbacks}: ${reason || "unknown"}`);
    }
  }
  return true;
}
var DEFAULT_CONFIG, config, counters;
var init_perf_guards = __esm({
  "src/gpu/perf-guards.ts"() {
    init_debug();
    DEFAULT_CONFIG = {
      allowGPUReadback: true,
      // Default to allowed for backward compatibility
      trackSubmitCount: false,
      trackAllocations: false,
      logExpensiveOps: false,
      strictMode: false
    };
    config = { ...DEFAULT_CONFIG };
    counters = {
      submits: 0,
      allocations: 0,
      readbacks: 0,
      startTime: 0
    };
  }
});

// src/gpu/submit-tracker.ts
function recordSubmit(durationMs, source) {
  if (!TRACK_SUBMITS)
    return;
  submitCount++;
  submitTimes.push(durationMs);
  totalSubmitMs += durationMs;
  maxSubmitMs = Math.max(maxSubmitMs, durationMs);
  minSubmitMs = Math.min(minSubmitMs, durationMs);
  if (source) {
    submitSources.set(source, (submitSources.get(source) || 0) + 1);
  }
  const ps = phaseStats[currentPhase];
  ps.count++;
  ps.times.push(durationMs);
  ps.totalMs += durationMs;
  ps.maxMs = Math.max(ps.maxMs, durationMs);
  ps.minMs = Math.min(ps.minMs, durationMs);
  if (source) {
    ps.sources.set(source, (ps.sources.get(source) || 0) + 1);
  }
}
function extractSourceFromStack() {
  const stack = new Error().stack;
  if (!stack)
    return "unknown";
  const lines = stack.split("\n");
  for (let i = 3; i < lines.length; i++) {
    const line = lines[i];
    const match = line.match(/\/([^\/]+\.ts):(\d+):/);
    if (match) {
      return `${match[1]}:${match[2]}`;
    }
  }
  return "unknown";
}
function wrapQueueForTracking(queue) {
  const originalSubmit = queue.submit.bind(queue);
  queue.submit = function(commandBuffers) {
    const start = TRACK_SUBMITS ? performance.now() : 0;
    const result = originalSubmit(commandBuffers);
    trackSubmit();
    if (!TRACK_SUBMITS) {
      return result;
    }
    const duration = performance.now() - start;
    recordSubmit(duration, extractSourceFromStack());
    return result;
  };
  return queue;
}
var TRACK_SUBMITS, submitCount, submitTimes, totalSubmitMs, maxSubmitMs, minSubmitMs, submitSources, currentPhase, phaseStats;
var init_submit_tracker = __esm({
  "src/gpu/submit-tracker.ts"() {
    init_perf_guards();
    init_debug();
    TRACK_SUBMITS = false;
    submitCount = 0;
    submitTimes = [];
    totalSubmitMs = 0;
    maxSubmitMs = 0;
    minSubmitMs = Infinity;
    submitSources = /* @__PURE__ */ new Map();
    currentPhase = "other";
    phaseStats = {
      prefill: { count: 0, times: [], totalMs: 0, maxMs: 0, minMs: Infinity, sources: /* @__PURE__ */ new Map() },
      decode: { count: 0, times: [], totalMs: 0, maxMs: 0, minMs: Infinity, sources: /* @__PURE__ */ new Map() },
      other: { count: 0, times: [], totalMs: 0, maxMs: 0, minMs: Infinity, sources: /* @__PURE__ */ new Map() }
    };
  }
});

// src/config/platforms/loader.js
var loader_exports = {};
__export(loader_exports, {
  clearPlatformCache: () => clearPlatformCache,
  detectPlatform: () => detectPlatform,
  getBufferAlignment: () => getBufferAlignment,
  getCapabilities: () => getCapabilities,
  getKernelOverride: () => getKernelOverride,
  getMemoryHints: () => getMemoryHints,
  getPlatform: () => getPlatform,
  getPreferredVariant: () => getPreferredVariant,
  getResolvedPlatformConfig: () => getResolvedPlatformConfig,
  getWgslOverrides: () => getWgslOverrides,
  getWorkgroupOverride: () => getWorkgroupOverride,
  initializePlatform: () => initializePlatform,
  prefersUnifiedMemory: () => prefersUnifiedMemory,
  setPlatformsBaseUrl: () => setPlatformsBaseUrl,
  shouldAvoidVariant: () => shouldAvoidVariant
});
function setPlatformsBaseUrl(baseUrl) {
  platformsBaseUrl = baseUrl;
  platformCache.clear();
  currentPlatform = null;
}
async function loadPlatformConfig(platformId) {
  if (platformCache.has(platformId)) {
    return platformCache.get(platformId) || null;
  }
  const baseUrl = platformsBaseUrl || new URL("./", import.meta.url).href;
  const url = `${baseUrl}${platformId}.json`;
  try {
    const response = await fetch(url);
    if (!response.ok) {
      return null;
    }
    const config2 = await response.json();
    platformCache.set(platformId, config2);
    return config2;
  } catch {
    return null;
  }
}
async function detectPlatform(adapterInfo) {
  const vendor = adapterInfo.vendor?.toLowerCase() || "";
  const architecture = adapterInfo.architecture?.toLowerCase() || "";
  const device2 = adapterInfo.device?.toLowerCase() || "";
  const description = adapterInfo.description?.toLowerCase() || "";
  for (const platformId of PLATFORM_FILES) {
    const config2 = await loadPlatformConfig(platformId);
    if (!config2)
      continue;
    const detection = config2.detection;
    let matches = true;
    if (detection.vendor && !vendor.includes(detection.vendor.toLowerCase())) {
      matches = false;
    }
    if (detection.architecture && !architecture.includes(detection.architecture.toLowerCase())) {
      matches = false;
    }
    if (detection.device && !device2.includes(detection.device.toLowerCase())) {
      matches = false;
    }
    if (detection.description && !description.includes(detection.description.toLowerCase())) {
      matches = false;
    }
    if (matches && !config2.isGeneric) {
      currentPlatform = config2;
      return config2;
    }
  }
  const genericConfig = await loadPlatformConfig("generic");
  if (genericConfig) {
    currentPlatform = genericConfig;
    return genericConfig;
  }
  const fallback = {
    id: "unknown",
    name: "Unknown Platform",
    detection: {},
    isGeneric: true
  };
  currentPlatform = fallback;
  return fallback;
}
async function initializePlatform(adapter) {
  const adapterInfo = adapter.info;
  const platform = await detectPlatform(adapterInfo);
  const features = adapter.features;
  currentCapabilities = {
    hasF16: features.has("shader-f16"),
    hasSubgroups: features.has("subgroups"),
    subgroupSize: features.has("subgroups") ? 32 : void 0,
    // TODO: detect actual size
    maxWorkgroupSize: adapter.limits.maxComputeWorkgroupSizeX,
    maxSharedMemory: adapter.limits.maxComputeWorkgroupStorageSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize
  };
  return {
    platform,
    capabilities: currentCapabilities
  };
}
function getPlatform() {
  if (!currentPlatform) {
    throw new Error("Platform not initialized. Call initializePlatform() first.");
  }
  return currentPlatform;
}
function getCapabilities() {
  if (!currentCapabilities) {
    throw new Error("Platform not initialized. Call initializePlatform() first.");
  }
  return currentCapabilities;
}
function getKernelOverride(operation) {
  const platform = getPlatform();
  return platform.kernelOverrides?.[operation];
}
function getPreferredVariant(operation) {
  return getKernelOverride(operation)?.preferred;
}
function shouldAvoidVariant(operation, variant) {
  const override = getKernelOverride(operation);
  return override?.avoid?.includes(variant) ?? false;
}
function getWorkgroupOverride(operation, variant) {
  const override = getKernelOverride(operation);
  return override?.workgroupOverrides?.[variant];
}
function getWgslOverrides(operation, variant) {
  const override = getKernelOverride(operation);
  return override?.wgslOverrides?.[variant];
}
function getMemoryHints() {
  return getPlatform().memoryHints;
}
function prefersUnifiedMemory() {
  return getMemoryHints()?.preferUnifiedMemory ?? false;
}
function getBufferAlignment() {
  return getMemoryHints()?.bufferAlignment ?? 256;
}
function clearPlatformCache() {
  platformCache.clear();
  currentPlatform = null;
  currentCapabilities = null;
}
function getResolvedPlatformConfig() {
  return {
    platform: getPlatform(),
    capabilities: getCapabilities()
  };
}
var currentPlatform, currentCapabilities, platformCache, platformsBaseUrl, PLATFORM_FILES;
var init_loader = __esm({
  "src/config/platforms/loader.js"() {
    currentPlatform = null;
    currentCapabilities = null;
    platformCache = /* @__PURE__ */ new Map();
    platformsBaseUrl = null;
    PLATFORM_FILES = [
      "apple-m3",
      "apple-m2",
      "apple-m1",
      "nvidia-rtx40",
      "nvidia-rtx30",
      "amd-rdna3",
      "generic"
      // Fallback
    ];
  }
});

// src/config/kernels/registry.js
var registry_exports = {};
__export(registry_exports, {
  clearRegistryCache: () => clearRegistryCache,
  getAvailableVariants: () => getAvailableVariants,
  getOperation: () => getOperation,
  getRegistry: () => getRegistry,
  getRegistrySync: () => getRegistrySync,
  getVariant: () => getVariant,
  getVariantNames: () => getVariantNames,
  isVariantAvailable: () => isVariantAvailable,
  mergeBindings: () => mergeBindings2,
  resolveKernelConfig: () => resolveKernelConfig2,
  setRegistryUrl: () => setRegistryUrl
});
function setRegistryUrl(url) {
  registryUrl = url;
  cachedRegistry = null;
}
async function getRegistry() {
  if (cachedRegistry) {
    return cachedRegistry;
  }
  const url = registryUrl || new URL("./registry.json", import.meta.url).href;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load kernel registry from ${url}: ${response.status}`);
  }
  cachedRegistry = await response.json();
  return cachedRegistry;
}
function getRegistrySync() {
  if (!cachedRegistry) {
    throw new Error("Kernel registry not loaded. Call await getRegistry() first.");
  }
  return cachedRegistry;
}
function clearRegistryCache() {
  cachedRegistry = null;
}
function getOperation(operation) {
  const registry = getRegistrySync();
  return registry.operations[operation];
}
function getVariant(operation, variant) {
  const op = getOperation(operation);
  return op?.variants[variant];
}
function getVariantNames(operation) {
  const op = getOperation(operation);
  return op ? Object.keys(op.variants) : [];
}
function isVariantAvailable(operation, variant, capabilities) {
  const variantSchema = getVariant(operation, variant);
  if (!variantSchema)
    return false;
  const requires = variantSchema.requires || [];
  for (const req of requires) {
    if (req === "shader-f16" && !capabilities.hasF16)
      return false;
    if (req === "subgroups" && !capabilities.hasSubgroups)
      return false;
    if (req === "subgroups-f16" && (!capabilities.hasSubgroups || !capabilities.hasF16))
      return false;
  }
  return true;
}
function getAvailableVariants(operation, capabilities) {
  return getVariantNames(operation).filter((v) => isVariantAvailable(operation, v, capabilities));
}
function mergeBindings2(base, override) {
  if (!override || override.length === 0) {
    return [...base];
  }
  const result = [...base];
  for (const binding of override) {
    const existingIdx = result.findIndex((b) => b.index === binding.index);
    if (existingIdx >= 0) {
      result[existingIdx] = binding;
    } else {
      result.push(binding);
    }
  }
  return result.sort((a, b) => a.index - b.index);
}
function resolveKernelConfig2(operation, variant) {
  const opSchema = getOperation(operation);
  const variantSchema = getVariant(operation, variant);
  if (!opSchema || !variantSchema) {
    return null;
  }
  return {
    operation,
    variant,
    wgsl: variantSchema.wgsl,
    entryPoint: variantSchema.entryPoint,
    workgroup: variantSchema.workgroup,
    requires: variantSchema.requires ?? [],
    bindings: mergeBindings2(opSchema.baseBindings, variantSchema.bindingsOverride),
    uniforms: variantSchema.uniformsOverride ?? opSchema.baseUniforms,
    wgslOverrides: variantSchema.wgslOverrides ?? {},
    sharedMemory: variantSchema.sharedMemory ?? 0
  };
}
var cachedRegistry, registryUrl;
var init_registry = __esm({
  "src/config/kernels/registry.js"() {
    cachedRegistry = null;
    registryUrl = null;
  }
});

// src/gpu/device.ts
function isWebGPUAvailable() {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}
async function requestAdapter(options = {}) {
  if (!isWebGPUAvailable()) {
    return null;
  }
  const adapterOptions = [
    { powerPreference: "high-performance", ...options },
    { powerPreference: "low-power", ...options },
    { ...options }
    // Default
  ];
  for (const opts of adapterOptions) {
    try {
      const adapter = await navigator.gpu.requestAdapter(opts);
      if (adapter) {
        return adapter;
      }
    } catch (e) {
    }
  }
  return null;
}
function detectFeatures(adapter) {
  const available = /* @__PURE__ */ new Set();
  for (const feature of adapter.features) {
    available.add(feature);
  }
  return available;
}
function buildFeatureRequests(available) {
  const requested = [];
  if (available.has(FEATURES.SHADER_F16)) {
    requested.push(FEATURES.SHADER_F16);
  }
  if (available.has(FEATURES.SUBGROUPS)) {
    requested.push(FEATURES.SUBGROUPS);
  }
  if (available.has(FEATURES.SUBGROUPS_F16)) {
    requested.push(FEATURES.SUBGROUPS_F16);
  }
  if (available.has(FEATURES.TIMESTAMP_QUERY)) {
    requested.push(FEATURES.TIMESTAMP_QUERY);
  }
  return requested;
}
function buildLimits(adapter) {
  const adapterLimits = adapter.limits;
  return {
    // Request maximum available storage buffer size (critical for large weight tensors)
    maxStorageBufferBindingSize: adapterLimits.maxStorageBufferBindingSize,
    // Request maximum buffer size
    maxBufferSize: adapterLimits.maxBufferSize,
    // Request maximum compute workgroup sizes
    maxComputeWorkgroupSizeX: adapterLimits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: adapterLimits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: adapterLimits.maxComputeWorkgroupSizeZ,
    maxComputeInvocationsPerWorkgroup: adapterLimits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupStorageSize: adapterLimits.maxComputeWorkgroupStorageSize,
    // Binding limits
    maxStorageBuffersPerShaderStage: adapterLimits.maxStorageBuffersPerShaderStage,
    maxUniformBufferBindingSize: adapterLimits.maxUniformBufferBindingSize
  };
}
async function initializePlatformAndRegistry(adapter) {
  if (platformInitialized) {
    return;
  }
  platformInitialized = true;
  try {
    const [platformLoader, kernelRegistry] = await Promise.all([
      Promise.resolve().then(() => (init_loader(), loader_exports)),
      Promise.resolve().then(() => (init_registry(), registry_exports))
    ]);
    resolvedPlatformConfig = await platformLoader.initializePlatform(adapter);
    await kernelRegistry.getRegistry();
    log.debug("GPU", "Platform: " + resolvedPlatformConfig.platform.name + " (" + resolvedPlatformConfig.platform.id + ")");
    log.debug("GPU", "Capabilities: f16=" + resolvedPlatformConfig.capabilities.hasF16 + ", subgroups=" + resolvedPlatformConfig.capabilities.hasSubgroups);
  } catch (e) {
    log.warn("GPU", "Platform/registry init failed (non-fatal): " + e.message);
    resolvedPlatformConfig = null;
  }
}
async function initDevice() {
  if (gpuDevice2) {
    return gpuDevice2;
  }
  if (!isWebGPUAvailable()) {
    throw new Error("WebGPU is not available in this browser");
  }
  const adapter = await requestAdapter();
  if (!adapter) {
    throw new Error("Failed to get WebGPU adapter");
  }
  await initializePlatformAndRegistry(adapter);
  const availableFeatures = detectFeatures(adapter);
  const requestedFeatures = buildFeatureRequests(availableFeatures);
  const limits = buildLimits(adapter);
  const adapterInfo = adapter.info || { vendor: "unknown", architecture: "unknown", device: "unknown", description: "" };
  try {
    gpuDevice2 = await adapter.requestDevice({
      requiredFeatures: requestedFeatures,
      requiredLimits: limits
    });
  } catch (e) {
    log.warn("GPU", "Failed to request device with features, trying minimal config: " + e.message);
    gpuDevice2 = await adapter.requestDevice();
  }
  if (!gpuDevice2) {
    throw new Error("Failed to create WebGPU device");
  }
  gpuDevice2.lost.then((info) => {
    log.error("GPU", "Device lost: " + info.message + ", Reason: " + info.reason);
    gpuDevice2 = null;
    kernelCapabilities = null;
    resolvedPlatformConfig = null;
    platformInitialized = false;
  });
  wrapQueueForTracking(gpuDevice2.queue);
  kernelCapabilities = {
    hasSubgroups: gpuDevice2.features.has(FEATURES.SUBGROUPS),
    hasSubgroupsF16: gpuDevice2.features.has(FEATURES.SUBGROUPS_F16),
    hasF16: gpuDevice2.features.has(FEATURES.SHADER_F16),
    hasTimestampQuery: gpuDevice2.features.has(FEATURES.TIMESTAMP_QUERY),
    maxBufferSize: gpuDevice2.limits.maxStorageBufferBindingSize,
    maxWorkgroupSize: gpuDevice2.limits.maxComputeInvocationsPerWorkgroup,
    maxWorkgroupStorageSize: gpuDevice2.limits.maxComputeWorkgroupStorageSize,
    adapterInfo: {
      vendor: adapterInfo.vendor || "unknown",
      architecture: adapterInfo.architecture || "unknown",
      device: adapterInfo.device || "unknown",
      description: adapterInfo.description || ""
    }
  };
  const features = [
    kernelCapabilities.hasF16 && "f16",
    kernelCapabilities.hasSubgroups && "subgroups"
  ].filter(Boolean).join("/") || "basic";
  console.log("[GPU] " + (adapterInfo.vendor || "unknown") + " " + (adapterInfo.architecture || adapterInfo.device || "") + ", " + features + ", " + (kernelCapabilities.maxBufferSize / (1024 * 1024 * 1024)).toFixed(1) + "GB");
  return gpuDevice2;
}
function getKernelCapabilities() {
  if (!kernelCapabilities) {
    throw new Error("Device not initialized. Call initDevice() first.");
  }
  return { ...kernelCapabilities };
}
function getDevice() {
  return gpuDevice2;
}
function hasFeature(feature) {
  if (!gpuDevice2) {
    return false;
  }
  return gpuDevice2.features.has(feature);
}
function getDeviceLimits() {
  if (!gpuDevice2) {
    return null;
  }
  return {
    maxStorageBufferBindingSize: gpuDevice2.limits.maxStorageBufferBindingSize,
    maxBufferSize: gpuDevice2.limits.maxBufferSize,
    maxComputeWorkgroupSizeX: gpuDevice2.limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: gpuDevice2.limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: gpuDevice2.limits.maxComputeWorkgroupSizeZ,
    maxComputeInvocationsPerWorkgroup: gpuDevice2.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupStorageSize: gpuDevice2.limits.maxComputeWorkgroupStorageSize,
    maxStorageBuffersPerShaderStage: gpuDevice2.limits.maxStorageBuffersPerShaderStage,
    maxUniformBufferBindingSize: gpuDevice2.limits.maxUniformBufferBindingSize,
    maxComputeWorkgroupsPerDimension: gpuDevice2.limits.maxComputeWorkgroupsPerDimension
  };
}
var gpuDevice2, kernelCapabilities, resolvedPlatformConfig, platformInitialized, FEATURES;
var init_device = __esm({
  "src/gpu/device.ts"() {
    init_submit_tracker();
    init_debug();
    gpuDevice2 = null;
    kernelCapabilities = null;
    resolvedPlatformConfig = null;
    platformInitialized = false;
    FEATURES = {
      SHADER_F16: "shader-f16",
      SUBGROUPS: "subgroups",
      SUBGROUPS_F16: "subgroups-f16",
      TIMESTAMP_QUERY: "timestamp-query"
    };
  }
});

// src/gpu/tensor.ts
function createTensor(buffer, dtype, shape, label) {
  return {
    buffer,
    dtype,
    shape: Object.freeze([...shape]),
    label
  };
}
function dtypeBytes(dtype) {
  return dtype === "f16" ? 2 : 4;
}
function inferOutputDtype(a, b) {
  return a.dtype === "f16" && b.dtype === "f16" ? "f16" : "f32";
}
var init_tensor = __esm({
  "src/gpu/tensor.ts"() {
  }
});

// src/gpu/profiler.ts
var GPUProfiler;
var init_profiler = __esm({
  "src/gpu/profiler.ts"() {
    init_device();
    init_perf_guards();
    init_debug();
    GPUProfiler = class {
      device;
      hasTimestampQuery;
      // Query set for timestamp queries (if supported)
      querySet = null;
      queryBuffer = null;
      readbackBuffer = null;
      queryCapacity = 256;
      // Max number of timestamp pairs
      // Tracking state
      activeLabels = /* @__PURE__ */ new Map();
      nextQueryIndex = 0;
      pendingResolves = [];
      // Results storage
      results = /* @__PURE__ */ new Map();
      // CPU fallback timing
      cpuTimings = /* @__PURE__ */ new Map();
      /**
       * @param device - WebGPU device (uses global if not provided)
       */
      constructor(device2 = null) {
        this.device = device2 || getDevice();
        this.hasTimestampQuery = this.device?.features?.has(FEATURES.TIMESTAMP_QUERY) ?? false;
        if (this.hasTimestampQuery && this.device) {
          this._initQueryResources();
        }
      }
      /**
       * Initialize GPU query resources
       * @private
       */
      _initQueryResources() {
        if (!this.device)
          return;
        try {
          this.querySet = this.device.createQuerySet({
            type: "timestamp",
            count: this.queryCapacity * 2
            // Start and end for each measurement
          });
          this.queryBuffer = this.device.createBuffer({
            size: this.queryCapacity * 2 * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
          });
          this.readbackBuffer = this.device.createBuffer({
            size: this.queryCapacity * 2 * 8,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
          });
        } catch (e) {
          log.warn("GPUProfiler", `Failed to create timestamp query resources: ${e}`);
          this.hasTimestampQuery = false;
        }
      }
      /**
       * Begin timing a labeled region.
       * Uses CPU timing; use writeTimestamp() inside passes for GPU timestamps.
       * @param label - Unique label for this measurement
       */
      begin(label) {
        if (this.activeLabels.has(label)) {
          log.warn("GPUProfiler", `Label "${label}" already active`);
          return;
        }
        const startTime = performance.now();
        this.activeLabels.set(label, {
          cpuStartTime: startTime
        });
      }
      /**
       * End timing a labeled region
       * @param label - Label started with begin()
       */
      end(label) {
        const active = this.activeLabels.get(label);
        if (!active) {
          log.warn("GPUProfiler", `No active measurement for label "${label}"`);
          return;
        }
        const endTime = performance.now();
        this.activeLabels.delete(label);
        if (this.hasTimestampQuery && "startQueryIndex" in active) {
          this.pendingResolves.push({
            label,
            startIndex: active.startQueryIndex,
            endIndex: active.startQueryIndex + 1,
            cpuStartTime: active.cpuStartTime,
            cpuEndTime: endTime
          });
        } else {
          this._recordResult(label, endTime - active.cpuStartTime);
        }
      }
      /**
       * Write timestamp to query set within a compute pass
       * Call this instead of begin/end when inside a pass
       * @param pass - Compute pass encoder
       * @param label - Label for this measurement
       * @param isEnd - true for end timestamp
       */
      writeTimestamp(pass, label, isEnd = false) {
        if (!this.hasTimestampQuery || !this.querySet)
          return;
        let queryIndex;
        if (!isEnd) {
          queryIndex = this.nextQueryIndex;
          this.nextQueryIndex += 2;
          this.activeLabels.set(label, {
            startQueryIndex: queryIndex,
            cpuStartTime: performance.now()
          });
        } else {
          const active = this.activeLabels.get(label);
          if (!active || !("startQueryIndex" in active))
            return;
          queryIndex = active.startQueryIndex + 1;
          this.activeLabels.delete(label);
          this.pendingResolves.push({
            label,
            startIndex: active.startQueryIndex,
            endIndex: queryIndex,
            cpuStartTime: active.cpuStartTime,
            cpuEndTime: performance.now()
          });
        }
        pass.writeTimestamp(this.querySet, queryIndex);
      }
      /**
       * Resolve pending timestamp queries and update results
       * Call this after command buffer submission
       */
      async resolve() {
        if (!this.hasTimestampQuery || this.pendingResolves.length === 0) {
          return;
        }
        if (!this.device || !this.querySet || !this.queryBuffer || !this.readbackBuffer) {
          log.warn("GPUProfiler", "Missing required resources for resolve");
          return;
        }
        const encoder = this.device.createCommandEncoder();
        const maxIndex = Math.max(...this.pendingResolves.map((p) => p.endIndex)) + 1;
        encoder.resolveQuerySet(this.querySet, 0, maxIndex, this.queryBuffer, 0);
        encoder.copyBufferToBuffer(
          this.queryBuffer,
          0,
          this.readbackBuffer,
          0,
          maxIndex * 8
        );
        this.device.queue.submit([encoder.finish()]);
        if (!allowReadback("GPUProfiler.resolve")) {
          return;
        }
        await this.readbackBuffer.mapAsync(GPUMapMode.READ);
        const timestamps = new BigUint64Array(this.readbackBuffer.getMappedRange());
        for (const pending of this.pendingResolves) {
          const startNs = timestamps[pending.startIndex];
          const endNs = timestamps[pending.endIndex];
          const durationMs = Number(endNs - startNs) / 1e6;
          if (durationMs < 0 || durationMs > 6e4) {
            this._recordResult(pending.label, pending.cpuEndTime - pending.cpuStartTime);
          } else {
            this._recordResult(pending.label, durationMs);
          }
        }
        this.readbackBuffer.unmap();
        this.pendingResolves = [];
        this.nextQueryIndex = 0;
      }
      /**
       * Record a timing result
       * @private
       */
      _recordResult(label, timeMs) {
        if (!this.results.has(label)) {
          this.results.set(label, {
            times: [],
            min: Infinity,
            max: -Infinity,
            sum: 0,
            count: 0
          });
        }
        const result = this.results.get(label);
        result.times.push(timeMs);
        result.min = Math.min(result.min, timeMs);
        result.max = Math.max(result.max, timeMs);
        result.sum += timeMs;
        result.count++;
        if (result.times.length > 100) {
          const removed = result.times.shift();
          result.sum -= removed;
          result.count--;
          if (result.times.length % 20 === 0) {
            result.min = Math.min(...result.times);
            result.max = Math.max(...result.times);
          }
        }
      }
      /**
       * Get profiling results
       */
      getResults() {
        const output = {};
        for (const [label, data] of this.results) {
          output[label] = {
            avg: data.sum / data.count,
            min: data.min,
            max: data.max,
            count: data.count,
            total: data.sum
          };
        }
        return output;
      }
      /**
       * Get result for a specific label
       * @param label - Label to get result for
       */
      getResult(label) {
        const data = this.results.get(label);
        if (!data)
          return null;
        return {
          avg: data.sum / data.count,
          min: data.min,
          max: data.max,
          count: data.count,
          total: data.sum
        };
      }
      /**
       * Reset all profiling data
       */
      reset() {
        this.results.clear();
        this.activeLabels.clear();
        this.pendingResolves = [];
        this.nextQueryIndex = 0;
      }
      /**
       * Get formatted report string
       */
      getReport() {
        const results = this.getResults();
        const labels = Object.keys(results).sort();
        if (labels.length === 0) {
          return "No profiling data collected";
        }
        let report = "GPU Profiler Results\n";
        report += "\u2500".repeat(60) + "\n";
        report += "Label".padEnd(30) + "Avg (ms)".padStart(10) + "Min".padStart(10) + "Max".padStart(10) + "\n";
        report += "\u2500".repeat(60) + "\n";
        for (const label of labels) {
          const r = results[label];
          report += label.padEnd(30);
          report += r.avg.toFixed(3).padStart(10);
          report += r.min.toFixed(3).padStart(10);
          report += r.max.toFixed(3).padStart(10);
          report += "\n";
        }
        return report;
      }
      /**
       * Check if timestamp queries are available
       */
      isGPUTimingAvailable() {
        return this.hasTimestampQuery;
      }
      /**
       * Destroy profiler resources
       */
      destroy() {
        if (this.querySet) {
          this.querySet.destroy();
          this.querySet = null;
        }
        if (this.queryBuffer) {
          this.queryBuffer.destroy();
          this.queryBuffer = null;
        }
        if (this.readbackBuffer) {
          this.readbackBuffer.destroy();
          this.readbackBuffer = null;
        }
        this.results.clear();
        this.activeLabels.clear();
      }
    };
  }
});

// src/gpu/kernel-tuner.ts
function getTunerConfig() {
  return getRuntimeConfig().tuner;
}
async function getKernelTuner() {
  if (!globalTuner) {
    globalTuner = new KernelTuner();
    await globalTuner.init();
  }
  return globalTuner;
}
var KernelTuner, globalTuner;
var init_kernel_tuner = __esm({
  "src/gpu/kernel-tuner.ts"() {
    init_device();
    init_profiler();
    init_debug();
    init_runtime();
    KernelTuner = class {
      device;
      profiler;
      limits;
      capabilities;
      cache;
      constructor() {
        this.device = null;
        this.profiler = null;
        this.limits = null;
        this.capabilities = null;
        this.cache = /* @__PURE__ */ new Map();
      }
      /**
       * Initialize the tuner
       */
      async init() {
        this.device = getDevice();
        if (!this.device) {
          throw new Error("GPU device not initialized");
        }
        this.profiler = new GPUProfiler(this.device);
        this.limits = getDeviceLimits();
        this.capabilities = getKernelCapabilities();
        this._loadCache();
      }
      /**
       * Get device signature for cache key
       * @private
       */
      _getDeviceSignature() {
        const info = this.capabilities?.adapterInfo || { vendor: "", architecture: "", device: "" };
        return `${info.vendor}_${info.architecture}_${info.device}`.replace(/[^a-zA-Z0-9]/g, "_");
      }
      /**
       * Load cached tuning results from localStorage
       * @private
       */
      _loadCache() {
        if (typeof localStorage === "undefined")
          return;
        const signature = this._getDeviceSignature();
        const cacheKey = getTunerConfig().cacheKeyPrefix + signature;
        try {
          const cached = localStorage.getItem(cacheKey);
          if (cached) {
            const data = JSON.parse(cached);
            this.cache = new Map(Object.entries(data));
          }
        } catch (e) {
          log.warn("KernelTuner", `Failed to load cache: ${e}`);
        }
      }
      /**
       * Save cached results to localStorage
       * @private
       */
      _saveCache() {
        if (typeof localStorage === "undefined")
          return;
        const signature = this._getDeviceSignature();
        const cacheKey = getTunerConfig().cacheKeyPrefix + signature;
        try {
          const data = Object.fromEntries(this.cache);
          localStorage.setItem(cacheKey, JSON.stringify(data));
        } catch (e) {
          log.warn("KernelTuner", `Failed to save cache: ${e}`);
        }
      }
      /**
       * Generate workgroup size candidates based on device limits
       * @private
       */
      _generateWorkgroupCandidates() {
        const maxX = this.limits?.maxComputeWorkgroupSizeX || 256;
        const maxY = this.limits?.maxComputeWorkgroupSizeY || 256;
        const maxInvocations = this.limits?.maxComputeInvocationsPerWorkgroup || 256;
        const candidates = [];
        for (const x of [64, 128, 256, 512]) {
          if (x <= maxX && x <= maxInvocations) {
            candidates.push([x, 1, 1]);
          }
        }
        for (const x of [8, 16, 32]) {
          for (const y of [8, 16, 32]) {
            if (x <= maxX && y <= maxY && x * y <= maxInvocations) {
              candidates.push([x, y, 1]);
            }
          }
        }
        return candidates;
      }
      /**
       * Tune a kernel by running benchmarks
       * @param kernelName - Name of kernel to tune
       * @param inputSizes - Input dimensions for tuning
       * @param options - Tuning options
       * @returns Promise resolving to tuning result
       */
      async tuneKernel(kernelName, inputSizes, options = {}) {
        const {
          warmup = getTunerConfig().defaultWarmupIterations,
          iterations = getTunerConfig().defaultTimedIterations,
          forceRetune = false
        } = options;
        const cacheKey = `${kernelName}_${JSON.stringify(inputSizes)}`;
        if (!forceRetune && this.cache.has(cacheKey)) {
          return this.cache.get(cacheKey);
        }
        const candidates = this._generateWorkgroupCandidates();
        let bestResult;
        switch (kernelName) {
          case "matmul":
            bestResult = await this._tuneMatmul(inputSizes, candidates, warmup, iterations);
            break;
          case "attention":
            bestResult = await this._tuneAttention(inputSizes, candidates, warmup, iterations);
            break;
          case "softmax":
            bestResult = await this._tuneSoftmax(inputSizes, candidates, warmup, iterations);
            break;
          case "rmsnorm":
            bestResult = await this._tuneRMSNorm(inputSizes, candidates, warmup, iterations);
            break;
          case "dequant":
            bestResult = await this._tuneDequant(inputSizes, candidates, warmup, iterations);
            break;
          default:
            bestResult = await this._tuneGeneric(kernelName, inputSizes, candidates, warmup, iterations);
        }
        this.cache.set(cacheKey, bestResult);
        this._saveCache();
        return bestResult;
      }
      /**
       * Tune matmul kernel
       * @private
       */
      async _tuneMatmul(inputSizes, candidates, warmup, iterations) {
        const { M = 1024, N = 1024, K = 1024 } = inputSizes;
        const matmulCandidates = candidates.filter((c) => c[1] > 1);
        let best = {
          optimalWorkgroupSize: [16, 16, 1],
          optimalTileSize: 16,
          throughput: 0,
          timeMs: Infinity,
          deviceInfo: this.capabilities?.adapterInfo
        };
        if (!this.device) {
          return best;
        }
        const bufferA = this.device.createBuffer({
          size: M * K * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const bufferB = this.device.createBuffer({
          size: K * N * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const bufferC = this.device.createBuffer({
          size: M * N * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const dataA = new Float32Array(M * K);
        const dataB = new Float32Array(K * N);
        for (let i = 0; i < dataA.length; i++)
          dataA[i] = Math.random();
        for (let i = 0; i < dataB.length; i++)
          dataB[i] = Math.random();
        this.device.queue.writeBuffer(bufferA, 0, dataA);
        this.device.queue.writeBuffer(bufferB, 0, dataB);
        for (const [wgX, wgY] of matmulCandidates) {
          try {
            const shader = this._createMatmulShader(wgX, wgY);
            const pipeline = await this._createComputePipeline(shader, "main");
            const uniformBuffer = this.device.createBuffer({
              size: 16,
              usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            const uniformData = new Uint32Array([M, N, K, 0]);
            this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);
            const bindGroup = this.device.createBindGroup({
              layout: pipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufferA } },
                { binding: 2, resource: { buffer: bufferB } },
                { binding: 3, resource: { buffer: bufferC } }
              ]
            });
            for (let i = 0; i < warmup; i++) {
              const encoder = this.device.createCommandEncoder();
              const pass = encoder.beginComputePass();
              pass.setPipeline(pipeline);
              pass.setBindGroup(0, bindGroup);
              pass.dispatchWorkgroups(Math.ceil(M / wgX), Math.ceil(N / wgY));
              pass.end();
              this.device.queue.submit([encoder.finish()]);
            }
            await this.device.queue.onSubmittedWorkDone();
            const times = [];
            for (let i = 0; i < iterations; i++) {
              const start = performance.now();
              const encoder = this.device.createCommandEncoder();
              const pass = encoder.beginComputePass();
              pass.setPipeline(pipeline);
              pass.setBindGroup(0, bindGroup);
              pass.dispatchWorkgroups(Math.ceil(M / wgX), Math.ceil(N / wgY));
              pass.end();
              this.device.queue.submit([encoder.finish()]);
              await this.device.queue.onSubmittedWorkDone();
              times.push(performance.now() - start);
            }
            const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
            const flops = 2 * M * N * K;
            const gflops = flops / avgTime / 1e6;
            if (avgTime < best.timeMs) {
              best = {
                optimalWorkgroupSize: [wgX, wgY, 1],
                optimalTileSize: wgX,
                throughput: gflops,
                timeMs: avgTime,
                deviceInfo: this.capabilities?.adapterInfo
              };
            }
            uniformBuffer.destroy();
          } catch (e) {
            continue;
          }
        }
        bufferA.destroy();
        bufferB.destroy();
        bufferC.destroy();
        return best;
      }
      /**
       * Create matmul shader with specified workgroup size
       * @private
       */
      _createMatmulShader(wgX, wgY) {
        return `
struct Uniforms {
    M: u32, N: u32, K: u32, _pad: u32,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(${wgX}, ${wgY}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= uniforms.M || col >= uniforms.N) { return; }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < uniforms.K; k = k + 1u) {
        sum = sum + A[row * uniforms.K + k] * B[k * uniforms.N + col];
    }
    C[row * uniforms.N + col] = sum;
}`;
      }
      /**
       * Tune attention kernel
       * @private
       */
      async _tuneAttention(inputSizes, candidates, warmup, iterations) {
        const { seqLen = 2048, numHeads = 32, headDim = 128 } = inputSizes;
        let best = {
          optimalWorkgroupSize: [64, 1, 1],
          optimalTileSize: 64,
          throughput: 0,
          timeMs: Infinity,
          deviceInfo: this.capabilities?.adapterInfo
        };
        if (!this.device) {
          return best;
        }
        const attentionCandidates = candidates.filter((c) => c[1] === 1);
        if (attentionCandidates.length === 0) {
          return best;
        }
        const maxElements = 2e6;
        const totalHeadsRaw = Math.max(1, seqLen * numHeads);
        let benchSeqLen = seqLen;
        let totalHeads = totalHeadsRaw;
        let totalElements = totalHeads * headDim;
        if (totalElements > maxElements) {
          benchSeqLen = Math.max(1, Math.floor(maxElements / (numHeads * headDim)));
          totalHeads = Math.max(1, benchSeqLen * numHeads);
          totalElements = totalHeads * headDim;
        }
        const bufferQ = this.device.createBuffer({
          size: totalElements * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const bufferK = this.device.createBuffer({
          size: totalElements * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const bufferOut = this.device.createBuffer({
          size: totalHeads * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const dataQ = new Float32Array(totalElements);
        const dataK = new Float32Array(totalElements);
        for (let i = 0; i < totalElements; i++) {
          dataQ[i] = Math.random();
          dataK[i] = Math.random();
        }
        this.device.queue.writeBuffer(bufferQ, 0, dataQ);
        this.device.queue.writeBuffer(bufferK, 0, dataK);
        for (const [wgX] of attentionCandidates) {
          try {
            const shader = this._createAttentionShader(wgX);
            const pipeline = await this._createComputePipeline(shader, "main");
            const uniformBuffer = this.device.createBuffer({
              size: 16,
              usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            const uniformData = new Uint32Array([headDim, numHeads, benchSeqLen, 0]);
            this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);
            const bindGroup = this.device.createBindGroup({
              layout: pipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufferQ } },
                { binding: 2, resource: { buffer: bufferK } },
                { binding: 3, resource: { buffer: bufferOut } }
              ]
            });
            const avgTime = await this._benchmarkPipeline(
              pipeline,
              bindGroup,
              [totalHeads, 1, 1],
              warmup,
              iterations
            );
            const flops = 2 * totalHeads * headDim;
            const gflops = avgTime > 0 ? flops / avgTime / 1e6 : 0;
            if (avgTime < best.timeMs) {
              best = {
                optimalWorkgroupSize: [wgX, 1, 1],
                optimalTileSize: wgX,
                throughput: gflops,
                timeMs: avgTime,
                deviceInfo: this.capabilities?.adapterInfo
              };
            }
            uniformBuffer.destroy();
          } catch (e) {
            continue;
          }
        }
        bufferQ.destroy();
        bufferK.destroy();
        bufferOut.destroy();
        return best;
      }
      /**
       * Tune softmax kernel
       * @private
       */
      async _tuneSoftmax(inputSizes, candidates, warmup, iterations) {
        const { innerSize = 32e3, outerSize = 1 } = inputSizes;
        let best = {
          optimalWorkgroupSize: [256, 1, 1],
          optimalTileSize: 256,
          throughput: 0,
          timeMs: Infinity,
          deviceInfo: this.capabilities?.adapterInfo
        };
        if (!this.device) {
          return best;
        }
        const softmaxCandidates = candidates.filter((c) => c[1] === 1);
        if (softmaxCandidates.length === 0) {
          return best;
        }
        const totalElements = Math.max(1, innerSize * outerSize);
        const bufferIn = this.device.createBuffer({
          size: totalElements * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const bufferOut = this.device.createBuffer({
          size: totalElements * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const dataIn = new Float32Array(totalElements);
        for (let i = 0; i < totalElements; i++) {
          dataIn[i] = Math.random();
        }
        this.device.queue.writeBuffer(bufferIn, 0, dataIn);
        for (const [wgX] of softmaxCandidates) {
          try {
            const shader = this._createSoftmaxShader(wgX);
            const pipeline = await this._createComputePipeline(shader, "main");
            const uniformBuffer = this.device.createBuffer({
              size: 16,
              usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            const uniformData = new Uint32Array([innerSize, outerSize, 0, 0]);
            this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);
            const bindGroup = this.device.createBindGroup({
              layout: pipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufferIn } },
                { binding: 2, resource: { buffer: bufferOut } }
              ]
            });
            const avgTime = await this._benchmarkPipeline(
              pipeline,
              bindGroup,
              [outerSize, 1, 1],
              warmup,
              iterations
            );
            const ops = 2 * totalElements;
            const gops = avgTime > 0 ? ops / avgTime / 1e6 : 0;
            if (avgTime < best.timeMs) {
              best = {
                optimalWorkgroupSize: [wgX, 1, 1],
                optimalTileSize: wgX,
                throughput: gops,
                timeMs: avgTime,
                deviceInfo: this.capabilities?.adapterInfo
              };
            }
            uniformBuffer.destroy();
          } catch (e) {
            continue;
          }
        }
        bufferIn.destroy();
        bufferOut.destroy();
        return best;
      }
      /**
       * Tune RMSNorm kernel
       * @private
       */
      async _tuneRMSNorm(inputSizes, candidates, warmup, iterations) {
        const { hiddenSize = 4096, numTokens = 1 } = inputSizes;
        let best = {
          optimalWorkgroupSize: [256, 1, 1],
          optimalTileSize: 256,
          throughput: 0,
          timeMs: Infinity,
          deviceInfo: this.capabilities?.adapterInfo
        };
        if (!this.device) {
          return best;
        }
        const rmsCandidates = candidates.filter((c) => c[1] === 1);
        if (rmsCandidates.length === 0) {
          return best;
        }
        const totalElements = Math.max(1, hiddenSize * numTokens);
        const bufferIn = this.device.createBuffer({
          size: totalElements * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const bufferWeight = this.device.createBuffer({
          size: hiddenSize * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const bufferOut = this.device.createBuffer({
          size: totalElements * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const dataIn = new Float32Array(totalElements);
        const dataWeight = new Float32Array(hiddenSize);
        for (let i = 0; i < totalElements; i++) {
          dataIn[i] = Math.random();
        }
        for (let i = 0; i < hiddenSize; i++) {
          dataWeight[i] = Math.random();
        }
        this.device.queue.writeBuffer(bufferIn, 0, dataIn);
        this.device.queue.writeBuffer(bufferWeight, 0, dataWeight);
        for (const [wgX] of rmsCandidates) {
          try {
            const shader = this._createRMSNormShader(wgX);
            const pipeline = await this._createComputePipeline(shader, "main");
            const uniformBuffer = this.device.createBuffer({
              size: 16,
              usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            const uniformData = new ArrayBuffer(16);
            const uniformView = new DataView(uniformData);
            uniformView.setUint32(0, hiddenSize, true);
            uniformView.setUint32(4, numTokens, true);
            uniformView.setFloat32(8, 1e-5, true);
            uniformView.setUint32(12, 0, true);
            this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);
            const bindGroup = this.device.createBindGroup({
              layout: pipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufferIn } },
                { binding: 2, resource: { buffer: bufferWeight } },
                { binding: 3, resource: { buffer: bufferOut } }
              ]
            });
            const avgTime = await this._benchmarkPipeline(
              pipeline,
              bindGroup,
              [numTokens, 1, 1],
              warmup,
              iterations
            );
            const ops = 2 * totalElements;
            const gops = avgTime > 0 ? ops / avgTime / 1e6 : 0;
            if (avgTime < best.timeMs) {
              best = {
                optimalWorkgroupSize: [wgX, 1, 1],
                optimalTileSize: wgX,
                throughput: gops,
                timeMs: avgTime,
                deviceInfo: this.capabilities?.adapterInfo
              };
            }
            uniformBuffer.destroy();
          } catch (e) {
            continue;
          }
        }
        bufferIn.destroy();
        bufferWeight.destroy();
        bufferOut.destroy();
        return best;
      }
      /**
       * Tune dequantization kernel
       * @private
       */
      async _tuneDequant(inputSizes, candidates, warmup, iterations) {
        const { numBlocks = 1e3 } = inputSizes;
        let best = {
          optimalWorkgroupSize: [64, 1, 1],
          optimalTileSize: 64,
          throughput: 0,
          timeMs: Infinity,
          deviceInfo: this.capabilities?.adapterInfo
        };
        if (!this.device) {
          return best;
        }
        const dequantCandidates = candidates.filter((c) => c[1] === 1);
        if (dequantCandidates.length === 0) {
          return best;
        }
        const numElements = Math.max(1, numBlocks * 256);
        const bufferIn = this.device.createBuffer({
          size: numElements * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const bufferOut = this.device.createBuffer({
          size: numElements * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const dataIn = new Uint32Array(numElements);
        for (let i = 0; i < numElements; i++) {
          dataIn[i] = i & 65535;
        }
        this.device.queue.writeBuffer(bufferIn, 0, dataIn);
        for (const [wgX] of dequantCandidates) {
          try {
            const shader = this._createDequantShader(wgX);
            const pipeline = await this._createComputePipeline(shader, "main");
            const uniformBuffer = this.device.createBuffer({
              size: 16,
              usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            const uniformData = new ArrayBuffer(16);
            const uniformView = new DataView(uniformData);
            uniformView.setUint32(0, numElements, true);
            uniformView.setFloat32(4, 0.01, true);
            uniformView.setUint32(8, 0, true);
            uniformView.setUint32(12, 0, true);
            this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);
            const bindGroup = this.device.createBindGroup({
              layout: pipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufferIn } },
                { binding: 2, resource: { buffer: bufferOut } }
              ]
            });
            const workgroups = Math.ceil(numElements / wgX);
            const avgTime = await this._benchmarkPipeline(
              pipeline,
              bindGroup,
              [workgroups, 1, 1],
              warmup,
              iterations
            );
            const ops = numElements;
            const gops = avgTime > 0 ? ops / avgTime / 1e6 : 0;
            if (avgTime < best.timeMs) {
              best = {
                optimalWorkgroupSize: [wgX, 1, 1],
                optimalTileSize: wgX,
                throughput: gops,
                timeMs: avgTime,
                deviceInfo: this.capabilities?.adapterInfo
              };
            }
            uniformBuffer.destroy();
          } catch (e) {
            continue;
          }
        }
        bufferIn.destroy();
        bufferOut.destroy();
        return best;
      }
      /**
       * Generic tuning for unknown kernels
       * @private
       */
      async _tuneGeneric(kernelName, inputSizes, candidates, warmup, iterations) {
        return {
          optimalWorkgroupSize: [256, 1, 1],
          optimalTileSize: 256,
          throughput: 0,
          timeMs: 0,
          deviceInfo: this.capabilities?.adapterInfo
        };
      }
      async _benchmarkPipeline(pipeline, bindGroup, workgroups, warmup, iterations) {
        if (!this.device) {
          return Infinity;
        }
        const [wgX, wgY, wgZ] = workgroups;
        if (wgX === 0 || wgY === 0 || wgZ === 0) {
          return Infinity;
        }
        for (let i = 0; i < warmup; i++) {
          const encoder = this.device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(wgX, wgY, wgZ);
          pass.end();
          this.device.queue.submit([encoder.finish()]);
        }
        await this.device.queue.onSubmittedWorkDone();
        const times = [];
        for (let i = 0; i < iterations; i++) {
          const start = performance.now();
          const encoder = this.device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(wgX, wgY, wgZ);
          pass.end();
          this.device.queue.submit([encoder.finish()]);
          await this.device.queue.onSubmittedWorkDone();
          times.push(performance.now() - start);
        }
        return times.reduce((a, b) => a + b, 0) / times.length;
      }
      /**
       * Create compute pipeline from shader source
       * @private
       */
      async _createComputePipeline(shaderSource, entryPoint) {
        if (!this.device) {
          throw new Error("Device not initialized");
        }
        const module = this.device.createShaderModule({ code: shaderSource });
        return await this.device.createComputePipelineAsync({
          layout: "auto",
          compute: { module, entryPoint }
        });
      }
      _createAttentionShader(wgSize) {
        return `
const WG_SIZE: u32 = ${wgSize}u;

struct Uniforms {
  headDim: u32,
  numHeads: u32,
  seqLen: u32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;

var<workgroup> shared: array<f32, WG_SIZE>;

@compute @workgroup_size(${wgSize}, 1, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let totalHeads = uniforms.numHeads * uniforms.seqLen;
  let idx = wg_id.x;
  if (idx >= totalHeads) { return; }

  let headDim = uniforms.headDim;
  let offset = idx * headDim;
  let lane = local_id.x;

  var sum: f32 = 0.0;
  var i: u32 = lane;
  loop {
    if (i >= headDim) { break; }
    sum = sum + Q[offset + i] * K[offset + i];
    i = i + WG_SIZE;
  }

  shared[lane] = sum;
  workgroupBarrier();

  var stride: u32 = WG_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      shared[lane] = shared[lane] + shared[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (lane == 0u) {
    Out[idx] = shared[0];
  }
}`;
      }
      _createSoftmaxShader(wgSize) {
        return `
const WG_SIZE: u32 = ${wgSize}u;

struct Uniforms {
  innerSize: u32,
  outerSize: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared: array<f32, WG_SIZE>;

@compute @workgroup_size(${wgSize}, 1, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let row = wg_id.x;
  if (row >= uniforms.outerSize) { return; }

  let inner = uniforms.innerSize;
  let lane = local_id.x;
  let offset = row * inner;

  var localMax: f32 = -3.402823e+38;
  var i: u32 = lane;
  loop {
    if (i >= inner) { break; }
    localMax = max(localMax, input[offset + i]);
    i = i + WG_SIZE;
  }

  shared[lane] = localMax;
  workgroupBarrier();

  var stride: u32 = WG_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      shared[lane] = max(shared[lane], shared[lane + stride]);
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  let rowMax = shared[0];
  var localSum: f32 = 0.0;
  i = lane;
  loop {
    if (i >= inner) { break; }
    localSum = localSum + exp(input[offset + i] - rowMax);
    i = i + WG_SIZE;
  }

  shared[lane] = localSum;
  workgroupBarrier();

  stride = WG_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      shared[lane] = shared[lane] + shared[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  let denom = shared[0];
  i = lane;
  loop {
    if (i >= inner) { break; }
    output[offset + i] = exp(input[offset + i] - rowMax) / denom;
    i = i + WG_SIZE;
  }
}`;
      }
      _createRMSNormShader(wgSize) {
        return `
const WG_SIZE: u32 = ${wgSize}u;

struct Uniforms {
  hiddenSize: u32,
  numTokens: u32,
  eps: f32,
  _pad0: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

var<workgroup> shared: array<f32, WG_SIZE>;

@compute @workgroup_size(${wgSize}, 1, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let tokenIdx = wg_id.x;
  if (tokenIdx >= uniforms.numTokens) { return; }

  let size = uniforms.hiddenSize;
  let base = tokenIdx * size;
  let lane = local_id.x;

  var localSumSq: f32 = 0.0;
  var i: u32 = lane;
  loop {
    if (i >= size) { break; }
    let x = input[base + i];
    localSumSq = localSumSq + x * x;
    i = i + WG_SIZE;
  }

  shared[lane] = localSumSq;
  workgroupBarrier();

  var stride: u32 = WG_SIZE / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lane < stride) {
      shared[lane] = shared[lane] + shared[lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  let invRms = 1.0 / sqrt(shared[0] / f32(size) + uniforms.eps);
  i = lane;
  loop {
    if (i >= size) { break; }
    output[base + i] = input[base + i] * invRms * weight[i];
    i = i + WG_SIZE;
  }
}`;
      }
      _createDequantShader(wgSize) {
        return `
const WG_SIZE: u32 = ${wgSize}u;

struct Uniforms {
  count: u32,
  scale: f32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${wgSize}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uniforms.count) { return; }
  output[idx] = f32(input[idx]) * uniforms.scale;
}`;
      }
      /**
       * Get cached tuning result
       * @param kernelName - Kernel name
       * @param inputSizes - Input sizes
       * @returns Cached result or null
       */
      getCachedResult(kernelName, inputSizes) {
        const cacheKey = `${kernelName}_${JSON.stringify(inputSizes)}`;
        return this.cache.get(cacheKey) || null;
      }
      /**
       * Clear all cached results
       */
      clearCache() {
        this.cache.clear();
        if (typeof localStorage !== "undefined") {
          const signature = this._getDeviceSignature();
          localStorage.removeItem(getTunerConfig().cacheKeyPrefix + signature);
        }
      }
      /**
       * Get all cached results
       * @returns Object with all cached results
       */
      getAllCachedResults() {
        return Object.fromEntries(this.cache);
      }
      /**
       * Destroy tuner resources
       */
      destroy() {
        if (this.profiler) {
          this.profiler.destroy();
        }
      }
    };
    globalTuner = null;
  }
});

// src/gpu/uniform-cache.ts
function hashArrayBuffer(data) {
  const view = new Uint8Array(data);
  let hash = 2166136261;
  for (let i = 0; i < view.length; i++) {
    hash ^= view[i];
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}
function releaseUniformBuffer(buffer) {
  const cache = getUniformCache();
  if (cache.isCached(buffer)) {
    cache.release(buffer);
  } else {
    buffer.destroy();
  }
}
function getUniformCache() {
  if (!globalUniformCache) {
    globalUniformCache = new UniformBufferCache();
  }
  return globalUniformCache;
}
var UniformBufferCache, globalUniformCache;
var init_uniform_cache = __esm({
  "src/gpu/uniform-cache.ts"() {
    init_device();
    init_runtime();
    UniformBufferCache = class {
      cache = /* @__PURE__ */ new Map();
      stats = {
        hits: 0,
        misses: 0,
        evictions: 0,
        currentSize: 0
      };
      /** Buffers evicted from cache, awaiting destruction after GPU work completes */
      pendingDestruction = [];
      maxEntries;
      maxAgeMs;
      constructor(maxEntries = getRuntimeConfig().gpuCache.uniformCacheMaxEntries, maxAgeMs = getRuntimeConfig().gpuCache.uniformCacheMaxAgeMs) {
        this.maxEntries = maxEntries;
        this.maxAgeMs = maxAgeMs;
      }
      /**
       * Get or create a uniform buffer with the given contents.
       * Returns a cached buffer if one exists with identical data.
       */
      getOrCreate(data, label) {
        const hash = hashArrayBuffer(data);
        const existing = this.cache.get(hash);
        if (existing) {
          existing.lastUsed = performance.now();
          existing.refCount++;
          this.stats.hits++;
          return existing.buffer;
        }
        this.stats.misses++;
        const device2 = getDevice();
        if (!device2) {
          throw new Error("GPU device not initialized");
        }
        const buffer = device2.createBuffer({
          label: `${label}_cached`,
          size: data.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        device2.queue.writeBuffer(buffer, 0, data);
        if (this.cache.size >= this.maxEntries) {
          this.evictLRU();
        }
        this.cache.set(hash, {
          buffer,
          lastUsed: performance.now(),
          refCount: 1
        });
        this.stats.currentSize = this.cache.size;
        return buffer;
      }
      /**
       * Release a reference to a cached buffer.
       * Buffer is NOT destroyed - it stays in cache for reuse.
       * Call this instead of buffer.destroy() for cached uniforms.
       */
      release(buffer) {
        for (const [hash, entry] of this.cache) {
          if (entry.buffer === buffer) {
            entry.refCount = Math.max(0, entry.refCount - 1);
            return;
          }
        }
      }
      /**
       * Evict least recently used entry.
       * IMPORTANT: Buffer is NOT destroyed immediately - it's queued for deferred
       * destruction to avoid use-after-destroy bugs with pending command buffers.
       */
      evictLRU() {
        let oldestHash = null;
        let oldestTime = Infinity;
        for (const [hash, entry] of this.cache) {
          if (entry.refCount === 0 && entry.lastUsed < oldestTime) {
            oldestTime = entry.lastUsed;
            oldestHash = hash;
          }
        }
        if (oldestHash === null) {
          for (const [hash, entry] of this.cache) {
            if (entry.lastUsed < oldestTime) {
              oldestTime = entry.lastUsed;
              oldestHash = hash;
            }
          }
        }
        if (oldestHash) {
          const entry = this.cache.get(oldestHash);
          if (entry) {
            this.pendingDestruction.push(entry.buffer);
            this.cache.delete(oldestHash);
            this.stats.evictions++;
            this.stats.currentSize = this.cache.size;
          }
        }
      }
      /**
       * Evict stale entries (older than maxAgeMs).
       * Buffers are queued for deferred destruction.
       */
      evictStale() {
        const now = performance.now();
        let evicted = 0;
        for (const [hash, entry] of this.cache) {
          if (entry.refCount === 0 && now - entry.lastUsed > this.maxAgeMs) {
            this.pendingDestruction.push(entry.buffer);
            this.cache.delete(hash);
            evicted++;
          }
        }
        this.stats.evictions += evicted;
        this.stats.currentSize = this.cache.size;
        return evicted;
      }
      /**
       * Clear all cached buffers.
       * Also flushes any pending destruction queue.
       */
      clear() {
        this.flushPendingDestruction();
        for (const entry of this.cache.values()) {
          entry.buffer.destroy();
        }
        this.cache.clear();
        this.stats.currentSize = 0;
      }
      /**
       * Destroy all buffers in the pending destruction queue.
       * Call this after GPU work completes (e.g., after onSubmittedWorkDone).
       *
       * This is critical for avoiding use-after-destroy bugs: when the uniform
       * cache evicts a buffer that's still referenced by a pending command buffer,
       * the buffer is queued here instead of being destroyed immediately.
       */
      flushPendingDestruction() {
        const count = this.pendingDestruction.length;
        for (const buffer of this.pendingDestruction) {
          buffer.destroy();
        }
        this.pendingDestruction = [];
        return count;
      }
      /**
       * Get the number of buffers pending destruction.
       */
      getPendingDestructionCount() {
        return this.pendingDestruction.length;
      }
      /**
       * Check if a buffer is managed by this cache
       */
      isCached(buffer) {
        for (const entry of this.cache.values()) {
          if (entry.buffer === buffer) {
            return true;
          }
        }
        return false;
      }
      /**
       * Get cache statistics
       */
      getStats() {
        const total = this.stats.hits + this.stats.misses;
        const hitRate = total > 0 ? (this.stats.hits / total * 100).toFixed(1) + "%" : "0%";
        return { ...this.stats, hitRate, pendingDestruction: this.pendingDestruction.length };
      }
    };
    globalUniformCache = null;
  }
});

// src/gpu/kernels/utils.ts
function getKernelBasePath() {
  if (typeof location !== "undefined") {
    const path = location.pathname;
    if (path.startsWith("/d") || path.startsWith("/doppler/") || location.host.includes("replo")) {
      return "/doppler/gpu/kernels";
    }
  }
  return "/gpu/kernels";
}
function validateAttentionLimits(seqLen, numHeads, headDim) {
  const limits = getDeviceLimits();
  if (!limits)
    return;
  const workgroupInvocations = seqLen * numHeads;
  if (workgroupInvocations > limits.maxComputeWorkgroupsPerDimension) {
    throw new Error(
      `Attention parameters exceed device limits: ${workgroupInvocations} workgroups > ${limits.maxComputeWorkgroupsPerDimension} max per dimension. Try reducing seqLen (${seqLen}) or numHeads (${numHeads}).`
    );
  }
  const kvCacheSize = seqLen * numHeads * headDim * 4;
  if (kvCacheSize > limits.maxStorageBufferBindingSize) {
    throw new Error(
      `KV cache size ${(kvCacheSize / 1e9).toFixed(2)}GB exceeds device limit ${(limits.maxStorageBufferBindingSize / 1e9).toFixed(2)}GB. Reduce sequence length or use paged attention.`
    );
  }
  const tileSize = 64;
  const sharedMemRequired = tileSize * headDim * 4 * 2;
  if (sharedMemRequired > limits.maxComputeWorkgroupStorageSize) {
    log.warn("KernelSelector", `Attention may be slow: tile requires ${sharedMemRequired} bytes but device has ${limits.maxComputeWorkgroupStorageSize} bytes shared memory.`);
  }
}
async function loadShaderSource(filename) {
  if (shaderSourceCache.has(filename)) {
    return shaderSourceCache.get(filename);
  }
  const url = `${KERNEL_BASE_PATH}/${filename}`;
  try {
    const response = await fetch(url, { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`Failed to load shader ${filename}: ${response.status}`);
    }
    const source = await response.text();
    shaderSourceCache.set(filename, source);
    return source;
  } catch (error) {
    log.error("KernelSelector", `Failed to load shader ${filename}: ${error}`);
    throw error;
  }
}
function hasRequiredFeatures(required, capabilities) {
  for (const feature of required) {
    if (feature === "shader-f16" && !capabilities.hasF16)
      return false;
    if (feature === "subgroups" && !capabilities.hasSubgroups)
      return false;
    if (feature === "subgroups-f16" && !capabilities.hasSubgroups)
      return false;
  }
  return true;
}
function getKernelConfig(operation, variant) {
  const config2 = KERNEL_CONFIGS[operation]?.[variant];
  if (!config2) {
    throw new Error(`Unknown kernel: ${operation}/${variant}`);
  }
  return config2;
}
async function compileShader(device2, source, label) {
  const module = device2.createShaderModule({
    label,
    code: source
  });
  const compilationInfo = await module.getCompilationInfo();
  if (compilationInfo.messages.length > 0) {
    for (const msg of compilationInfo.messages) {
      if (msg.type === "error") {
        log.error("compileShader", `${label}: ${msg.message} (line ${msg.lineNum}:${msg.linePos})`);
      } else if (msg.type === "warning") {
        log.warn("compileShader", `${label}: ${msg.message} (line ${msg.lineNum}:${msg.linePos})`);
      } else {
        log.debug("compileShader", `${label}: ${msg.message} (line ${msg.lineNum}:${msg.linePos})`);
      }
    }
    if (compilationInfo.messages.some((m) => m.type === "error")) {
      throw new Error(`Shader compilation failed for ${label}`);
    }
  }
  return module;
}
async function getShaderModule(device2, shaderFile, label) {
  const cacheKey = shaderFile;
  const cached = shaderModuleCache.get(cacheKey);
  if (cached) {
    return cached;
  }
  const compilePromise = (async () => {
    const shaderSource = await loadShaderSource(shaderFile);
    return compileShader(device2, shaderSource, label);
  })();
  shaderModuleCache.set(cacheKey, compilePromise);
  try {
    return await compilePromise;
  } catch (err) {
    shaderModuleCache.delete(cacheKey);
    throw err;
  }
}
function getOrCreateBindGroupLayout(label, entries, deviceOverride = null) {
  const cached = bindGroupLayoutCache.get(label);
  if (cached) {
    return cached;
  }
  const device2 = deviceOverride || getDevice();
  if (!device2) {
    throw new Error("Device not initialized");
  }
  const layout = device2.createBindGroupLayout({ label, entries });
  bindGroupLayoutCache.set(label, layout);
  return layout;
}
function getOrCreatePipelineLayout(label, bindGroupLayouts, deviceOverride = null) {
  const cached = pipelineLayoutCache.get(label);
  if (cached) {
    return cached;
  }
  const device2 = deviceOverride || getDevice();
  if (!device2) {
    throw new Error("Device not initialized");
  }
  const layout = device2.createPipelineLayout({
    label,
    bindGroupLayouts
  });
  pipelineLayoutCache.set(label, layout);
  return layout;
}
function getCachedPipeline(operation, variant) {
  const cacheKey = `${operation}:${variant}`;
  return pipelineCache.get(cacheKey) || null;
}
async function getPipelineFast(operation, variant, bindGroupLayout = null) {
  const cached = getCachedPipeline(operation, variant);
  if (cached) {
    return cached;
  }
  return createPipeline(operation, variant, bindGroupLayout);
}
async function createPipeline(operation, variant, bindGroupLayout = null) {
  const cacheKey = `${operation}:${variant}`;
  if (pipelineCache.has(cacheKey)) {
    return pipelineCache.get(cacheKey);
  }
  const device2 = getDevice();
  if (!device2) {
    throw new Error("Device not initialized");
  }
  const config2 = getKernelConfig(operation, variant);
  const capabilities = getKernelCapabilities();
  if (!hasRequiredFeatures(config2.requires, capabilities)) {
    throw new Error(
      `Kernel ${operation}/${variant} requires features: ${config2.requires.join(", ")}`
    );
  }
  trace.kernels(
    `KernelLayout: ${operation}/${variant} file=${config2.shaderFile} entry=${config2.entryPoint} workgroup=[${config2.workgroupSize.join(",")}] requires=${config2.requires.length > 0 ? config2.requires.join("|") : "none"}`
  );
  const shaderModule = await getShaderModule(device2, config2.shaderFile, `${operation}_${variant}`);
  const layoutLabel = bindGroupLayout?.label || `${operation}_${variant}_layout`;
  const pipelineDescriptor = {
    label: `${operation}_${variant}_pipeline`,
    layout: bindGroupLayout ? getOrCreatePipelineLayout(layoutLabel, [bindGroupLayout], device2) : "auto",
    compute: {
      module: shaderModule,
      entryPoint: config2.entryPoint
    }
  };
  const pipeline = await device2.createComputePipelineAsync(pipelineDescriptor);
  pipelineCache.set(cacheKey, pipeline);
  return pipeline;
}
function clearKernelCaches() {
  pipelineCache.clear();
  shaderSourceCache.clear();
  shaderModuleCache.clear();
  bindGroupLayoutCache.clear();
  pipelineLayoutCache.clear();
}
function clearPipelineCache() {
  clearKernelCaches();
}
function getCacheStats() {
  return {
    pipelines: pipelineCache.size,
    shaders: shaderSourceCache.size,
    shaderModules: shaderModuleCache.size,
    bindGroupLayouts: bindGroupLayoutCache.size,
    pipelineLayouts: pipelineLayoutCache.size
  };
}
async function getTunedWorkgroupSize(operation, inputSizes = {}) {
  try {
    const tuner = await getKernelTuner();
    const result = tuner.getCachedResult(operation, inputSizes);
    if (result) {
      return result.optimalWorkgroupSize;
    }
    const tuneResult = await tuner.tuneKernel(operation, inputSizes);
    return tuneResult.optimalWorkgroupSize;
  } catch (e) {
    log.warn("KernelSelector", `Tuning failed for ${operation}, using defaults: ${e.message}`);
    switch (operation) {
      case "matmul":
        return [16, 16, 1];
      case "attention":
      case "rmsnorm":
      case "softmax":
        return [256, 1, 1];
      case "dequant":
        return [64, 1, 1];
      default:
        return [256, 1, 1];
    }
  }
}
async function autoTuneKernels(modelConfig = {}) {
  const {
    hiddenSize = 4096,
    intermediateSize = 14336,
    numHeads = 32,
    headDim = 128,
    maxSeqLen = 4096,
    vocabSize = 32e3
  } = modelConfig;
  const tuner = await getKernelTuner();
  const results = {};
  results.matmul_hidden = await tuner.tuneKernel("matmul", {
    M: 1,
    N: hiddenSize,
    K: hiddenSize
  });
  results.matmul_ffn = await tuner.tuneKernel("matmul", {
    M: 1,
    N: intermediateSize,
    K: hiddenSize
  });
  results.attention = await tuner.tuneKernel("attention", {
    seqLen: 1,
    numHeads,
    headDim
  });
  results.softmax = await tuner.tuneKernel("softmax", {
    innerSize: vocabSize,
    outerSize: 1
  });
  results.rmsnorm = await tuner.tuneKernel("rmsnorm", {
    hiddenSize,
    numTokens: 1
  });
  results.dequant = await tuner.tuneKernel("dequant", {
    numBlocks: 1e3
  });
  log.debug("KernelSelector", `Auto-tuning complete: ${JSON.stringify(results)}`);
  return results;
}
async function prewarmKernels(options = {}) {
  const caps = getKernelCapabilities();
  const mode = options.mode ?? "parallel";
  const entries = Object.entries(KERNEL_CONFIGS).sort(([a], [b]) => a.localeCompare(b)).map(([operation, variants]) => [operation, Object.entries(variants).sort(([a], [b]) => a.localeCompare(b))]);
  if (mode === "sequential") {
    let count = 0;
    for (const [operation, variants] of entries) {
      for (const [variant, cfg] of variants) {
        if (cfg.requires && !hasRequiredFeatures(cfg.requires, caps)) {
          continue;
        }
        try {
          await createPipeline(operation, variant);
          count += 1;
        } catch (e) {
          log.warn("KernelSelector", `Prewarm failed for ${operation}/${variant}: ${e.message}`);
        }
      }
    }
    log.debug("KernelSelector", `Prewarmed ${count} kernel pipelines`);
    return;
  }
  const jobs = [];
  for (const [operation, variants] of entries) {
    for (const [variant, cfg] of variants) {
      if (cfg.requires && !hasRequiredFeatures(cfg.requires, caps)) {
        continue;
      }
      jobs.push(
        createPipeline(operation, variant).then(() => {
        }).catch((e) => {
          log.warn("KernelSelector", `Prewarm failed for ${operation}/${variant}: ${e.message}`);
        })
      );
    }
  }
  await Promise.all(jobs);
  log.debug("KernelSelector", `Prewarmed ${jobs.length} kernel pipelines`);
}
function createUniformBufferFromData(label, data, recorder, deviceOverride, options) {
  if (recorder) {
    return recorder.createUniformBuffer(data, label);
  }
  const arrayBuffer = data instanceof ArrayBuffer ? data : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  const useCache = options?.useCache ?? true;
  if (useCache && !deviceOverride) {
    return getUniformCache().getOrCreate(arrayBuffer, label);
  }
  const device2 = deviceOverride ?? getDevice();
  if (!device2) {
    throw new Error("GPU device not initialized");
  }
  const byteLength = arrayBuffer.byteLength;
  const buffer = device2.createBuffer({
    label,
    size: byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  device2.queue.writeBuffer(buffer, 0, arrayBuffer);
  return buffer;
}
function createUniformBufferWithView(label, byteLength, writer, recorder, deviceOverride) {
  const data = new ArrayBuffer(byteLength);
  const view = new DataView(data);
  writer(view);
  return createUniformBufferFromData(label, data, recorder, deviceOverride);
}
var shaderSourceCache, shaderModuleCache, pipelineCache, bindGroupLayoutCache, pipelineLayoutCache, KERNEL_BASE_PATH, KERNEL_CONFIGS;
var init_utils = __esm({
  "src/gpu/kernels/utils.ts"() {
    init_device();
    init_kernel_tuner();
    init_uniform_cache();
    init_debug();
    shaderSourceCache = /* @__PURE__ */ new Map();
    shaderModuleCache = /* @__PURE__ */ new Map();
    pipelineCache = /* @__PURE__ */ new Map();
    bindGroupLayoutCache = /* @__PURE__ */ new Map();
    pipelineLayoutCache = /* @__PURE__ */ new Map();
    KERNEL_BASE_PATH = getKernelBasePath();
    KERNEL_CONFIGS = {
      matmul: {
        f16: {
          shaderFile: "matmul_f16.wgsl",
          entryPoint: "main",
          workgroupSize: [16, 16, 1],
          requires: ["shader-f16"],
          outputDtype: "f16"
        },
        f16_vec4: {
          shaderFile: "matmul_f16.wgsl",
          entryPoint: "main_vec4",
          workgroupSize: [16, 16, 1],
          requires: ["shader-f16"],
          outputDtype: "f16"
        },
        f16w_f32a: {
          shaderFile: "matmul_f16w_f32a.wgsl",
          entryPoint: "main",
          workgroupSize: [16, 16, 1],
          requires: ["shader-f16"]
        },
        // Optimized GEMV for M=1 decode: uses shared memory for A vector
        gemv: {
          shaderFile: "matmul_gemv.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        },
        // Subgroup-optimized GEMV - 1.5x faster using subgroupAdd
        gemv_subgroup: {
          shaderFile: "matmul_gemv_subgroup.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16", "subgroups"]
        },
        gemv_subgroup_vec4: {
          shaderFile: "matmul_gemv_subgroup.wgsl",
          entryPoint: "main_vec4",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16", "subgroups"]
        },
        // Multi-column GEMV for large vocab (LM head F16) - 32 columns per workgroup
        // Reduces workgroups from 65K to 8K for vocab=262144
        gemv_subgroup_multicol: {
          shaderFile: "matmul_gemv_subgroup.wgsl",
          entryPoint: "main_multicol",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16", "subgroups"],
          variantMetadata: { colsPerWg: 32 }
        },
        // Fused Q4_K dequant + matmul - 2-3x faster (no separate dequant pass)
        q4_fused: {
          shaderFile: "fused_matmul_q4.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["subgroups"]
        },
        q4_fused_batched: {
          shaderFile: "fused_matmul_q4_batched.wgsl",
          entryPoint: "main_batched",
          workgroupSize: [64, 4, 1],
          requires: ["subgroups"],
          variantMetadata: { tileM: 4 }
        },
        // Multi-column GEMV for large vocab (LM head) - 32 columns per workgroup
        q4_fused_multicol: {
          shaderFile: "fused_matmul_q4.wgsl",
          entryPoint: "main_multicol",
          workgroupSize: [256, 1, 1],
          requires: ["subgroups"],
          variantMetadata: { colsPerWg: 32 }
        },
        // F16 output variants - same as above but output to f16 buffer
        q4_fused_multicol_f16: {
          shaderFile: "fused_matmul_q4_multicol_f16.wgsl",
          entryPoint: "main_multicol_f16",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16", "subgroups"],
          outputDtype: "f16",
          variantMetadata: { colsPerWg: 32 }
        },
        q4_fused_batched_f16: {
          shaderFile: "fused_matmul_q4_batched_f16.wgsl",
          entryPoint: "main_batched_f16",
          workgroupSize: [64, 4, 1],
          requires: ["shader-f16", "subgroups"],
          outputDtype: "f16",
          variantMetadata: { tileM: 4 }
        },
        f32: {
          shaderFile: "matmul_f32.wgsl",
          entryPoint: "main",
          workgroupSize: [16, 16, 1],
          requires: []
        }
      },
      // Fused FFN kernels (Tier 2 P0)
      fused_ffn: {
        default: {
          shaderFile: "fused_ffn.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["subgroups"]
        },
        multi: {
          shaderFile: "fused_ffn.wgsl",
          entryPoint: "main_multi",
          workgroupSize: [256, 1, 1],
          requires: ["subgroups"]
        },
        f16: {
          shaderFile: "fused_ffn.wgsl",
          entryPoint: "main_f16",
          workgroupSize: [256, 1, 1],
          requires: ["subgroups"]
        },
        batched: {
          shaderFile: "fused_ffn.wgsl",
          entryPoint: "main_batched",
          workgroupSize: [256, 1, 1],
          requires: ["subgroups"]
        },
        q4k: {
          shaderFile: "fused_ffn_q4k.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["subgroups"]
        },
        q4k_batched: {
          shaderFile: "fused_ffn_q4k.wgsl",
          entryPoint: "main_batched",
          workgroupSize: [256, 1, 1],
          requires: ["subgroups"]
        }
      },
      // Optimized attention decode (Tier 2 P0)
      attention_decode_optimized: {
        default: {
          shaderFile: "attention_decode_optimized.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["subgroups"]
        },
        multihead: {
          shaderFile: "attention_decode_optimized.wgsl",
          entryPoint: "main_multihead",
          workgroupSize: [256, 1, 1],
          requires: ["subgroups"]
        },
        f16kv: {
          shaderFile: "attention_decode_optimized.wgsl",
          entryPoint: "main_f16kv",
          workgroupSize: [256, 1, 1],
          requires: ["subgroups"]
        }
      },
      dequant: {
        subgroup: {
          shaderFile: "dequant_subgroup.wgsl",
          entryPoint: "main",
          workgroupSize: [64, 1, 1],
          requires: ["subgroups"]
        },
        subgroup_vec4: {
          shaderFile: "dequant_subgroup.wgsl",
          entryPoint: "main_vec4",
          workgroupSize: [64, 1, 1],
          requires: ["subgroups"]
        },
        subgroup_f16out: {
          shaderFile: "dequant_f16_out.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        },
        subgroup_vec4_f16out: {
          shaderFile: "dequant_f16_out_vec4.wgsl",
          entryPoint: "main_vec4",
          workgroupSize: [64, 1, 1],
          requires: ["shader-f16"]
        },
        shared: {
          shaderFile: "dequant_shared.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        shared_vec4: {
          shaderFile: "dequant_shared_vec4.wgsl",
          entryPoint: "main_vec4",
          workgroupSize: [64, 1, 1],
          requires: []
        },
        shared_f16out: {
          shaderFile: "dequant_f16_out.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        },
        shared_vec4_f16out: {
          shaderFile: "dequant_f16_out_vec4.wgsl",
          entryPoint: "main_vec4",
          workgroupSize: [64, 1, 1],
          requires: ["shader-f16"]
        },
        // MXFP4 dequantization (GPT-OSS)
        mxfp4: {
          shaderFile: "dequant_mxfp4.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        mxfp4_vec4: {
          shaderFile: "dequant_mxfp4_vec4.wgsl",
          entryPoint: "main_vec4",
          workgroupSize: [64, 1, 1],
          requires: []
        },
        mxfp4_expert: {
          shaderFile: "dequant_mxfp4_expert.wgsl",
          entryPoint: "main_expert",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        // Q6_K dequantization (GGUF 6-bit quantization)
        q6k_f16out: {
          shaderFile: "dequant_q6k.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        },
        // Q8_0 dequantization (GGUF 8-bit quantization)
        q8_0_f16out: {
          shaderFile: "dequant_q8_0.wgsl",
          entryPoint: "main",
          workgroupSize: [32, 1, 1],
          requires: ["shader-f16"]
        }
      },
      attention: {
        prefill: {
          shaderFile: "attention.wgsl",
          entryPoint: "main",
          workgroupSize: [64, 1, 1],
          requires: [],
          validate: validateAttentionLimits
        },
        decode: {
          shaderFile: "attention_decode.wgsl",
          entryPoint: "attention_decode",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        prefill_small: {
          shaderFile: "attention_small.wgsl",
          entryPoint: "main",
          workgroupSize: [32, 1, 1],
          requires: [],
          validate: validateAttentionLimits
        },
        decode_small: {
          shaderFile: "attention_small.wgsl",
          entryPoint: "main",
          workgroupSize: [32, 1, 1],
          requires: []
        },
        prefill_streaming: {
          shaderFile: "attention_streaming.wgsl",
          entryPoint: "main",
          workgroupSize: [1, 1, 1],
          requires: [],
          validate: validateAttentionLimits
        },
        decode_streaming: {
          shaderFile: "attention_streaming.wgsl",
          entryPoint: "main",
          workgroupSize: [1, 1, 1],
          requires: []
        },
        prefill_f16kv: {
          shaderFile: "attention_f16kv.wgsl",
          entryPoint: "main",
          workgroupSize: [64, 1, 1],
          requires: ["shader-f16"],
          validate: validateAttentionLimits
        },
        decode_f16kv: {
          shaderFile: "attention_decode_f16kv.wgsl",
          entryPoint: "attention_decode",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        },
        prefill_small_f16kv: {
          shaderFile: "attention_small_f16kv.wgsl",
          entryPoint: "main",
          workgroupSize: [32, 1, 1],
          requires: ["shader-f16"],
          validate: validateAttentionLimits
        },
        decode_small_f16kv: {
          shaderFile: "attention_small_f16kv.wgsl",
          entryPoint: "main",
          workgroupSize: [32, 1, 1],
          requires: ["shader-f16"]
        },
        prefill_streaming_f16kv: {
          shaderFile: "attention_streaming_f16kv.wgsl",
          entryPoint: "main",
          workgroupSize: [1, 1, 1],
          requires: ["shader-f16"],
          validate: validateAttentionLimits
        },
        decode_streaming_f16kv: {
          shaderFile: "attention_streaming_f16kv.wgsl",
          entryPoint: "main",
          workgroupSize: [1, 1, 1],
          requires: ["shader-f16"]
        },
        // Chunked decode kernel - parallelizes headDim for models with few heads (e.g., Gemma 3)
        decode_chunked_f16kv: {
          shaderFile: "attention_decode_chunked_f16kv.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"],
          variantMetadata: { maxKVLen: 2048 }
        },
        // Subgroup-optimized decode kernel - 4 barriers (vs 80), 100% thread utilization
        decode_subgroup: {
          shaderFile: "attention_decode_subgroup.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          // headDim threads per workgroup
          requires: ["subgroups"]
        }
      },
      rmsnorm: {
        default: {
          shaderFile: "rmsnorm.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        small: {
          shaderFile: "rmsnorm.wgsl",
          entryPoint: "main_small",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        cached: {
          shaderFile: "rmsnorm.wgsl",
          entryPoint: "main_cached",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        // Legacy alias for residual (now uses main_cached)
        residual: {
          shaderFile: "rmsnorm.wgsl",
          entryPoint: "main_cached",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        // Subgroup-accelerated variants (3-5x faster reductions)
        subgroup: {
          shaderFile: "rmsnorm.wgsl",
          entryPoint: "main_subgroup",
          workgroupSize: [256, 1, 1],
          requires: ["subgroups"]
        },
        small_subgroup: {
          shaderFile: "rmsnorm.wgsl",
          entryPoint: "main_small_subgroup",
          workgroupSize: [256, 1, 1],
          requires: ["subgroups"]
        },
        // F16 variants for reduced memory bandwidth
        default_f16: {
          shaderFile: "rmsnorm_f16.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        },
        small_f16: {
          shaderFile: "rmsnorm_f16.wgsl",
          entryPoint: "rmsnorm_small_f16",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        }
      },
      // Fused GEMV + RMSNorm for decode (M=1)
      // Combines down projection matmul with RMSNorm in single kernel
      fused_matmul_rmsnorm: {
        default: {
          shaderFile: "fused_matmul_rmsnorm.wgsl",
          entryPoint: "gemv_rmsnorm_medium",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        small: {
          shaderFile: "fused_matmul_rmsnorm.wgsl",
          entryPoint: "gemv_rmsnorm_small",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        // Medium variant for N up to ~4096 (covers Gemma 3's hiddenSize=1152)
        medium: {
          shaderFile: "fused_matmul_rmsnorm.wgsl",
          entryPoint: "gemv_rmsnorm_medium",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        phase1: {
          shaderFile: "fused_matmul_rmsnorm.wgsl",
          entryPoint: "gemv_rmsnorm_phase1",
          workgroupSize: [256, 1, 1],
          requires: []
        }
      },
      // Fused GEMV + Residual for decode (M=1)
      // Combines output projection matmul with residual add in single kernel
      fused_matmul_residual: {
        default: {
          shaderFile: "matmul_gemv_residual.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        }
      },
      softmax: {
        default: {
          shaderFile: "softmax.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        small: {
          shaderFile: "softmax.wgsl",
          entryPoint: "softmax_small",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        online: {
          shaderFile: "softmax.wgsl",
          entryPoint: "softmax_online",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        // Subgroup-accelerated variants (3-5x faster reductions)
        subgroup: {
          shaderFile: "softmax.wgsl",
          entryPoint: "main_subgroup",
          workgroupSize: [256, 1, 1],
          requires: ["subgroups"]
        },
        small_subgroup: {
          shaderFile: "softmax.wgsl",
          entryPoint: "softmax_small_subgroup",
          workgroupSize: [256, 1, 1],
          requires: ["subgroups"]
        }
      },
      rope: {
        default: {
          shaderFile: "rope.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        compute_freqs: {
          shaderFile: "rope.wgsl",
          entryPoint: "rope_compute_freqs",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        qk: {
          shaderFile: "rope.wgsl",
          entryPoint: "rope_qk",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        ntk: {
          shaderFile: "rope.wgsl",
          entryPoint: "rope_ntk_scaled",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        yarn: {
          shaderFile: "rope.wgsl",
          entryPoint: "rope_yarn",
          workgroupSize: [256, 1, 1],
          requires: []
        }
      },
      silu: {
        default: {
          shaderFile: "silu.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        gate: {
          shaderFile: "silu.wgsl",
          entryPoint: "silu_gate",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        gate_split: {
          shaderFile: "silu.wgsl",
          entryPoint: "silu_gate_split",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        vec4: {
          shaderFile: "silu.wgsl",
          entryPoint: "silu_vec4",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        gelu: {
          shaderFile: "silu.wgsl",
          entryPoint: "gelu",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        geglu: {
          shaderFile: "silu.wgsl",
          entryPoint: "geglu",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        gate_rowsplit: {
          shaderFile: "silu.wgsl",
          entryPoint: "silu_gate_rowsplit",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        geglu_rowsplit: {
          shaderFile: "silu.wgsl",
          entryPoint: "geglu_rowsplit",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        // F16 variants for reduced memory bandwidth
        default_f16: {
          shaderFile: "silu_f16.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        },
        gate_f16: {
          shaderFile: "silu_f16.wgsl",
          entryPoint: "silu_gate_f16",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        },
        vec4_f16: {
          shaderFile: "silu_f16.wgsl",
          entryPoint: "silu_vec4_f16",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        },
        gate_rowsplit_f16: {
          shaderFile: "silu_f16.wgsl",
          entryPoint: "silu_gate_rowsplit_f16",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        },
        geglu_rowsplit_f16: {
          shaderFile: "silu_f16.wgsl",
          entryPoint: "geglu_rowsplit_f16",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        }
      },
      scale: {
        default: {
          shaderFile: "scale.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        inplace: {
          shaderFile: "scale.wgsl",
          entryPoint: "main_inplace",
          workgroupSize: [256, 1, 1],
          requires: []
        }
      },
      gather: {
        default: {
          shaderFile: "gather.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        vec4: {
          shaderFile: "gather_vec4.wgsl",
          entryPoint: "gather_vec4",
          workgroupSize: [64, 1, 1],
          requires: []
        },
        // F16 embeddings → F32 output (for weight-tied lm_head optimization)
        f16: {
          shaderFile: "gather_f16.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        },
        f16_vec4: {
          shaderFile: "gather_f16_vec4.wgsl",
          entryPoint: "gather_vec4",
          workgroupSize: [64, 1, 1],
          requires: ["shader-f16"]
        },
        // F32 embeddings → F16 output (for F16 activation mode)
        f16_out: {
          shaderFile: "gather_f16_out.wgsl",
          entryPoint: "gather_f16_out",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"],
          outputDtype: "f16",
          variantMetadata: { outputBinding: 4 }
        },
        vec4_f16_out: {
          shaderFile: "gather_vec4_f16_out.wgsl",
          entryPoint: "gather_vec4_f16_out",
          workgroupSize: [64, 1, 1],
          requires: ["shader-f16"],
          outputDtype: "f16",
          variantMetadata: { outputBinding: 4 }
        },
        // F16 embeddings → F16 output (for F16 activation mode with F16 embeddings)
        f16_f16_out: {
          shaderFile: "gather_f16_f16_out.wgsl",
          entryPoint: "gather_f16_out",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"],
          outputDtype: "f16",
          variantMetadata: { outputBinding: 4 }
        },
        f16_vec4_f16_out: {
          shaderFile: "gather_f16_vec4_f16_out.wgsl",
          entryPoint: "gather_vec4_f16_out",
          workgroupSize: [64, 1, 1],
          requires: ["shader-f16"],
          outputDtype: "f16",
          variantMetadata: { outputBinding: 4 }
        }
      },
      residual: {
        default: {
          shaderFile: "residual.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        vec4: {
          shaderFile: "residual_vec4.wgsl",
          entryPoint: "add_vec4",
          workgroupSize: [64, 1, 1],
          requires: []
        },
        default_f16: {
          shaderFile: "residual_f16.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"],
          outputDtype: "f16"
        },
        vec4_f16: {
          shaderFile: "residual_f16_vec4.wgsl",
          entryPoint: "add_vec4",
          workgroupSize: [64, 1, 1],
          requires: ["shader-f16"],
          outputDtype: "f16"
        }
      },
      topk: {
        default: {
          shaderFile: "topk.wgsl",
          entryPoint: "main",
          workgroupSize: [32, 1, 1],
          requires: []
        },
        small: {
          shaderFile: "topk.wgsl",
          entryPoint: "topk_2_small",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        fused: {
          shaderFile: "topk.wgsl",
          entryPoint: "softmax_topk",
          workgroupSize: [32, 1, 1],
          requires: []
        }
      },
      scatter_add: {
        default: {
          shaderFile: "scatter_add.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        vec4: {
          shaderFile: "scatter_add_vec4.wgsl",
          entryPoint: "scatter_add_vec4",
          workgroupSize: [64, 1, 1],
          requires: []
        },
        dynamic: {
          shaderFile: "scatter_add_dynamic.wgsl",
          entryPoint: "scatter_add_dynamic",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        accumulate: {
          shaderFile: "scatter_add.wgsl",
          entryPoint: "scatter_add_accumulate",
          workgroupSize: [256, 1, 1],
          requires: []
        }
      },
      moe_gather: {
        count: {
          shaderFile: "moe_gather.wgsl",
          entryPoint: "count_and_map",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        gather: {
          shaderFile: "moe_gather.wgsl",
          entryPoint: "gather_tokens",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        gather_vec4: {
          shaderFile: "moe_gather_vec4.wgsl",
          entryPoint: "gather_tokens_vec4",
          workgroupSize: [64, 1, 1],
          requires: []
        },
        single_pass: {
          shaderFile: "moe_gather.wgsl",
          entryPoint: "gather_single_pass",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        sparse: {
          shaderFile: "moe_gather.wgsl",
          entryPoint: "count_and_map",
          workgroupSize: [256, 1, 1],
          requires: []
        }
      },
      swiglu: {
        rowsplit_bias: {
          shaderFile: "fused_swiglu.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        }
      },
      bias_add: {
        default: {
          shaderFile: "bias_add.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        f16: {
          shaderFile: "bias_add_f16.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        }
      },
      cast: {
        f32_to_f16: {
          shaderFile: "cast_f32_to_f16.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        },
        f16_to_f32: {
          shaderFile: "cast_f16_to_f32.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        }
      },
      // Split fused QKV output into separate Q, K, V buffers
      split_qkv: {
        default: {
          shaderFile: "split_qkv.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        }
      },
      sample: {
        argmax: {
          shaderFile: "sample.wgsl",
          entryPoint: "argmax",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        argmax_reduce: {
          shaderFile: "sample.wgsl",
          entryPoint: "argmax_reduce",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        find_topk_phase1: {
          shaderFile: "sample.wgsl",
          entryPoint: "find_topk_phase1",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        find_topk_phase2: {
          shaderFile: "sample.wgsl",
          entryPoint: "find_topk_phase2",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        softmax_and_sample: {
          shaderFile: "sample.wgsl",
          entryPoint: "softmax_and_sample",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        single_pass: {
          shaderFile: "sample.wgsl",
          entryPoint: "sample_single_pass",
          workgroupSize: [256, 1, 1],
          requires: []
        }
      },
      bf16_to_f32: {
        default: {
          shaderFile: "bf16_to_f32.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        }
      },
      bf16_to_f16: {
        default: {
          shaderFile: "bf16_to_f16.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
        }
      }
    };
  }
});

// src/gpu/weight-buffer.ts
function isWeightBuffer(value) {
  return typeof value === "object" && value !== null && "buffer" in value && "dtype" in value && "layout" in value && "shape" in value;
}
function getBuffer(weight) {
  return isWeightBuffer(weight) ? weight.buffer : weight;
}
function getLayout(weight) {
  return isWeightBuffer(weight) ? weight.layout : null;
}
function getWeightDtype(weight) {
  return isWeightBuffer(weight) ? weight.dtype : null;
}
var init_weight_buffer = __esm({
  "src/gpu/weight-buffer.ts"() {
  }
});

// src/gpu/buffer-pool.ts
var buffer_pool_exports = {};
__export(buffer_pool_exports, {
  BufferPool: () => BufferPool,
  BufferUsage: () => BufferUsage,
  acquireBuffer: () => acquireBuffer,
  createBufferPool: () => createBufferPool,
  createStagingBuffer: () => createStagingBuffer,
  createUniformBuffer: () => createUniformBuffer,
  createUploadBuffer: () => createUploadBuffer,
  destroyBufferPool: () => destroyBufferPool,
  getBufferPool: () => getBufferPool,
  readBuffer: () => readBuffer,
  releaseBuffer: () => releaseBuffer,
  uploadData: () => uploadData,
  withBuffer: () => withBuffer
});
function alignTo(size, alignment) {
  return Math.ceil(size / alignment) * alignment;
}
function getSizeBucket(size, maxAllowedSize = Infinity, bucketConfig = getRuntimeConfig().bufferPool.bucket) {
  const minBucket = bucketConfig.minBucketSizeBytes;
  if (size <= minBucket)
    return minBucket;
  const largeThreshold = bucketConfig.largeBufferThresholdBytes;
  if (size >= largeThreshold) {
    const largeStep = bucketConfig.largeBufferStepBytes;
    const bucket2 = Math.ceil(size / largeStep) * largeStep;
    if (bucket2 > maxAllowedSize) {
      return alignTo(size, minBucket);
    }
    return bucket2;
  }
  const bits = 32 - Math.clz32(size - 1);
  const bucket = Math.pow(2, bits);
  if (bucket > maxAllowedSize) {
    return alignTo(size, minBucket);
  }
  return bucket;
}
function getBufferPool() {
  if (!globalPool) {
    globalPool = new BufferPool();
  }
  return globalPool;
}
function createBufferPool(debugMode, schemaConfig) {
  return new BufferPool(debugMode, schemaConfig);
}
function destroyBufferPool() {
  if (globalPool) {
    globalPool.destroy();
    globalPool = null;
  }
}
async function withBuffer(size, usage, fn) {
  const pool = getBufferPool();
  const buffer = pool.acquire(size, usage);
  try {
    return await fn(buffer);
  } finally {
    pool.release(buffer);
  }
}
var BufferUsage, BufferPool, globalPool, createStagingBuffer, createUploadBuffer, createUniformBuffer, acquireBuffer, releaseBuffer, uploadData, readBuffer;
var init_buffer_pool = __esm({
  "src/gpu/buffer-pool.ts"() {
    init_device();
    init_perf_guards();
    init_debug();
    init_runtime();
    BufferUsage = {
      STORAGE: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      STORAGE_READ: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      UNIFORM: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      STAGING_READ: GPUMapMode.READ | GPUBufferUsage.COPY_DST,
      STAGING_WRITE: GPUMapMode.WRITE | GPUBufferUsage.COPY_SRC
    };
    BufferPool = class {
      // Pools organized by usage and size bucket
      // Map<usage, Map<sizeBucket, GPUBuffer[]>>
      pools;
      // Active buffers (currently in use)
      activeBuffers;
      // Buffer metadata for leak detection (debug mode)
      bufferMetadata;
      // Deferred destruction queue (buffers destroyed after GPU work completes)
      pendingDestruction;
      destructionScheduled;
      // Statistics
      stats;
      // Configuration
      config;
      // Schema-based configuration
      schemaConfig;
      // Debug mode flag
      debugMode;
      constructor(debugMode = false, schemaConfig) {
        this.pools = /* @__PURE__ */ new Map();
        this.activeBuffers = /* @__PURE__ */ new Set();
        this.bufferMetadata = /* @__PURE__ */ new Map();
        this.debugMode = debugMode;
        this.schemaConfig = schemaConfig ?? getRuntimeConfig().bufferPool;
        this.pendingDestruction = /* @__PURE__ */ new Set();
        this.destructionScheduled = false;
        this.stats = {
          allocations: 0,
          reuses: 0,
          totalBytesAllocated: 0,
          peakBytesAllocated: 0,
          currentBytesAllocated: 0
        };
        this.config = {
          maxPoolSizePerBucket: this.schemaConfig.limits.maxBuffersPerBucket,
          maxTotalPooledBuffers: this.schemaConfig.limits.maxTotalPooledBuffers,
          enablePooling: true,
          alignmentBytes: this.schemaConfig.alignment.alignmentBytes
        };
      }
      /**
       * Get or create a buffer of the specified size
       */
      acquire(size, usage = BufferUsage.STORAGE, label = "pooled_buffer") {
        const device2 = getDevice();
        if (!device2) {
          throw new Error("Device not initialized");
        }
        const limits = getDeviceLimits();
        const maxSize = limits?.maxBufferSize || Infinity;
        const maxStorageSize = limits?.maxStorageBufferBindingSize || Infinity;
        const isStorageBuffer = (usage & GPUBufferUsage.STORAGE) !== 0;
        const alignedSize = alignTo(size, this.config.alignmentBytes);
        const maxAllowedBucket = isStorageBuffer ? Math.min(maxSize, maxStorageSize) : maxSize;
        const bucket = getSizeBucket(alignedSize, maxAllowedBucket, this.schemaConfig.bucket);
        if (bucket > maxSize) {
          throw new Error(
            `Buffer size ${bucket} exceeds device maxBufferSize (${maxSize}). Requested: ${size} bytes, bucketed to: ${bucket} bytes.`
          );
        }
        if (isStorageBuffer && bucket > maxStorageSize) {
          throw new Error(
            `Storage buffer size ${bucket} exceeds device maxStorageBufferBindingSize (${maxStorageSize}). Consider splitting into smaller buffers or using a different strategy.`
          );
        }
        if (this.config.enablePooling) {
          const pooled = this._getFromPool(bucket, usage);
          if (pooled) {
            this.activeBuffers.add(pooled);
            this.stats.reuses++;
            if (this.debugMode) {
              this._trackBuffer(pooled, bucket, usage, label);
            }
            return pooled;
          }
        }
        const buffer = device2.createBuffer({
          label: `${label}_${bucket}`,
          size: bucket,
          usage
        });
        this.activeBuffers.add(buffer);
        this.stats.allocations++;
        this.stats.totalBytesAllocated += bucket;
        this.stats.currentBytesAllocated += bucket;
        this.stats.peakBytesAllocated = Math.max(
          this.stats.peakBytesAllocated,
          this.stats.currentBytesAllocated
        );
        trackAllocation(bucket, label);
        if (this.debugMode) {
          this._trackBuffer(buffer, bucket, usage, label);
        }
        return buffer;
      }
      /**
       * Release a buffer back to the pool
       */
      release(buffer) {
        if (!this.activeBuffers.has(buffer)) {
          log.warn("BufferPool", "Releasing buffer not tracked as active");
          return;
        }
        this.activeBuffers.delete(buffer);
        if (this.debugMode) {
          this.bufferMetadata.delete(buffer);
        }
        if (!this.config.enablePooling) {
          this.deferDestroy(buffer);
          this.stats.currentBytesAllocated -= buffer.size;
          return;
        }
        const bucket = buffer.size;
        const usage = buffer.usage;
        if (!this.pools.has(usage)) {
          this.pools.set(usage, /* @__PURE__ */ new Map());
        }
        const usagePool = this.pools.get(usage);
        if (!usagePool.has(bucket)) {
          usagePool.set(bucket, []);
        }
        const bucketPool = usagePool.get(bucket);
        if (bucketPool.length < this.config.maxPoolSizePerBucket && this._getTotalPooledCount() < this.config.maxTotalPooledBuffers) {
          bucketPool.push(buffer);
        } else {
          this.deferDestroy(buffer);
          this.stats.currentBytesAllocated -= buffer.size;
        }
      }
      /**
       * Defer buffer destruction until all submitted GPU work completes.
       * This avoids destroying buffers still referenced by in-flight command buffers.
       */
      deferDestroy(buffer) {
        this.pendingDestruction.add(buffer);
        if (this.destructionScheduled) {
          return;
        }
        const device2 = getDevice();
        if (!device2) {
          for (const pending of this.pendingDestruction) {
            pending.destroy();
          }
          this.pendingDestruction.clear();
          this.destructionScheduled = false;
          return;
        }
        this.destructionScheduled = true;
        device2.queue.onSubmittedWorkDone().then(() => {
          for (const pending of this.pendingDestruction) {
            pending.destroy();
          }
          this.pendingDestruction.clear();
          this.destructionScheduled = false;
        }).catch((err) => {
          log.warn("BufferPool", `Deferred destruction failed: ${err.message}`);
          this.pendingDestruction.clear();
          this.destructionScheduled = false;
        });
      }
      /**
       * Get a buffer from the pool if available
       */
      _getFromPool(bucket, usage) {
        const usagePool = this.pools.get(usage);
        if (!usagePool)
          return null;
        const bucketPool = usagePool.get(bucket);
        if (!bucketPool || bucketPool.length === 0)
          return null;
        return bucketPool.pop();
      }
      /**
       * Get total count of pooled buffers
       */
      _getTotalPooledCount() {
        let count = 0;
        for (const usagePool of this.pools.values()) {
          for (const bucketPool of usagePool.values()) {
            count += bucketPool.length;
          }
        }
        return count;
      }
      /**
       * Track buffer metadata for leak detection (debug mode)
       */
      _trackBuffer(buffer, size, usage, label) {
        const metadata = {
          size,
          usage,
          label,
          acquiredAt: Date.now()
        };
        if (Error.captureStackTrace) {
          const obj = {};
          Error.captureStackTrace(obj);
          metadata.stackTrace = obj.stack;
        }
        this.bufferMetadata.set(buffer, metadata);
      }
      /**
       * Detect leaked buffers (debug mode)
       */
      detectLeaks(thresholdMs = 6e4) {
        if (!this.debugMode) {
          log.warn("BufferPool", "Leak detection requires debug mode");
          return [];
        }
        const now = Date.now();
        const leaks = [];
        for (const [buffer, metadata] of this.bufferMetadata.entries()) {
          if (this.activeBuffers.has(buffer)) {
            const age = now - metadata.acquiredAt;
            if (age > thresholdMs) {
              leaks.push(metadata);
            }
          }
        }
        return leaks;
      }
      /**
       * Create a staging buffer for CPU readback
       */
      createStagingBuffer(size) {
        return this.acquire(size, BufferUsage.STAGING_READ, "staging_read");
      }
      /**
       * Create a staging buffer for CPU upload
       */
      createUploadBuffer(size) {
        return this.acquire(size, BufferUsage.STAGING_WRITE, "staging_write");
      }
      /**
       * Create a uniform buffer
       */
      createUniformBuffer(size) {
        const alignedSize = alignTo(size, 256);
        return this.acquire(alignedSize, BufferUsage.UNIFORM, "uniform");
      }
      /**
       * Upload data to GPU buffer
       */
      uploadData(buffer, data, offset = 0) {
        const device2 = getDevice();
        if (!device2) {
          throw new Error("Device not initialized");
        }
        device2.queue.writeBuffer(buffer, offset, data);
      }
      /**
       * Read data from GPU buffer
       * NOTE: GPU readbacks are expensive (0.5-2ms overhead per call). Use sparingly.
       */
      async readBuffer(buffer, size = buffer.size) {
        if (!allowReadback("BufferPool.readBuffer")) {
          return new ArrayBuffer(0);
        }
        const device2 = getDevice();
        if (!device2) {
          throw new Error("Device not initialized");
        }
        const staging = this.createStagingBuffer(size);
        const encoder = device2.createCommandEncoder({ label: "readback_encoder" });
        encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
        device2.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = staging.getMappedRange(0, size).slice(0);
        staging.unmap();
        this.release(staging);
        return data;
      }
      /**
       * Clear all pooled buffers
       */
      clearPool() {
        for (const usagePool of this.pools.values()) {
          for (const bucketPool of usagePool.values()) {
            for (const buffer of bucketPool) {
              buffer.destroy();
              this.stats.currentBytesAllocated -= buffer.size;
            }
            bucketPool.length = 0;
          }
        }
        this.pools.clear();
        for (const buffer of this.pendingDestruction) {
          buffer.destroy();
        }
        this.pendingDestruction.clear();
        this.destructionScheduled = false;
      }
      /**
       * Destroy all buffers (active and pooled)
       */
      destroy() {
        for (const buffer of this.activeBuffers) {
          buffer.destroy();
        }
        this.activeBuffers.clear();
        this.bufferMetadata.clear();
        this.clearPool();
        this.stats.currentBytesAllocated = 0;
      }
      /**
       * Get pool statistics
       */
      getStats() {
        return {
          ...this.stats,
          activeBuffers: this.activeBuffers.size,
          pooledBuffers: this._getTotalPooledCount(),
          hitRate: this.stats.allocations > 0 ? (this.stats.reuses / (this.stats.allocations + this.stats.reuses) * 100).toFixed(1) + "%" : "0%"
        };
      }
      /**
       * Configure pool settings
       */
      configure(config2) {
        Object.assign(this.config, config2);
      }
    };
    globalPool = null;
    createStagingBuffer = (size) => getBufferPool().createStagingBuffer(size);
    createUploadBuffer = (size) => getBufferPool().createUploadBuffer(size);
    createUniformBuffer = (size) => getBufferPool().createUniformBuffer(size);
    acquireBuffer = (size, usage, label) => getBufferPool().acquire(size, usage, label);
    releaseBuffer = (buffer) => getBufferPool().release(buffer);
    uploadData = (buffer, data, offset) => getBufferPool().uploadData(buffer, data, offset);
    readBuffer = (buffer, size) => getBufferPool().readBuffer(buffer, size);
  }
});

// src/gpu/kernels/dispatch.ts
function dispatch(device2, pipeline, bindGroup, workgroups, label = "compute") {
  const encoder = device2.createCommandEncoder({ label: `${label}_encoder` });
  const pass = encoder.beginComputePass({ label: `${label}_pass` });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  if (typeof workgroups === "number") {
    pass.dispatchWorkgroups(workgroups);
  } else {
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
  }
  pass.end();
  device2.queue.submit([encoder.finish()]);
}
function recordDispatch(recorder, pipeline, bindGroup, workgroups, label = "compute") {
  const pass = recorder.beginComputePass(label);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  if (typeof workgroups === "number") {
    pass.dispatchWorkgroups(workgroups);
  } else {
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
  }
  pass.end();
}
function dispatchIndirect(device2, pipeline, bindGroup, indirectBuffer, indirectOffset = 0, label = "compute") {
  const encoder = device2.createCommandEncoder({ label: `${label}_encoder` });
  const pass = encoder.beginComputePass({ label: `${label}_pass` });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset);
  pass.end();
  device2.queue.submit([encoder.finish()]);
}
function recordDispatchIndirect(recorder, pipeline, bindGroup, indirectBuffer, indirectOffset = 0, label = "compute") {
  const pass = recorder.beginComputePass(label);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset);
  pass.end();
}
var init_dispatch = __esm({
  "src/gpu/kernels/dispatch.ts"() {
  }
});

// src/gpu/kernels/constants.ts
var WORKGROUP_SIZES, VEC4_ELEMENTS_PER_WG, GPU_LIMITS, MEMORY_THRESHOLDS, DIMENSION_LIMITS, TILE_SIZES, QUANTIZATION, ALIGNMENT, PERFORMANCE;
var init_constants = __esm({
  "src/gpu/kernels/constants.ts"() {
    WORKGROUP_SIZES = {
      /** Default workgroup size for most kernels */
      DEFAULT: 256,
      /** Vec4 workgroup thread count (64 threads × 4 elements = 256 elements) */
      VEC4_THREADS: 64,
      /** Attention kernels (large blocks) */
      ATTENTION_LARGE_BLOCK: 64,
      /** Attention kernels (small blocks) */
      ATTENTION_SMALL_BLOCK: 32,
      /** Subgroup size (typical for modern GPUs) */
      SUBGROUP: 32,
      /** RMSNorm workgroup size */
      RMSNORM: 256,
      /** Softmax workgroup size */
      SOFTMAX: 256,
      /** Matmul tile sizes */
      MATMUL_TILE_M: 16,
      MATMUL_TILE_N: 16,
      MATMUL_TILE_K: 16,
      /** MoE workgroup size */
      MOE: 256
    };
    VEC4_ELEMENTS_PER_WG = WORKGROUP_SIZES.VEC4_THREADS * 4;
    GPU_LIMITS = {
      /** Max workgroups per dimension (WebGPU minimum) */
      MAX_WORKGROUPS: 65535
    };
    MEMORY_THRESHOLDS = {
      /** Large attention tier shared memory requirement */
      ATTENTION_LARGE_SHARED: 49152,
      // 48KB
      /** Small attention tier shared memory requirement (F32) */
      ATTENTION_SMALL_SHARED_F32: 8192,
      // 8KB (2 * 32 * 32 * 4)
      /** Small attention tier shared memory requirement (F16) */
      ATTENTION_SMALL_SHARED_F16: 4096,
      // 4KB (2 * 32 * 32 * 2)
      /** Subgroup attention tier shared memory requirement */
      ATTENTION_SUBGROUP_SHARED: 8192,
      // 2048 * 4 bytes for scores array
      /** Minimum shared memory for any GPU */
      MIN_SHARED_MEMORY: 16384
      // 16KB (WebGPU minimum spec)
    };
    DIMENSION_LIMITS = {
      /** Maximum head dimension for large attention tier */
      ATTENTION_LARGE_MAX_HEAD_DIM: 64,
      /** Maximum head dimension for small attention tier */
      ATTENTION_SMALL_MAX_HEAD_DIM: 256,
      /** Maximum head dimension for subgroup attention tier */
      ATTENTION_SUBGROUP_MAX_HEAD_DIM: 256,
      /** Maximum sequence length for practical inference */
      MAX_SEQ_LEN: 32768,
      /** Maximum vocab size for typical models */
      MAX_VOCAB_SIZE: 262144,
      // Gemma 3
      /** Maximum batch size for prefill */
      MAX_BATCH_SIZE: 128
    };
    TILE_SIZES = {
      /** Attention tile sizes (large) */
      ATTENTION_LARGE_BLOCK_SIZE: 64,
      ATTENTION_LARGE_HEAD_TILE: 64,
      /** Attention tile sizes (small) */
      ATTENTION_SMALL_BLOCK_SIZE: 32,
      ATTENTION_SMALL_HEAD_TILE: 32,
      /** Matmul tile sizes */
      MATMUL_M: 16,
      MATMUL_N: 16,
      MATMUL_K: 16,
      /** Q4K dequant tile sizes */
      Q4K_BLOCK_SIZE: 32,
      Q4K_SUPER_BLOCK_SIZE: 256
    };
    QUANTIZATION = {
      /** Q4K_M bits per weight */
      Q4K_BITS: 4.5,
      /** Q4K block bytes per 256-element super-block */
      Q4K_BLOCK_BYTES: 144,
      /** Q8_0 bits per weight */
      Q8_BITS: 8.5,
      /** F16 bits per weight */
      F16_BITS: 16,
      /** BF16 bits per weight */
      BF16_BITS: 16,
      /** F32 bits per weight */
      F32_BITS: 32,
      /** MXFP4 bits per weight (including shared exponent) */
      MXFP4_BITS: 4
    };
    ALIGNMENT = {
      /** WebGPU buffer alignment */
      BUFFER: 256,
      /** Uniform buffer alignment */
      UNIFORM: 256,
      /** Storage buffer alignment */
      STORAGE: 256,
      /** Vertex buffer alignment */
      VERTEX: 4
    };
    PERFORMANCE = {
      /** Number of warmup runs for benchmarks */
      WARMUP_RUNS: 5,
      /** Number of timed runs for benchmarks */
      TIMED_RUNS: 20,
      /** Default timeout for operations (ms) */
      DEFAULT_TIMEOUT: 12e4,
      /** Max buffer pool size per bucket */
      MAX_POOL_SIZE_PER_BUCKET: 8,
      /** Max total pooled buffers */
      MAX_TOTAL_POOLED_BUFFERS: 64
    };
  }
});

// src/gpu/kernels/fused_matmul_rmsnorm.ts
var fused_matmul_rmsnorm_exports = {};
__export(fused_matmul_rmsnorm_exports, {
  recordMatmulRMSNormFused: () => recordMatmulRMSNormFused,
  runMatmulRMSNormFused: () => runMatmulRMSNormFused,
  selectMatmulRMSNormFusedVariant: () => selectMatmulRMSNormFusedVariant,
  shouldUseFusedMatmulRMSNorm: () => shouldUseFusedMatmulRMSNorm
});
function selectMatmulRMSNormFusedVariant(N) {
  if (N <= WORKGROUP_SIZES.DEFAULT) {
    return "small";
  }
  return "medium";
}
async function runMatmulRMSNormFused(input, weight, normWeight, options) {
  const device2 = getDevice();
  const {
    N,
    K,
    eps = 1e-5,
    residual = null,
    outputBuffer = null,
    transposeB = true
    // Default: GGUF row-major weights
  } = options;
  const { colsPerWg } = getKernelThresholds().fusedMatmul;
  if (N > colsPerWg) {
    throw new Error(`[MatmulRMSNormFused] N=${N} exceeds colsPerWg=${colsPerWg}; kernel only supports single-workgroup RMSNorm.`);
  }
  const weightBuffer = getBuffer(weight);
  const variant = selectMatmulRMSNormFusedVariant(N);
  trace.kernels(`MatmulRMSNormFused: N=${N}, K=${K}, variant=${variant}, hasResidual=${!!residual}, transposeB=${transposeB}`);
  const pipeline = await getPipelineFast("fused_matmul_rmsnorm", variant);
  const outputSize = N * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "matmul_rmsnorm_fused_output");
  const uniformBuffer = createUniformBufferWithView(
    "matmul_rmsnorm_fused_uniforms",
    32,
    (view) => {
      view.setUint32(0, N, true);
      view.setUint32(4, K, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, residual ? 1 : 0, true);
      view.setUint32(16, transposeB ? 1 : 0, true);
    },
    null,
    device2
  );
  const residualBuffer = residual || device2.createBuffer({
    label: "matmul_rmsnorm_residual_placeholder",
    size: 4,
    usage: GPUBufferUsage.STORAGE
  });
  const bindGroup = device2.createBindGroup({
    label: "matmul_rmsnorm_fused_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weightBuffer } },
      { binding: 3, resource: { buffer: normWeight } },
      { binding: 4, resource: { buffer: output } },
      { binding: 5, resource: { buffer: residualBuffer } }
    ]
  });
  let workgroups;
  if (variant === "small" || variant === "medium") {
    workgroups = 1;
  } else {
    workgroups = Math.ceil(N / getKernelThresholds().fusedMatmul.colsPerWg);
  }
  dispatch(device2, pipeline, bindGroup, workgroups, "matmul_rmsnorm_fused");
  uniformBuffer.destroy();
  if (!residual)
    residualBuffer.destroy();
  return createTensor(output, input.dtype, [1, N], "matmul_rmsnorm_fused_output");
}
async function recordMatmulRMSNormFused(recorder, input, weight, normWeight, options) {
  const device2 = recorder.device;
  const {
    N,
    K,
    eps = 1e-5,
    residual = null,
    outputBuffer = null,
    transposeB = true
    // Default: GGUF row-major weights
  } = options;
  const { colsPerWg } = getKernelThresholds().fusedMatmul;
  if (N > colsPerWg) {
    throw new Error(`[MatmulRMSNormFused] N=${N} exceeds colsPerWg=${colsPerWg}; kernel only supports single-workgroup RMSNorm.`);
  }
  const weightBuffer = getBuffer(weight);
  const variant = selectMatmulRMSNormFusedVariant(N);
  trace.kernels(`recordMatmulRMSNormFused: N=${N}, K=${K}, variant=${variant}, hasResidual=${!!residual}, transposeB=${transposeB}`);
  const pipeline = await getPipelineFast("fused_matmul_rmsnorm", variant);
  const outputSize = N * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "matmul_rmsnorm_fused_output");
  const uniformBuffer = createUniformBufferWithView(
    "matmul_rmsnorm_fused_uniforms",
    32,
    (view) => {
      view.setUint32(0, N, true);
      view.setUint32(4, K, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, residual ? 1 : 0, true);
      view.setUint32(16, transposeB ? 1 : 0, true);
    },
    recorder
  );
  const residualBuffer = residual || device2.createBuffer({
    label: "matmul_rmsnorm_residual_placeholder",
    size: 4,
    usage: GPUBufferUsage.STORAGE
  });
  const bindGroup = device2.createBindGroup({
    label: "matmul_rmsnorm_fused_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weightBuffer } },
      { binding: 3, resource: { buffer: normWeight } },
      { binding: 4, resource: { buffer: output } },
      { binding: 5, resource: { buffer: residualBuffer } }
    ]
  });
  let workgroups;
  if (variant === "small" || variant === "medium") {
    workgroups = 1;
  } else {
    workgroups = Math.ceil(N / getKernelThresholds().fusedMatmul.colsPerWg);
  }
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "matmul_rmsnorm_fused");
  if (!residual) {
    recorder.trackTemporaryBuffer(residualBuffer);
  }
  return createTensor(output, input.dtype, [1, N], "matmul_rmsnorm_fused_output");
}
function shouldUseFusedMatmulRMSNorm(M, N) {
  if (M !== 1) {
    return false;
  }
  const { colsPerWg } = getKernelThresholds().fusedMatmul;
  if (N > colsPerWg) {
    return false;
  }
  return true;
}
var init_fused_matmul_rmsnorm = __esm({
  "src/gpu/kernels/fused_matmul_rmsnorm.ts"() {
    init_device();
    init_buffer_pool();
    init_tensor();
    init_weight_buffer();
    init_dispatch();
    init_utils();
    init_constants();
    init_kernel_thresholds_schema();
    init_debug();
  }
});

// kernel-tests/browser/test-page.ts
init_device();
init_tensor();
init_loader();
init_registry();

// src/config/kernel-path-loader.ts
init_kernel_path_schema();
init_utils();

// src/config/presets/kernel-paths/gemma2-q4k-fused.json
var gemma2_q4k_fused_default = {
  id: "gemma2-q4k-fused",
  name: "Gemma 2 Q4K Fused",
  description: "Q4K weights with fused dequant+matmul and fused FFN. Best throughput for Q4K.",
  decode: {
    steps: [
      { op: "input_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "q_proj", kernel: "fused_matmul_q4.wgsl", entry: "main_multicol", weights: "layer.{L}.self_attn.q_proj" },
      { op: "k_proj", kernel: "fused_matmul_q4.wgsl", entry: "main_multicol", weights: "layer.{L}.self_attn.k_proj" },
      { op: "v_proj", kernel: "fused_matmul_q4.wgsl", entry: "main_multicol", weights: "layer.{L}.self_attn.v_proj" },
      { op: "rope_q", kernel: "rope.wgsl", entry: "main" },
      { op: "rope_k", kernel: "rope.wgsl", entry: "main" },
      { op: "attention", kernel: "attention_decode.wgsl", entry: "attention_decode", constants: { SOFTCAP: 50 } },
      { op: "o_proj", kernel: "fused_matmul_q4.wgsl", entry: "main_multicol", weights: "layer.{L}.self_attn.o_proj" },
      { op: "attn_residual", kernel: "residual.wgsl", entry: "main" },
      { op: "post_attn_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "ffn_gate_up", kernel: "fused_ffn_q4k.wgsl", entry: "main", weights: "layer.{L}.mlp", constants: { ACTIVATION: 1 } },
      { op: "down_proj", kernel: "fused_matmul_q4.wgsl", entry: "main_multicol", weights: "layer.{L}.mlp.down_proj" },
      { op: "ffn_residual", kernel: "residual.wgsl", entry: "main" }
    ]
  },
  prefill: {
    steps: [
      { op: "input_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "q_proj", kernel: "fused_matmul_q4_batched.wgsl", entry: "main_batched", weights: "layer.{L}.self_attn.q_proj" },
      { op: "k_proj", kernel: "fused_matmul_q4_batched.wgsl", entry: "main_batched", weights: "layer.{L}.self_attn.k_proj" },
      { op: "v_proj", kernel: "fused_matmul_q4_batched.wgsl", entry: "main_batched", weights: "layer.{L}.self_attn.v_proj" },
      { op: "rope_q", kernel: "rope.wgsl", entry: "main" },
      { op: "rope_k", kernel: "rope.wgsl", entry: "main" },
      { op: "attention", kernel: "attention.wgsl", entry: "main", constants: { SOFTCAP: 50 } },
      { op: "o_proj", kernel: "fused_matmul_q4_batched.wgsl", entry: "main_batched", weights: "layer.{L}.self_attn.o_proj" },
      { op: "attn_residual", kernel: "residual.wgsl", entry: "main" },
      { op: "post_attn_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "ffn_gate_up", kernel: "fused_ffn_q4k.wgsl", entry: "main_batched", weights: "layer.{L}.mlp", constants: { ACTIVATION: 1 } },
      { op: "down_proj", kernel: "fused_matmul_q4_batched.wgsl", entry: "main_batched", weights: "layer.{L}.mlp.down_proj" },
      { op: "ffn_residual", kernel: "residual.wgsl", entry: "main" }
    ]
  },
  preLayer: [
    { op: "embed", kernel: "gather.wgsl", entry: "main", weights: "embed_tokens" }
  ],
  postLayer: [
    { op: "final_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
    { op: "lm_head", kernel: "fused_matmul_q4.wgsl", entry: "main_multicol", weights: "lm_head" }
  ],
  sampling: [
    { op: "softcap", kernel: "sample.wgsl", entry: "apply_softcap", constants: { SOFTCAP: 30 } },
    { op: "sample", kernel: "sample.wgsl", entry: "sample_single_pass" }
  ]
};

// src/config/presets/kernel-paths/gemma2-q4k-dequant-f32.json
var gemma2_q4k_dequant_f32_default = {
  id: "gemma2-q4k-dequant-f32",
  name: "Gemma 2 Q4K Dequant F32",
  description: "Q4K weights dequantized to F32 at load time. Best accuracy, slower throughput.",
  decode: {
    steps: [
      { op: "input_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "q_proj", kernel: "matmul_f32.wgsl", entry: "main", weights: "layer.{L}.self_attn.q_proj" },
      { op: "k_proj", kernel: "matmul_f32.wgsl", entry: "main", weights: "layer.{L}.self_attn.k_proj" },
      { op: "v_proj", kernel: "matmul_f32.wgsl", entry: "main", weights: "layer.{L}.self_attn.v_proj" },
      { op: "rope_q", kernel: "rope.wgsl", entry: "main" },
      { op: "rope_k", kernel: "rope.wgsl", entry: "main" },
      { op: "attention", kernel: "attention_decode.wgsl", entry: "attention_decode", constants: { SOFTCAP: 50 } },
      { op: "o_proj", kernel: "matmul_f32.wgsl", entry: "main", weights: "layer.{L}.self_attn.o_proj" },
      { op: "attn_residual", kernel: "residual.wgsl", entry: "main" },
      { op: "post_attn_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "gate_proj", kernel: "matmul_f32.wgsl", entry: "main", weights: "layer.{L}.mlp.gate_proj" },
      { op: "up_proj", kernel: "matmul_f32.wgsl", entry: "main", weights: "layer.{L}.mlp.up_proj" },
      { op: "activation", kernel: "silu.wgsl", entry: "geglu" },
      { op: "down_proj", kernel: "matmul_f32.wgsl", entry: "main", weights: "layer.{L}.mlp.down_proj" },
      { op: "ffn_residual", kernel: "residual.wgsl", entry: "main" }
    ]
  },
  prefill: {
    steps: [
      { op: "input_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "q_proj", kernel: "matmul_f32.wgsl", entry: "main", weights: "layer.{L}.self_attn.q_proj" },
      { op: "k_proj", kernel: "matmul_f32.wgsl", entry: "main", weights: "layer.{L}.self_attn.k_proj" },
      { op: "v_proj", kernel: "matmul_f32.wgsl", entry: "main", weights: "layer.{L}.self_attn.v_proj" },
      { op: "rope_q", kernel: "rope.wgsl", entry: "main" },
      { op: "rope_k", kernel: "rope.wgsl", entry: "main" },
      { op: "attention", kernel: "attention.wgsl", entry: "main", constants: { SOFTCAP: 50 } },
      { op: "o_proj", kernel: "matmul_f32.wgsl", entry: "main", weights: "layer.{L}.self_attn.o_proj" },
      { op: "attn_residual", kernel: "residual.wgsl", entry: "main" },
      { op: "post_attn_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "gate_proj", kernel: "matmul_f32.wgsl", entry: "main", weights: "layer.{L}.mlp.gate_proj" },
      { op: "up_proj", kernel: "matmul_f32.wgsl", entry: "main", weights: "layer.{L}.mlp.up_proj" },
      { op: "activation", kernel: "silu.wgsl", entry: "geglu" },
      { op: "down_proj", kernel: "matmul_f32.wgsl", entry: "main", weights: "layer.{L}.mlp.down_proj" },
      { op: "ffn_residual", kernel: "residual.wgsl", entry: "main" }
    ]
  },
  preLayer: [
    { op: "embed", kernel: "gather.wgsl", entry: "main", weights: "embed_tokens" }
  ],
  postLayer: [
    { op: "final_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
    { op: "lm_head", kernel: "matmul_f32.wgsl", entry: "main", weights: "lm_head" }
  ],
  sampling: [
    { op: "softcap", kernel: "sample.wgsl", entry: "apply_softcap", constants: { SOFTCAP: 30 } },
    { op: "sample", kernel: "sample.wgsl", entry: "sample_single_pass" }
  ]
};

// src/config/presets/kernel-paths/gemma2-q4k-dequant-f16.json
var gemma2_q4k_dequant_f16_default = {
  id: "gemma2-q4k-dequant-f16",
  name: "Gemma 2 Q4K Dequant F16",
  description: "Q4K weights dequantized to F16. Balanced speed/accuracy, lower VRAM than F32.",
  decode: {
    steps: [
      { op: "input_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "q_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.self_attn.q_proj" },
      { op: "k_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.self_attn.k_proj" },
      { op: "v_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.self_attn.v_proj" },
      { op: "rope_q", kernel: "rope.wgsl", entry: "main" },
      { op: "rope_k", kernel: "rope.wgsl", entry: "main" },
      { op: "attention", kernel: "attention_decode.wgsl", entry: "attention_decode", constants: { SOFTCAP: 50 } },
      { op: "o_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.self_attn.o_proj" },
      { op: "attn_residual", kernel: "residual.wgsl", entry: "main" },
      { op: "post_attn_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "gate_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.mlp.gate_proj" },
      { op: "up_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.mlp.up_proj" },
      { op: "activation", kernel: "silu.wgsl", entry: "geglu" },
      { op: "down_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.mlp.down_proj" },
      { op: "ffn_residual", kernel: "residual.wgsl", entry: "main" }
    ]
  },
  prefill: {
    steps: [
      { op: "input_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "q_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.self_attn.q_proj" },
      { op: "k_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.self_attn.k_proj" },
      { op: "v_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.self_attn.v_proj" },
      { op: "rope_q", kernel: "rope.wgsl", entry: "main" },
      { op: "rope_k", kernel: "rope.wgsl", entry: "main" },
      { op: "attention", kernel: "attention.wgsl", entry: "main", constants: { SOFTCAP: 50 } },
      { op: "o_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.self_attn.o_proj" },
      { op: "attn_residual", kernel: "residual.wgsl", entry: "main" },
      { op: "post_attn_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "gate_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.mlp.gate_proj" },
      { op: "up_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.mlp.up_proj" },
      { op: "activation", kernel: "silu.wgsl", entry: "geglu" },
      { op: "down_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.mlp.down_proj" },
      { op: "ffn_residual", kernel: "residual.wgsl", entry: "main" }
    ]
  },
  preLayer: [
    { op: "embed", kernel: "gather.wgsl", entry: "main", weights: "embed_tokens" }
  ],
  postLayer: [
    { op: "final_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
    { op: "lm_head", kernel: "matmul_f16.wgsl", entry: "main", weights: "lm_head" }
  ],
  sampling: [
    { op: "softcap", kernel: "sample.wgsl", entry: "apply_softcap", constants: { SOFTCAP: 30 } },
    { op: "sample", kernel: "sample.wgsl", entry: "sample_single_pass" }
  ]
};

// src/config/presets/kernel-paths/gemma2-f16-native.json
var gemma2_f16_native_default = {
  id: "gemma2-f16-native",
  name: "Gemma 2 F16 Native",
  description: "Native F16 weights, no dequantization. Baseline accuracy.",
  decode: {
    steps: [
      { op: "input_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "q_proj", kernel: "matmul_f16w_f32a.wgsl", entry: "main", weights: "layer.{L}.self_attn.q_proj" },
      { op: "k_proj", kernel: "matmul_f16w_f32a.wgsl", entry: "main", weights: "layer.{L}.self_attn.k_proj" },
      { op: "v_proj", kernel: "matmul_f16w_f32a.wgsl", entry: "main", weights: "layer.{L}.self_attn.v_proj" },
      { op: "rope_q", kernel: "rope.wgsl", entry: "main" },
      { op: "rope_k", kernel: "rope.wgsl", entry: "main" },
      { op: "attention", kernel: "attention_decode.wgsl", entry: "attention_decode", constants: { SOFTCAP: 50 } },
      { op: "o_proj", kernel: "matmul_f16w_f32a.wgsl", entry: "main", weights: "layer.{L}.self_attn.o_proj" },
      { op: "attn_residual", kernel: "residual.wgsl", entry: "main" },
      { op: "post_attn_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "gate_proj", kernel: "matmul_f16w_f32a.wgsl", entry: "main", weights: "layer.{L}.mlp.gate_proj" },
      { op: "up_proj", kernel: "matmul_f16w_f32a.wgsl", entry: "main", weights: "layer.{L}.mlp.up_proj" },
      { op: "activation", kernel: "silu.wgsl", entry: "geglu" },
      { op: "down_proj", kernel: "matmul_f16w_f32a.wgsl", entry: "main", weights: "layer.{L}.mlp.down_proj" },
      { op: "ffn_residual", kernel: "residual.wgsl", entry: "main" }
    ]
  },
  prefill: {
    steps: [
      { op: "input_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "q_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.self_attn.q_proj" },
      { op: "k_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.self_attn.k_proj" },
      { op: "v_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.self_attn.v_proj" },
      { op: "rope_q", kernel: "rope.wgsl", entry: "main" },
      { op: "rope_k", kernel: "rope.wgsl", entry: "main" },
      { op: "attention", kernel: "attention.wgsl", entry: "main", constants: { SOFTCAP: 50 } },
      { op: "o_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.self_attn.o_proj" },
      { op: "attn_residual", kernel: "residual.wgsl", entry: "main" },
      { op: "post_attn_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
      { op: "gate_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.mlp.gate_proj" },
      { op: "up_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.mlp.up_proj" },
      { op: "activation", kernel: "silu.wgsl", entry: "geglu" },
      { op: "down_proj", kernel: "matmul_f16.wgsl", entry: "main", weights: "layer.{L}.mlp.down_proj" },
      { op: "ffn_residual", kernel: "residual.wgsl", entry: "main" }
    ]
  },
  preLayer: [
    { op: "embed", kernel: "gather_f16.wgsl", entry: "main", weights: "embed_tokens" }
  ],
  postLayer: [
    { op: "final_norm", kernel: "rmsnorm.wgsl", entry: "main", constants: { RMS_NORM_OFFSET: true } },
    { op: "lm_head", kernel: "matmul_f16w_f32a.wgsl", entry: "main", weights: "lm_head" }
  ],
  sampling: [
    { op: "softcap", kernel: "sample.wgsl", entry: "apply_softcap", constants: { SOFTCAP: 30 } },
    { op: "sample", kernel: "sample.wgsl", entry: "sample_single_pass" }
  ]
};

// src/config/kernel-path-loader.ts
var KERNEL_PATH_REGISTRY = {
  // Gemma 2 Q4K variants
  "gemma2-q4k-fused": gemma2_q4k_fused_default,
  "gemma2-q4k-dequant-f32": gemma2_q4k_dequant_f32_default,
  "gemma2-q4k-dequant-f16": gemma2_q4k_dequant_f16_default,
  // Gemma 2 F16 native
  "gemma2-f16-native": gemma2_f16_native_default,
  // Aliases for generic access (model-agnostic)
  "q4k-fused": gemma2_q4k_fused_default,
  "q4k-dequant-f32": gemma2_q4k_dequant_f32_default,
  "q4k-dequant-f16": gemma2_q4k_dequant_f16_default,
  "f16-native": gemma2_f16_native_default,
  // Semantic aliases
  "q4k-safe": gemma2_q4k_dequant_f32_default,
  // Max compatibility, no fusion
  "q4k-fast": gemma2_q4k_fused_default,
  // Best throughput
  "q4k-balanced": gemma2_q4k_dequant_f16_default
  // Good speed/accuracy tradeoff
};
function getKernelPath(id) {
  return KERNEL_PATH_REGISTRY[id] ?? null;
}
function listKernelPaths() {
  return Object.keys(KERNEL_PATH_REGISTRY);
}
function resolveKernelPath(ref) {
  if (typeof ref === "string") {
    const path = getKernelPath(ref);
    if (!path) {
      throw new Error(`Unknown kernel path: ${ref}. Available: ${listKernelPaths().join(", ")}`);
    }
    return path;
  }
  return ref;
}
function getLayerSteps(path, layerIndex, phase) {
  if (path.layerOverrides) {
    for (const override of path.layerOverrides) {
      if (override.layers.includes(layerIndex)) {
        return override.steps;
      }
    }
  }
  const layerPath = phase === "prefill" && path.prefill ? path.prefill : path.decode;
  return layerPath.steps;
}
var MATMUL_ROLE_ALIASES = {
  q_proj: { section: "layer", ops: ["q_proj"] },
  k_proj: { section: "layer", ops: ["k_proj"] },
  v_proj: { section: "layer", ops: ["v_proj"] },
  qkv_proj: { section: "layer", ops: ["qkv_proj", "q_proj"] },
  o_proj: { section: "layer", ops: ["o_proj"] },
  ffn_gate: { section: "layer", ops: ["ffn_gate", "gate_proj"] },
  ffn_up: { section: "layer", ops: ["ffn_up", "up_proj"] },
  ffn_down: { section: "layer", ops: ["ffn_down", "down_proj"] },
  ffn_gate_up: { section: "layer", ops: ["ffn_gate_up"] },
  lm_head: { section: "postLayer", ops: ["lm_head"] }
};
function normalizeKernelFile(kernel) {
  const trimmed = kernel.trim();
  if (!trimmed)
    return trimmed;
  const parts = trimmed.split("/");
  return parts[parts.length - 1] ?? trimmed;
}
function getKernelPathStepsForSection(path, section, phase, layerIndex) {
  switch (section) {
    case "preLayer":
      return path.preLayer ?? [];
    case "postLayer":
      return path.postLayer ?? [];
    case "sampling":
      return path.sampling ?? [];
    case "layer":
    default:
      return getLayerSteps(path, layerIndex, phase);
  }
}
function findStepByOp(steps, op) {
  return steps.find((step) => step.op === op) ?? null;
}
function findKernelVariant(operation, kernel, entry) {
  const variants = KERNEL_CONFIGS[operation];
  if (!variants)
    return null;
  const normalizedKernel = normalizeKernelFile(kernel);
  const normalizedEntry = entry ?? DEFAULT_ENTRY;
  let fallbackVariant = null;
  let fallbackCount = 0;
  for (const [variant, config2] of Object.entries(variants)) {
    if (config2.shaderFile !== normalizedKernel)
      continue;
    fallbackVariant = variant;
    fallbackCount += 1;
    if (config2.entryPoint === normalizedEntry) {
      return variant;
    }
  }
  if (fallbackCount === 1) {
    return fallbackVariant;
  }
  return null;
}
function getKernelPathMatmulVariant(role, phase, layerIndex) {
  if (!activeKernelPath || !role)
    return null;
  const alias = MATMUL_ROLE_ALIASES[role] ?? { section: "layer", ops: [role] };
  const steps = getKernelPathStepsForSection(activeKernelPath, alias.section, phase, layerIndex ?? 0);
  for (const op of alias.ops) {
    const step = findStepByOp(steps, op);
    if (!step)
      continue;
    const variant = findKernelVariant("matmul", step.kernel, step.entry);
    if (variant) {
      return variant;
    }
  }
  return null;
}
function getKernelPathAttentionVariant(phase, layerIndex) {
  if (!activeKernelPath)
    return null;
  const steps = getKernelPathStepsForSection(activeKernelPath, "layer", phase, layerIndex ?? 0);
  const step = findStepByOp(steps, "attention");
  if (!step)
    return null;
  return findKernelVariant("attention", step.kernel, step.entry);
}
var activeKernelPath = null;
var activeKernelPathSource = "none";
function setActiveKernelPath(path, source = "none") {
  activeKernelPath = path;
  activeKernelPathSource = path ? source : "none";
}
function getKernelPathStrict() {
  return activeKernelPathSource !== "auto" && activeKernelPathSource !== "none";
}
function isActiveKernelPathFusedQ4K() {
  if (!activeKernelPath)
    return true;
  const kernelSteps = [
    ...activeKernelPath.decode?.steps ?? [],
    ...activeKernelPath.prefill?.steps ?? [],
    ...activeKernelPath.preLayer ?? [],
    ...activeKernelPath.postLayer ?? [],
    ...activeKernelPath.layerOverrides?.flatMap((override) => override.steps) ?? []
  ];
  return kernelSteps.some((step) => step.kernel.includes("fused_matmul_q4"));
}

// src/gpu/kernel-selector.ts
var kernel_selector_exports = {};
__export(kernel_selector_exports, {
  CommandRecorder: () => CommandRecorder,
  KERNEL_CONFIGS: () => KERNEL_CONFIGS,
  analyzeDecodePerformance: () => analyzeDecodePerformance,
  autoTuneKernels: () => autoTuneKernels,
  benchmarkAttentionDecode: () => benchmarkAttentionDecode,
  benchmarkDecodePass: () => benchmarkDecodePass,
  benchmarkMatmul: () => benchmarkMatmul,
  benchmarkMatmulRMSNormFused: () => benchmarkMatmulRMSNormFused,
  benchmarkRMSNorm: () => benchmarkRMSNorm,
  benchmarkSiLU: () => benchmarkSiLU,
  calculateFusedFFNSavings: () => calculateFusedFFNSavings,
  castF16ToF32: () => castF16ToF32,
  castF32ToF16: () => castF32ToF16,
  clearKernelCaches: () => clearKernelCaches,
  clearPipelineCache: () => clearPipelineCache,
  clearProfile: () => clearProfile,
  compareBenchmarks: () => compareBenchmarks,
  compileShader: () => compileShader,
  createCommandRecorder: () => createCommandRecorder,
  createDequantBindGroupLayout: () => createDequantBindGroupLayout,
  createMatmulBindGroupLayout: () => createMatmulBindGroupLayout,
  createPipeline: () => createPipeline,
  createProfilingRecorder: () => createProfilingRecorder,
  dequantize: () => dequantize,
  dequantizeMXFP4: () => dequantizeMXFP4,
  dequantizeMXFP4Expert: () => dequantizeMXFP4Expert,
  dequantizeQ6K: () => dequantizeQ6K,
  dequantizeQ8_0: () => dequantizeQ8_0,
  doRecordMatmulRMSNormFused: () => recordMatmulRMSNormFused,
  exportBenchmarkJSON: () => exportBenchmarkJSON,
  exportProfileJSON: () => exportProfileJSON,
  getCacheStats: () => getCacheStats,
  getKernelConfig: () => getKernelConfig,
  getOrCreateBindGroupLayout: () => getOrCreateBindGroupLayout,
  getOrCreatePipelineLayout: () => getOrCreatePipelineLayout,
  getProfileReport: () => getProfileReport,
  getTunedWorkgroupSize: () => getTunedWorkgroupSize,
  hasRequiredFeatures: () => hasRequiredFeatures,
  isFusedQ4KDisabled: () => isFusedQ4KDisabled,
  isGPUSamplingAvailable: () => isGPUSamplingAvailable,
  isProfilingEnabled: () => isProfilingEnabled,
  loadShaderSource: () => loadShaderSource,
  prewarmKernels: () => prewarmKernels,
  printBenchmarkReport: () => printBenchmarkReport,
  printProfileReport: () => printProfileReport,
  profileAsync: () => profileAsync,
  profileKernel: () => profileKernel,
  profileSync: () => profileSync,
  recordArgmax: () => recordArgmax,
  recordAttention: () => recordAttention,
  recordBiasAdd: () => recordBiasAdd,
  recordCastF16ToF32: () => recordCastF16ToF32,
  recordCastF32ToF16: () => recordCastF32ToF16,
  recordDequantize: () => recordDequantize,
  recordFusedFFN: () => recordFusedFFN,
  recordGather: () => recordGather,
  recordGeLU: () => recordGeLU,
  recordMatmul: () => recordMatmul,
  recordMatmulRMSNormFused: () => recordMatmulRMSNormFused,
  recordMatmulResidualFused: () => recordMatmulResidualFused,
  recordProfileEntry: () => recordProfileEntry,
  recordRMSNorm: () => recordRMSNorm,
  recordResidualAdd: () => recordResidualAdd,
  recordRoPE: () => recordRoPE,
  recordScale: () => recordScale,
  recordSiLU: () => recordSiLU,
  recordSiLURowSplit: () => recordSiLURowSplit,
  recordSoftmax: () => recordSoftmax,
  recordSplitQKV: () => recordSplitQKV,
  runArgmax: () => runArgmax,
  runAttention: () => runAttention,
  runBF16ToF16: () => runBF16ToF16,
  runBF16ToF32: () => runBF16ToF32,
  runBiasAdd: () => runBiasAdd,
  runFusedFFN: () => runFusedFFN,
  runGPUSample: () => runGPUSample,
  runGather: () => runGather,
  runGeLU: () => runGeLU,
  runMatmul: () => runMatmul,
  runMatmulRMSNormFused: () => runMatmulRMSNormFused,
  runMatmulResidualFused: () => runMatmulResidualFused,
  runMoEGather: () => runMoEGather,
  runRMSNorm: () => runRMSNorm,
  runResidualAdd: () => runResidualAdd,
  runRoPE: () => runRoPE,
  runScale: () => runScale,
  runScatterAdd: () => runScatterAdd,
  runScatterAddDynamic: () => runScatterAddDynamic,
  runSiLU: () => runSiLU,
  runSiLURowSplit: () => runSiLURowSplit,
  runSoftmax: () => runSoftmax,
  runSoftmaxTopK: () => runSoftmaxTopK,
  runSplitQKV: () => runSplitQKV,
  runSwiGLURowsplitBias: () => runSwiGLURowsplitBias,
  runTopK: () => runTopK,
  selectDequantKernel: () => selectDequantKernel,
  selectMatmulKernel: () => selectMatmulKernel,
  selectMatmulRMSNormFusedVariant: () => selectMatmulRMSNormFusedVariant,
  selectRMSNormKernel: () => selectRMSNormKernel,
  setProfilingEnabled: () => setProfilingEnabled,
  shouldUseFusedMatmulRMSNorm: () => shouldUseFusedMatmulRMSNorm,
  shouldUseFusedMatmulResidual: () => shouldUseFusedMatmulResidual,
  startProfileSession: () => startProfileSession,
  validateAttentionLimits: () => validateAttentionLimits
});

// src/gpu/kernels/index.ts
init_utils();

// src/gpu/kernels/matmul.ts
init_device();
init_tensor();
init_weight_buffer();
init_debug();
init_buffer_pool();

// src/gpu/kernels/kernel-base.ts
init_dispatch();
init_utils();
var KernelBase = class {
  device;
  constructor(device2) {
    this.device = device2;
  }
  async getPipelineFor(operation, variant, bindGroupLayout = null) {
    return getPipelineFast(operation, variant, bindGroupLayout);
  }
  dispatchKernel(pipeline, bindGroup, workgroups, label) {
    dispatch(this.device, pipeline, bindGroup, workgroups, label);
  }
  recordKernel(recorder, pipeline, bindGroup, workgroups, label) {
    recordDispatch(recorder, pipeline, bindGroup, workgroups, label);
  }
};

// src/gpu/kernels/matmul.ts
init_constants();
init_utils();
init_uniform_cache();
init_schema();
var Q4K_FUSED_VARIANTS = {
  "true/true": "q4_fused_multicol_f16",
  "true/false": "q4_fused_multicol",
  "false/true": "q4_fused_batched_f16",
  "false/false": "q4_fused_batched"
};
function selectQ4KFusedVariant(isM1, wantF16Output) {
  const key = `${isM1}/${wantF16Output}`;
  return Q4K_FUSED_VARIANTS[key] ?? "q4_fused_batched";
}
function isFusedQ4KDisabled() {
  const debugFlags = typeof window !== "undefined" ? window : null;
  if (debugFlags?.DOPPLER_DISABLE_FUSED_Q4K)
    return true;
  if (!isActiveKernelPathFusedQ4K())
    return true;
  return false;
}
function toMatmulDtype(dtype) {
  if (dtype === "f16" || dtype === "bf16")
    return "f16";
  if (dtype === "q4k")
    return "q4k";
  return "f32";
}
function selectMatmulKernel(options = {}) {
  const capabilities = getKernelCapabilities();
  const {
    preferF16 = true,
    useVec4 = false,
    outputDtype = "f32",
    aDtype = null,
    bDtype = null
  } = options;
  const inputsAreF16 = aDtype === "f16" && bDtype === "f16";
  const weightsAreF16 = bDtype === "f16" && aDtype !== "f16";
  if (outputDtype === "f16" && preferF16 && inputsAreF16 && capabilities.hasF16) {
    return useVec4 ? "f16_vec4" : "f16";
  }
  if (preferF16 && weightsAreF16 && capabilities.hasF16) {
    return "f16w_f32a";
  }
  return "f32";
}
var MatmulKernel = class extends KernelBase {
  async getPipeline(variant) {
    return this.getPipelineFor("matmul", variant);
  }
  dispatch(pipeline, bindGroup, workgroups) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, "matmul");
  }
  record(recorder, pipeline, bindGroup, workgroups) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, "matmul");
  }
};
var _transposeDebugCount = 0;
function resolveTransposeB(B, transposeBOption) {
  if (transposeBOption === "auto") {
    const weightLayout = getLayout(B);
    const buffer = getBuffer(B);
    const isColMajor = weightLayout === "column";
    const result = !isColMajor;
    if (isTraceEnabled("kernels") && _transposeDebugCount < 50) {
      _transposeDebugCount++;
      trace.kernels(`resolveTransposeB: layout=${weightLayout}, isColumnMajor=${isColMajor}, transposeB=${result}, bufSize=${buffer.size}`);
    }
    return result;
  }
  return transposeBOption;
}
function validateMatmulDimensions(label, M, N, K) {
  if (!Number.isFinite(M) || !Number.isFinite(N) || !Number.isFinite(K)) {
    throw new Error(`[${label}] Invalid dimensions: M=${M}, N=${N}, K=${K}`);
  }
  if (M <= 0 || N <= 0 || K <= 0) {
    throw new Error(`[${label}] Dimensions must be positive: M=${M}, N=${N}, K=${K}`);
  }
}
function validateMatmulOffsets(label, aOffset, bOffset, cOffset) {
  if (!Number.isFinite(aOffset) || aOffset < 0 || !Number.isFinite(bOffset) || bOffset < 0 || !Number.isFinite(cOffset) || cOffset < 0) {
    throw new Error(`[${label}] Invalid buffer offsets: aOffset=${aOffset}, bOffset=${bOffset}, cOffset=${cOffset}`);
  }
  const storageAlignment = ALIGNMENT.STORAGE;
  if (aOffset % storageAlignment !== 0 || bOffset % storageAlignment !== 0 || cOffset % storageAlignment !== 0) {
    throw new Error(
      `[${label}] Buffer offsets must be ${storageAlignment}-byte aligned: aOffset=${aOffset}, bOffset=${bOffset}, cOffset=${cOffset}`
    );
  }
}
function getMatmulBindingSizes(label, A, B, M, N, K, aDtype, bDtype, transposeB, aOffset, bOffset) {
  const aBytesPerElem = aDtype === "f16" ? 2 : 4;
  const aBindingSize = Math.ceil(M * K * aBytesPerElem / 4) * 4;
  const aRequired = aOffset + aBindingSize;
  if (A.size < aRequired) {
    throw new Error(`[${label}] A buffer too small: ${A.size} < ${aRequired} (M=${M}, K=${K}, aDtype=${aDtype})`);
  }
  const QK_K2 = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  const Q4K_BLOCK_BYTES = QUANTIZATION.Q4K_BLOCK_BYTES;
  let bBindingSize;
  let bRequired;
  if (bDtype === "q4k") {
    const numBlocksPerRow = Math.ceil(K / QK_K2);
    bBindingSize = Math.ceil(N * numBlocksPerRow * Q4K_BLOCK_BYTES / 4) * 4;
    bRequired = bOffset + bBindingSize;
  } else {
    const bBytesPerElem = bDtype === "f16" ? 2 : 4;
    const bElements = transposeB ? N * K : K * N;
    bBindingSize = Math.ceil(bElements * bBytesPerElem / 4) * 4;
    bRequired = bOffset + bBindingSize;
  }
  if (B.size < bRequired) {
    throw new Error(
      `[${label}] B buffer too small: ${B.size} < ${bRequired} (N=${N}, K=${K}, bDtype=${bDtype}, transposeB=${transposeB})`
    );
  }
  return { aBindingSize, bBindingSize };
}
function isQ4KFusedVariant(variant) {
  return variant.startsWith("q4_fused");
}
function isGemvVariant(variant) {
  return variant.startsWith("gemv");
}
function resolveMatmulOverride(variantOverride, M, aDtype, bDtype, capabilities, strict) {
  const override = variantOverride.trim();
  if (!override)
    return null;
  const failOrWarn = (message) => {
    if (strict) {
      throw new Error(message);
    }
    log.warn("Matmul", message);
    return null;
  };
  let config2;
  try {
    config2 = getKernelConfig("matmul", override);
  } catch {
    return failOrWarn(`Unknown matmul kernel variant "${variantOverride}".`);
  }
  if (!hasRequiredFeatures(config2.requires, capabilities)) {
    return failOrWarn(`Matmul kernel "${variantOverride}" requires unsupported GPU features.`);
  }
  const useQ4KFused = isQ4KFusedVariant(override);
  if (useQ4KFused) {
    if (bDtype !== "q4k") {
      return failOrWarn(`Matmul kernel "${variantOverride}" requires Q4K weights but B dtype is ${bDtype}.`);
    }
    if (isFusedQ4KDisabled()) {
      return failOrWarn(`Matmul kernel "${variantOverride}" blocked by DOPPLER_DISABLE_FUSED_Q4K.`);
    }
  }
  const useGemv = isGemvVariant(override);
  if (useGemv && M !== 1) {
    return failOrWarn(`Matmul kernel "${variantOverride}" requires M=1 but got M=${M}.`);
  }
  return { variant: override, useQ4KFused, useGemv };
}
function selectMatmulVariantAndFlags(mode, M, N, K, aDtype, bDtype, transposeB, requestedOutputDtype, options) {
  const capabilities = getKernelCapabilities();
  const strict = getKernelPathStrict();
  const phase = M === 1 ? "decode" : "prefill";
  const pathVariant = getKernelPathMatmulVariant(options.role, phase, options.layerIdx);
  if (pathVariant) {
    const override = resolveMatmulOverride(pathVariant, M, aDtype, bDtype, capabilities, strict);
    if (override) {
      return override;
    }
  }
  let variant = "f32";
  let useQ4KFused = false;
  let useGemv = false;
  if (bDtype === "q4k") {
    const allowFused = !isFusedQ4KDisabled();
    const canFused = capabilities.hasSubgroups && allowFused;
    if (canFused) {
      useQ4KFused = true;
      const wantF16Output = requestedOutputDtype === "f16" && capabilities.hasF16;
      variant = selectQ4KFusedVariant(M === 1, wantF16Output);
    }
  }
  if (!useQ4KFused) {
    const effectiveBDtype = bDtype === "q4k" ? "f32" : bDtype;
    variant = selectMatmulKernel({
      ...options,
      aDtype: aDtype === "q4k" ? "f32" : aDtype,
      bDtype: effectiveBDtype,
      outputDtype: requestedOutputDtype
    });
    useGemv = M === 1 && effectiveBDtype === "f16" && aDtype === "f32";
    if (useGemv) {
      if (capabilities.hasSubgroups) {
        const { multicolThreshold } = getKernelThresholds().matmul;
        if (N > multicolThreshold) {
          variant = "gemv_subgroup_multicol";
        } else {
          variant = "gemv_subgroup";
        }
      } else {
        variant = "gemv";
      }
    }
  }
  return { variant, useQ4KFused, useGemv };
}
function resolveMatmulOutput(variant, M, N, outputBuffer) {
  const config2 = getKernelConfig("matmul", variant);
  const outputsF16 = config2.outputDtype === "f16";
  const elementSize = outputsF16 ? 2 : 4;
  const actualOutputDtype = outputsF16 ? "f16" : "f32";
  const outputSize = M * N * elementSize;
  const cBindingSize = Math.ceil(outputSize / 4) * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "matmul_output");
  return { output, outputSize, cBindingSize, actualOutputDtype };
}
function calculateMatmulDispatch(variant, useQ4KFused, useGemv, M, N, config2) {
  const maxWorkgroups = GPU_LIMITS.MAX_WORKGROUPS;
  const [wgX, wgY] = config2.workgroupSize;
  let workgroupsX = 1;
  let workgroupsY = 1;
  let uniformWorkgroupsX;
  const colsPerWg = config2.variantMetadata?.colsPerWg ?? 4;
  const tileM = config2.variantMetadata?.tileM ?? 4;
  if (useGemv && (variant === "gemv_subgroup" || variant === "gemv_subgroup_multicol")) {
    const gemvWorkgroupsX = Math.ceil(N / colsPerWg);
    if (gemvWorkgroupsX > maxWorkgroups) {
      workgroupsX = maxWorkgroups;
      workgroupsY = Math.ceil(gemvWorkgroupsX / maxWorkgroups);
    } else {
      workgroupsX = gemvWorkgroupsX;
      workgroupsY = 1;
    }
    uniformWorkgroupsX = workgroupsX;
    return { workgroups: [workgroupsX, workgroupsY, 1], uniformWorkgroupsX };
  }
  if (useQ4KFused) {
    if (variant === "q4_fused") {
      workgroupsX = N;
      workgroupsY = 1;
    } else if (config2.variantMetadata?.colsPerWg) {
      workgroupsX = Math.ceil(N / colsPerWg);
      workgroupsY = 1;
    } else if (config2.variantMetadata?.tileM) {
      workgroupsX = N;
      workgroupsY = Math.ceil(M / tileM);
    } else {
      workgroupsX = N;
      workgroupsY = 1;
    }
  } else if (useGemv) {
    workgroupsX = N;
    workgroupsY = 1;
  } else {
    const colsPerThread = variant === "f16_vec4" ? 4 : 1;
    workgroupsX = Math.ceil(M / wgX);
    workgroupsY = Math.ceil(N / (wgY * colsPerThread));
  }
  return { workgroups: [workgroupsX, workgroupsY, 1], uniformWorkgroupsX };
}
function createMatmulUniformBuffer(label, M, N, K, alpha, useQ4KFused, transposeB, uniformWorkgroupsX, recorder, device2) {
  const uniformSize = 32;
  return createUniformBufferWithView(
    label,
    uniformSize,
    (view) => {
      view.setUint32(0, M, true);
      view.setUint32(4, N, true);
      view.setUint32(8, K, true);
      view.setFloat32(12, alpha, true);
      if (useQ4KFused) {
        const numBlocksPerRow = Math.ceil(K / TILE_SIZES.Q4K_SUPER_BLOCK_SIZE);
        view.setUint32(16, numBlocksPerRow, true);
      } else {
        view.setUint32(16, transposeB ? 1 : 0, true);
      }
      view.setUint32(20, uniformWorkgroupsX ?? 0, true);
    },
    recorder,
    device2
  );
}
function createMatmulBindGroupLayout() {
  return getOrCreateBindGroupLayout("matmul_bind_group_layout", [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "uniform" }
    },
    {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" }
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" }
    },
    {
      binding: 3,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" }
    }
  ]);
}
var _runMatmulDebugCount = 0;
async function runMatmul(A, B, M, N, K, options = {}) {
  const device2 = getDevice();
  const {
    alpha = 1,
    outputBuffer = null,
    transposeB: transposeBOption = true,
    // Default: assume row-major (SafeTensors)
    aOffset = 0,
    bOffset = 0,
    cOffset = 0
  } = options;
  const bBuffer = getBuffer(B);
  const weightDtype = getWeightDtype(B);
  if (isTraceEnabled("kernels") && _runMatmulDebugCount < 20) {
    _runMatmulDebugCount++;
    const weightLayout = getLayout(B);
    trace.kernels(`runMatmul: M=${M}, N=${N}, K=${K}, transposeBOption=${transposeBOption}, weightLayout=${weightLayout}, weightDtype=${weightDtype}`);
  }
  const transposeB = resolveTransposeB(B, transposeBOption);
  validateMatmulDimensions("runMatmul", M, N, K);
  const aDtype = toMatmulDtype(A.dtype);
  const bDtype = toMatmulDtype(weightDtype ?? options.bDtype);
  const requestedOutputDtype = options.outputDtype || A.dtype;
  if (isTraceEnabled("kernels") && !weightDtype && !options.bDtype && M <= 2) {
    log.warn("Matmul", `runMatmul: B buffer dtype unknown! size=${bBuffer.size}, M=${M}, N=${N}, K=${K}. Assuming f32.`);
  }
  validateMatmulOffsets("runMatmul", aOffset, bOffset, cOffset);
  const { aBindingSize, bBindingSize } = getMatmulBindingSizes(
    "runMatmul",
    A.buffer,
    bBuffer,
    M,
    N,
    K,
    aDtype,
    bDtype,
    transposeB,
    aOffset,
    bOffset
  );
  const { variant, useQ4KFused, useGemv } = selectMatmulVariantAndFlags(
    "run",
    M,
    N,
    K,
    aDtype,
    bDtype,
    transposeB,
    requestedOutputDtype,
    options
  );
  if (isTraceEnabled("kernels") && bDtype === "q4k") {
    if (useQ4KFused) {
      trace.kernels(`Q4K FUSED: M=${M}, N=${N}, K=${K}, variant=${variant} (WARNING: 2.3x slower than dequant)`);
    } else {
      trace.kernels(`Q4K DEQUANT: M=${M}, N=${N}, K=${K}, will dequant first then matmul with variant=${variant}`);
    }
  }
  if (isTraceEnabled("kernels") && N > 1e5) {
    trace.kernels(`MATMUL_LARGE: N=${N}, variant=${variant}, aDtype=${aDtype}, bDtype=${bDtype}, transposeB=${transposeB}`);
  }
  const config2 = getKernelConfig("matmul", variant);
  const kernel = new MatmulKernel(device2);
  let pipeline = getCachedPipeline("matmul", variant);
  if (!pipeline) {
    pipeline = await createPipeline("matmul", variant);
  }
  const { output: C, outputSize, cBindingSize, actualOutputDtype } = resolveMatmulOutput(
    variant,
    M,
    N,
    outputBuffer
  );
  if (!Number.isFinite(outputSize) || outputSize <= 0) {
    throw new Error(`[runMatmul] Invalid output size: ${outputSize} (M=${M}, N=${N})`);
  }
  const cRequired = cOffset + cBindingSize;
  if (C.size < cRequired) {
    throw new Error(`[runMatmul] Output buffer too small: ${C.size} < ${cRequired} (M=${M}, N=${N})`);
  }
  const dispatchPlan = calculateMatmulDispatch(variant, useQ4KFused, useGemv, M, N, config2);
  const uniformBuffer = createMatmulUniformBuffer(
    "matmul_uniforms",
    M,
    N,
    K,
    alpha,
    useQ4KFused,
    transposeB,
    dispatchPlan.uniformWorkgroupsX,
    null,
    device2
  );
  const isQ4KF16 = variant === "q4_fused_multicol_f16" || variant === "q4_fused_batched_f16";
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: A.buffer, offset: aOffset, size: aBindingSize } },
    { binding: 2, resource: { buffer: bBuffer, offset: bOffset, size: bBindingSize } }
  ];
  if (isQ4KF16) {
    entries.push({ binding: 4, resource: { buffer: C, offset: cOffset, size: cBindingSize } });
  } else {
    entries.push({ binding: 3, resource: { buffer: C, offset: cOffset, size: cBindingSize } });
  }
  const bindGroup = device2.createBindGroup({
    label: "matmul_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries
  });
  kernel.dispatch(pipeline, bindGroup, dispatchPlan.workgroups);
  releaseUniformBuffer(uniformBuffer);
  return createTensor(C, actualOutputDtype, [M, N], "matmul_output");
}
var _recordMatmulDebugCount = 0;
async function recordMatmul(recorder, A, B, M, N, K, options = {}) {
  const device2 = recorder.device;
  const {
    alpha = 1,
    outputBuffer = null,
    transposeB: transposeBOption = true,
    // Default: assume row-major (SafeTensors)
    aOffset = 0,
    bOffset = 0,
    cOffset = 0
  } = options;
  const bBuffer = getBuffer(B);
  const weightDtype = getWeightDtype(B);
  if (isTraceEnabled("kernels") && _recordMatmulDebugCount < 20) {
    _recordMatmulDebugCount++;
    const weightLayout = getLayout(B);
    trace.kernels(`recordMatmul: M=${M}, N=${N}, K=${K}, transposeBOption=${transposeBOption}, weightLayout=${weightLayout}, weightDtype=${weightDtype}`);
  }
  const transposeB = resolveTransposeB(B, transposeBOption);
  validateMatmulDimensions("recordMatmul", M, N, K);
  const aDtype = toMatmulDtype(A.dtype);
  const bDtype = toMatmulDtype(weightDtype ?? options.bDtype);
  const requestedOutputDtype = options.outputDtype || A.dtype;
  validateMatmulOffsets("recordMatmul", aOffset, bOffset, cOffset);
  const { aBindingSize, bBindingSize } = getMatmulBindingSizes(
    "recordMatmul",
    A.buffer,
    bBuffer,
    M,
    N,
    K,
    aDtype,
    bDtype,
    transposeB,
    aOffset,
    bOffset
  );
  const { variant, useQ4KFused, useGemv } = selectMatmulVariantAndFlags(
    "record",
    M,
    N,
    K,
    aDtype,
    bDtype,
    transposeB,
    requestedOutputDtype,
    options
  );
  const config2 = getKernelConfig("matmul", variant);
  const kernel = new MatmulKernel(device2);
  let pipeline = getCachedPipeline("matmul", variant);
  if (!pipeline) {
    pipeline = await createPipeline("matmul", variant);
  }
  const { output: C, outputSize, cBindingSize, actualOutputDtype } = resolveMatmulOutput(
    variant,
    M,
    N,
    outputBuffer
  );
  if (!Number.isFinite(outputSize) || outputSize <= 0) {
    throw new Error(`[recordMatmul] Invalid output size: ${outputSize} (M=${M}, N=${N})`);
  }
  const cRequired = cOffset + cBindingSize;
  if (C.size < cRequired) {
    throw new Error(`[recordMatmul] Output buffer too small: ${C.size} < ${cRequired} (M=${M}, N=${N})`);
  }
  const dispatchPlan = calculateMatmulDispatch(variant, useQ4KFused, useGemv, M, N, config2);
  const uniformBuffer = createMatmulUniformBuffer(
    "matmul_uniforms",
    M,
    N,
    K,
    alpha,
    useQ4KFused,
    transposeB,
    dispatchPlan.uniformWorkgroupsX,
    recorder,
    device2
  );
  const isQ4KF16 = variant === "q4_fused_multicol_f16" || variant === "q4_fused_batched_f16";
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: A.buffer, offset: aOffset, size: aBindingSize } },
    { binding: 2, resource: { buffer: bBuffer, offset: bOffset, size: bBindingSize } }
  ];
  if (isQ4KF16) {
    entries.push({ binding: 4, resource: { buffer: C, offset: cOffset, size: cBindingSize } });
  } else {
    entries.push({ binding: 3, resource: { buffer: C, offset: cOffset, size: cBindingSize } });
  }
  const bindGroup = device2.createBindGroup({
    label: "matmul_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries
  });
  kernel.record(recorder, pipeline, bindGroup, dispatchPlan.workgroups);
  return createTensor(C, actualOutputDtype, [M, N], "matmul_output");
}

// src/gpu/kernels/dequant.ts
init_device();
init_buffer_pool();
init_tensor();
init_constants();

// src/loader/quantization-constants.ts
var Q8_0_BLOCK_SIZE = 32;

// src/gpu/kernels/dequant.ts
init_dispatch();
init_utils();
init_uniform_cache();
function selectDequantKernel(options = {}) {
  const capabilities = getKernelCapabilities();
  const { useVec4 = true, outputDtype = "f32" } = options;
  const wantsF16Out = outputDtype === "f16" && capabilities.hasF16;
  if (capabilities.hasSubgroups) {
    if (wantsF16Out) {
      return useVec4 ? "subgroup_vec4_f16out" : "subgroup_f16out";
    }
    return useVec4 ? "subgroup_vec4" : "subgroup";
  }
  if (wantsF16Out) {
    return useVec4 ? "shared_vec4_f16out" : "shared_f16out";
  }
  return useVec4 ? "shared_vec4" : "shared";
}
function calculateDequantWorkgroups(variant, numBlocks) {
  const QK_K2 = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  let workgroups;
  if (variant.includes("vec4")) {
    workgroups = numBlocks;
  } else if (variant.includes("shared")) {
    workgroups = numBlocks;
  } else {
    workgroups = Math.ceil(numBlocks * QK_K2 / (WORKGROUP_SIZES.DEFAULT / 4));
  }
  const maxWorkgroups = GPU_LIMITS.MAX_WORKGROUPS;
  if (workgroups <= maxWorkgroups) {
    return [workgroups, 1, 1];
  }
  const wgY = Math.ceil(workgroups / maxWorkgroups);
  const wgX = Math.min(workgroups, maxWorkgroups);
  return [wgX, wgY, 1];
}
function createDequantBindGroupLayout() {
  return getOrCreateBindGroupLayout("dequant_bind_group_layout", [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "uniform" }
    },
    {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" }
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" }
    }
  ]);
}
async function dequantize(quantized, numBlocks, options = {}) {
  const device2 = getDevice();
  const {
    outputOffset = 0,
    outputBuffer = null,
    outputDtype = "f32"
  } = options;
  const variant = selectDequantKernel({ ...options, outputDtype });
  const pipeline = await getPipelineFast("dequant", variant);
  const QK_K2 = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  const bytesPerElem = outputDtype === "f16" ? 2 : 4;
  const outputSize = numBlocks * QK_K2 * bytesPerElem;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "dequant_output");
  const uniformBuffer = createUniformBufferWithView(
    "dequant_uniforms",
    16,
    (view) => {
      view.setUint32(0, numBlocks, true);
      view.setUint32(4, outputOffset, true);
      view.setUint32(8, 0, true);
      view.setUint32(12, 0, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "dequant_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: quantized } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = calculateDequantWorkgroups(variant, numBlocks);
  dispatch(device2, pipeline, bindGroup, workgroups, "dequant");
  releaseUniformBuffer(uniformBuffer);
  const dtype = outputDtype === "f16" ? "f16" : "f32";
  return createTensor(output, dtype, [numBlocks * QK_K2], "dequant_output");
}
async function dequantizeMXFP4(blocks, scales, totalElements, numGroups, options = {}) {
  const device2 = getDevice();
  const {
    outputBuffer = null,
    groupSize = 32
    // 32 elements per group (16 bytes * 2 nibbles)
  } = options;
  const pipeline = await getPipelineFast("dequant", "mxfp4");
  const outputSize = totalElements * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "mxfp4_dequant_output");
  const uniformBuffer = createUniformBufferWithView(
    "mxfp4_dequant_uniforms",
    16,
    (view) => {
      view.setUint32(0, totalElements, true);
      view.setUint32(4, numGroups, true);
      view.setUint32(8, groupSize, true);
      view.setUint32(12, numGroups * groupSize, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "mxfp4_dequant_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: blocks } },
      { binding: 2, resource: { buffer: scales } },
      { binding: 3, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(totalElements / WORKGROUP_SIZES.DEFAULT);
  const dispatchSize = [
    Math.min(workgroups, GPU_LIMITS.MAX_WORKGROUPS),
    Math.max(1, Math.ceil(workgroups / GPU_LIMITS.MAX_WORKGROUPS)),
    1
  ];
  dispatch(device2, pipeline, bindGroup, dispatchSize, "mxfp4_dequant");
  releaseUniformBuffer(uniformBuffer);
  return createTensor(output, "f32", [totalElements], "mxfp4_dequant_output");
}
async function dequantizeMXFP4Expert(blocks, scales, expertIdx, numExperts, outDim, numGroups, options = {}) {
  const device2 = getDevice();
  const { outputBuffer = null } = options;
  const pipeline = await getPipelineFast("dequant", "mxfp4_expert");
  const totalOutput = outDim * numGroups * 32;
  const outputSize = totalOutput * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "mxfp4_expert_output");
  const uniformBuffer = createUniformBufferWithView(
    "mxfp4_expert_uniforms",
    32,
    (view) => {
      view.setUint32(0, expertIdx, true);
      view.setUint32(4, numExperts, true);
      view.setUint32(8, outDim, true);
      view.setUint32(12, numGroups, true);
      view.setUint32(16, totalOutput, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "mxfp4_expert_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: blocks } },
      { binding: 2, resource: { buffer: scales } },
      { binding: 3, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(totalOutput / WORKGROUP_SIZES.DEFAULT);
  const dispatchSize = [
    Math.min(workgroups, GPU_LIMITS.MAX_WORKGROUPS),
    Math.max(1, Math.ceil(workgroups / GPU_LIMITS.MAX_WORKGROUPS)),
    1
  ];
  dispatch(device2, pipeline, bindGroup, dispatchSize, "mxfp4_expert");
  releaseUniformBuffer(uniformBuffer);
  return createTensor(output, "f32", [outDim, numGroups * 32], "mxfp4_expert_output");
}
async function dequantizeQ6K(quantized, numBlocks, options = {}) {
  const device2 = getDevice();
  const {
    outputOffset = 0,
    outputBuffer = null,
    outputDtype = "f16"
    // Q6_K always outputs f16 for now
  } = options;
  const pipeline = await getPipelineFast("dequant", "q6k_f16out");
  const QK_K2 = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  const bytesPerElem = outputDtype === "f16" ? 2 : 4;
  const outputSize = numBlocks * QK_K2 * bytesPerElem;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "q6k_dequant_output");
  const maxWorkgroups = GPU_LIMITS.MAX_WORKGROUPS;
  const workgroupsX = Math.min(numBlocks, maxWorkgroups);
  const uniformBuffer = createUniformBufferWithView(
    "q6k_dequant_uniforms",
    16,
    (view) => {
      view.setUint32(0, numBlocks, true);
      view.setUint32(4, outputOffset, true);
      view.setUint32(8, workgroupsX, true);
      view.setUint32(12, 0, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "q6k_dequant_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: quantized } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = [
    workgroupsX,
    numBlocks > maxWorkgroups ? Math.ceil(numBlocks / maxWorkgroups) : 1,
    1
  ];
  dispatch(device2, pipeline, bindGroup, workgroups, "q6k_dequant");
  releaseUniformBuffer(uniformBuffer);
  const dtype = outputDtype === "f16" ? "f16" : "f32";
  return createTensor(output, dtype, [numBlocks * QK_K2], "q6k_dequant_output");
}
async function dequantizeQ8_0(quantized, numBlocks, options = {}) {
  const device2 = getDevice();
  const {
    outputOffset = 0,
    outputBuffer = null,
    outputDtype = "f16"
    // Q8_0 outputs f16 for now
  } = options;
  const pipeline = await getPipelineFast("dequant", "q8_0_f16out");
  const bytesPerElem = outputDtype === "f16" ? 2 : 4;
  const outputSize = numBlocks * Q8_0_BLOCK_SIZE * bytesPerElem;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "q8_0_dequant_output");
  const maxWorkgroups = GPU_LIMITS.MAX_WORKGROUPS;
  const workgroupsX = Math.min(numBlocks, maxWorkgroups);
  const uniformBuffer = createUniformBufferWithView(
    "q8_0_dequant_uniforms",
    16,
    (view) => {
      view.setUint32(0, numBlocks, true);
      view.setUint32(4, outputOffset, true);
      view.setUint32(8, workgroupsX, true);
      view.setUint32(12, 0, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "q8_0_dequant_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: quantized } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = [
    workgroupsX,
    numBlocks > maxWorkgroups ? Math.ceil(numBlocks / maxWorkgroups) : 1,
    1
  ];
  dispatch(device2, pipeline, bindGroup, workgroups, "q8_0_dequant");
  releaseUniformBuffer(uniformBuffer);
  const dtype = outputDtype === "f16" ? "f16" : "f32";
  return createTensor(output, dtype, [numBlocks * Q8_0_BLOCK_SIZE], "q8_0_dequant_output");
}
async function recordDequantize(recorder, quantized, numBlocks, options = {}) {
  const device2 = recorder.device;
  const {
    outputOffset = 0,
    outputBuffer = null,
    outputDtype = "f32"
  } = options;
  const variant = selectDequantKernel({ ...options, outputDtype });
  const pipeline = await getPipelineFast("dequant", variant);
  const QK_K2 = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  const bytesPerElem = outputDtype === "f16" ? 2 : 4;
  const outputSize = numBlocks * QK_K2 * bytesPerElem;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "dequant_output");
  const uniformBuffer = createUniformBufferWithView(
    "dequant_uniforms",
    16,
    (view) => {
      view.setUint32(0, numBlocks, true);
      view.setUint32(4, outputOffset, true);
    },
    recorder
  );
  const bindGroup = device2.createBindGroup({
    label: "dequant_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: quantized } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = calculateDequantWorkgroups(variant, numBlocks);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "dequant");
  const dtype = outputDtype === "f16" ? "f16" : "f32";
  return createTensor(output, dtype, [numBlocks * QK_K2], "dequant_output");
}

// src/gpu/kernels/attention.ts
init_device();
init_buffer_pool();
init_tensor();
init_constants();
init_kernel_thresholds_schema();
init_utils();
init_dispatch();
init_uniform_cache();
init_debug();
var loggedAttentionTier = false;
var _chunkedMaxKVLen = null;
function getChunkedMaxKVLen() {
  if (_chunkedMaxKVLen === null) {
    const config2 = getKernelConfig("attention", "decode_chunked_f16kv");
    _chunkedMaxKVLen = config2.variantMetadata?.maxKVLen ?? 2048;
  }
  return _chunkedMaxKVLen;
}
var kvLenFallbackBuffer = null;
function getKvLenFallbackBuffer(device2) {
  if (!kvLenFallbackBuffer) {
    kvLenFallbackBuffer = device2.createBuffer({
      label: "attention_kv_len_fallback",
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device2.queue.writeBuffer(kvLenFallbackBuffer, 0, new Uint32Array([0]));
  }
  return kvLenFallbackBuffer;
}
var AttentionKernel = class extends KernelBase {
  async getPipeline(variant) {
    return this.getPipelineFor("attention", variant);
  }
  dispatch(pipeline, bindGroup, workgroups) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, "attention");
  }
  record(recorder, pipeline, bindGroup, workgroups) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, "attention");
  }
};
function selectAttentionTier(headDim, seqLen, useF16KV, forcedTier, sharedLimit, caps, strict) {
  const isDecode = seqLen === 1;
  const canLarge = headDim <= DIMENSION_LIMITS.ATTENTION_LARGE_MAX_HEAD_DIM && sharedLimit >= MEMORY_THRESHOLDS.ATTENTION_LARGE_SHARED;
  const smallRequired = useF16KV ? MEMORY_THRESHOLDS.ATTENTION_SMALL_SHARED_F16 : MEMORY_THRESHOLDS.ATTENTION_SMALL_SHARED_F32;
  const canSmall = headDim <= DIMENSION_LIMITS.ATTENTION_SMALL_MAX_HEAD_DIM && sharedLimit >= smallRequired;
  const canSubgroup = caps.hasSubgroups && headDim <= DIMENSION_LIMITS.ATTENTION_SUBGROUP_MAX_HEAD_DIM && sharedLimit >= MEMORY_THRESHOLDS.ATTENTION_SUBGROUP_SHARED && isDecode;
  const failOrWarn = (message) => {
    if (strict) {
      throw new Error(message);
    }
    log.warn("Attention", message);
  };
  let tier = forcedTier;
  if (tier === "tiled_large" && !canLarge) {
    failOrWarn(`Requested tiled_large but device doesn't support it (headDim=${headDim}, shared=${sharedLimit}).`);
    tier = null;
  }
  if (tier === "tiled_small" && !canSmall) {
    failOrWarn(`Requested tiled_small but device doesn't support it (headDim=${headDim}, shared=${sharedLimit}).`);
    tier = null;
  }
  if (tier === "subgroup" && !canSubgroup) {
    failOrWarn(`Requested subgroup attention but device doesn't support it (headDim=${headDim}, shared=${sharedLimit}, subgroups=${caps.hasSubgroups}).`);
    tier = null;
  }
  if (!tier) {
    if (canSubgroup) {
      tier = "subgroup";
      if (!loggedAttentionTier) {
        trace.attn(0, `Using subgroup decode kernel (headDim=${headDim}, hasSubgroups=true)`);
        loggedAttentionTier = true;
      }
    } else if (canLarge) {
      tier = "tiled_large";
    } else if (canSmall) {
      tier = "tiled_small";
    } else if (isDecode) {
      tier = "streaming";
    } else {
      log.warn("Attention", `No tiled kernel fits prefill (headDim=${headDim}, shared=${sharedLimit}). Falling back to streaming. Expect slow prefill.`);
      tier = "streaming";
    }
  }
  return tier;
}
var loggedChunkedKernel = false;
function resolveAttentionVariant(tier, isDecode, useF16KV, numHeads, headDim, kvLen) {
  const base = isDecode ? "decode" : "prefill";
  const chunkedMaxKVLen = getChunkedMaxKVLen();
  const minHeadDimForChunked = getKernelThresholds().attention.minHeadDimForChunked;
  const canUseChunked = isDecode && useF16KV && headDim >= minHeadDimForChunked && kvLen <= chunkedMaxKVLen;
  const decodeSubgroupMaxKVLen = chunkedMaxKVLen;
  const decodeSubgroupMaxHeadDim = getKernelThresholds().attention.tierHeadDimLimits.tier1;
  const canUseDecodeSubgroup = isDecode && !useF16KV && headDim <= decodeSubgroupMaxHeadDim && kvLen <= decodeSubgroupMaxKVLen;
  if (tier === "subgroup") {
    if (useF16KV) {
      if (canUseChunked) {
        if (!loggedChunkedKernel) {
          trace.attn(0, `Using chunked decode kernel (headDim=${headDim}, numHeads=${numHeads}, f16kv=true)`);
          loggedChunkedKernel = true;
        }
        return "decode_chunked_f16kv";
      }
      return "decode_streaming_f16kv";
    }
    if (canUseDecodeSubgroup) {
      return "decode_subgroup";
    }
    return "decode_streaming";
  }
  if (tier === "tiled_large") {
    return base + (useF16KV ? "_f16kv" : "");
  }
  if (tier === "tiled_small") {
    return `${base}_small${useF16KV ? "_f16kv" : ""}`;
  }
  if (canUseChunked) {
    if (!loggedChunkedKernel) {
      trace.attn(0, `Using chunked decode kernel (headDim=${headDim}, numHeads=${numHeads}, f16kv=true)`);
      loggedChunkedKernel = true;
    }
    return "decode_chunked_f16kv";
  }
  return `${base}_streaming${useF16KV ? "_f16kv" : ""}`;
}
function calculateAttentionWorkgroups(tier, seqLen, numHeads) {
  if (tier === "subgroup") {
    return numHeads;
  }
  if (tier === "streaming") {
    return seqLen * numHeads;
  }
  if (tier === "tiled_large") {
    return Math.ceil(seqLen / TILE_SIZES.ATTENTION_LARGE_BLOCK_SIZE) * numHeads;
  }
  return Math.ceil(seqLen / TILE_SIZES.ATTENTION_SMALL_BLOCK_SIZE) * numHeads;
}
function inferAttentionTierFromVariant(variant) {
  if (variant === "decode_subgroup")
    return "subgroup";
  if (variant.startsWith("prefill_streaming") || variant.startsWith("decode_streaming") || variant === "decode_chunked_f16kv") {
    return "streaming";
  }
  if (variant.startsWith("prefill_small") || variant.startsWith("decode_small"))
    return "tiled_small";
  return "tiled_large";
}
function validateAttentionVariant(variant, isDecode, useF16KV, caps, strict) {
  const normalized = variant.trim();
  const failOrWarn = (message) => {
    if (strict) {
      throw new Error(message);
    }
    log.warn("Attention", message);
    return null;
  };
  let config2;
  try {
    config2 = getKernelConfig("attention", normalized);
  } catch {
    return failOrWarn(`Unknown attention kernel variant "${variant}".`);
  }
  if (!hasRequiredFeatures(config2.requires, caps)) {
    return failOrWarn(`Attention kernel "${variant}" requires unsupported GPU features.`);
  }
  const expectsF16KV = normalized.includes("_f16kv");
  if (expectsF16KV !== useF16KV) {
    const kvLabel = useF16KV ? "f16" : "f32";
    return failOrWarn(`Attention kernel "${variant}" incompatible with ${kvLabel} KV cache.`);
  }
  const isDecodeVariant = normalized.startsWith("decode");
  const isPrefillVariant = normalized.startsWith("prefill");
  if (isDecode && isPrefillVariant) {
    return failOrWarn(`Attention kernel "${variant}" is prefill-only but decode requested.`);
  }
  if (!isDecode && isDecodeVariant) {
    return failOrWarn(`Attention kernel "${variant}" is decode-only but prefill requested.`);
  }
  return normalized;
}
function resolveAttentionPlan(seqLen, kvLen, headDim, numHeads, kvDtype, sharedLimit, caps, layerIdx) {
  const useF16KV = kvDtype === "f16";
  const isDecode = seqLen === 1;
  const strict = getKernelPathStrict();
  const pathVariant = getKernelPathAttentionVariant(isDecode ? "decode" : "prefill", layerIdx);
  if (pathVariant) {
    const variantOverride = validateAttentionVariant(pathVariant, isDecode, useF16KV, caps, strict);
    if (variantOverride) {
      const tier2 = inferAttentionTierFromVariant(variantOverride);
      const workgroups2 = calculateAttentionWorkgroups(tier2, seqLen, numHeads);
      return { tier: tier2, variant: variantOverride, workgroups: workgroups2, useF16KV, isDecode };
    }
  }
  const tier = selectAttentionTier(headDim, seqLen, useF16KV, null, sharedLimit, caps, strict);
  const variant = resolveAttentionVariant(tier, isDecode, useF16KV, numHeads, headDim, kvLen);
  const workgroups = calculateAttentionWorkgroups(tier, seqLen, numHeads);
  return { tier, variant, workgroups, useF16KV, isDecode };
}
function createAttentionUniformBuffer(device2, recorder, params) {
  return createUniformBufferWithView(
    "attention_uniforms",
    48,
    // 44 bytes used + 4 padding for 16-byte alignment
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.kvLen, true);
      view.setUint32(16, params.seqLen, true);
      view.setFloat32(20, params.scale, true);
      view.setUint32(24, params.causal ? 1 : 0, true);
      view.setUint32(28, params.startPos, true);
      view.setFloat32(32, params.attnSoftcap, true);
      view.setUint32(36, params.slidingWindow, true);
      view.setUint32(40, params.kvLenSource, true);
    },
    recorder,
    device2
  );
}
async function runAttention(Q, K, V, mask, numHeads, headDim, options = {}) {
  const device2 = getDevice();
  const {
    seqLen = 1,
    kvLen = seqLen,
    numKVHeads = numHeads,
    scale = 1 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    layerIdx,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    kvLenBuffer = null,
    indirectBuffer = null,
    indirectOffset = 0
  } = options;
  const limits = getDeviceLimits();
  const sharedLimit = limits?.maxComputeWorkgroupStorageSize ?? Infinity;
  const caps = getKernelCapabilities();
  const kvDtype = K.dtype;
  const plan = resolveAttentionPlan(
    seqLen,
    kvLen,
    headDim,
    numHeads,
    kvDtype,
    sharedLimit,
    caps,
    layerIdx
  );
  const kernel = new AttentionKernel(device2);
  const pipeline = await kernel.getPipeline(plan.variant);
  const outputDtype = "f32";
  const outputSize = seqLen * numHeads * headDim * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, void 0, "attention_output");
  const uniformBuffer = createAttentionUniformBuffer(device2, null, {
    numHeads,
    numKVHeads,
    headDim,
    kvLen,
    seqLen,
    scale,
    causal,
    startPos,
    attnSoftcap,
    slidingWindow,
    kvLenSource: kvLenBuffer ? 1 : 0
  });
  const kvLenBinding = kvLenBuffer || getKvLenFallbackBuffer(device2);
  const bindGroup = device2.createBindGroup({
    label: "attention_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: K.buffer } },
      { binding: 3, resource: { buffer: V.buffer } },
      { binding: 4, resource: { buffer: outputBuf } },
      { binding: 5, resource: { buffer: kvLenBinding } }
    ]
  });
  if (!indirectBuffer && limits && plan.workgroups > limits.maxComputeWorkgroupsPerDimension) {
    throw new Error(
      `Attention dispatch requires ${plan.workgroups} workgroups but device limit is ${limits.maxComputeWorkgroupsPerDimension}. Reduce prompt length or use streaming attention.`
    );
  }
  if (indirectBuffer) {
    dispatchIndirect(device2, pipeline, bindGroup, indirectBuffer, indirectOffset, "attention");
  } else {
    kernel.dispatch(pipeline, bindGroup, plan.workgroups);
  }
  releaseUniformBuffer(uniformBuffer);
  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], "attention_output");
}
async function recordAttention(recorder, Q, K, V, mask, numHeads, headDim, options = {}) {
  const device2 = recorder.device;
  const {
    seqLen = 1,
    kvLen = seqLen,
    numKVHeads = numHeads,
    scale = 1 / Math.sqrt(headDim),
    causal = true,
    startPos = 0,
    layerIdx,
    outputBuffer = null,
    attnSoftcap = 0,
    slidingWindow = 0,
    kvLenBuffer = null,
    indirectBuffer = null,
    indirectOffset = 0
  } = options;
  const limits = getDeviceLimits();
  const sharedLimit = limits?.maxComputeWorkgroupStorageSize ?? Infinity;
  const caps = getKernelCapabilities();
  const kvDtype = K.dtype;
  const plan = resolveAttentionPlan(
    seqLen,
    kvLen,
    headDim,
    numHeads,
    kvDtype,
    sharedLimit,
    caps,
    layerIdx
  );
  trace.attn(0, `recordAttention: isDecode=${plan.isDecode}, tier=${plan.tier}, variant=${plan.variant}, seqLen=${seqLen}, kvLen=${kvLen}, numHeads=${numHeads}, headDim=${headDim}, useF16KV=${plan.useF16KV}`);
  const kernel = new AttentionKernel(device2);
  const pipeline = await kernel.getPipeline(plan.variant);
  const outputDtype = "f32";
  const outputSize = seqLen * numHeads * headDim * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, void 0, "attention_output");
  const uniformBuffer = createAttentionUniformBuffer(device2, recorder, {
    numHeads,
    numKVHeads,
    headDim,
    kvLen,
    seqLen,
    scale,
    causal,
    startPos,
    attnSoftcap,
    slidingWindow,
    kvLenSource: kvLenBuffer ? 1 : 0
  });
  const kvLenBinding = kvLenBuffer || getKvLenFallbackBuffer(device2);
  const bindGroup = device2.createBindGroup({
    label: "attention_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q.buffer } },
      { binding: 2, resource: { buffer: K.buffer } },
      { binding: 3, resource: { buffer: V.buffer } },
      { binding: 4, resource: { buffer: outputBuf } },
      { binding: 5, resource: { buffer: kvLenBinding } }
    ]
  });
  if (!indirectBuffer && limits && plan.workgroups > limits.maxComputeWorkgroupsPerDimension) {
    throw new Error(
      `Attention dispatch requires ${plan.workgroups} workgroups but device limit is ${limits.maxComputeWorkgroupsPerDimension}. Reduce prompt length or use streaming attention.`
    );
  }
  if (indirectBuffer) {
    recordDispatchIndirect(recorder, pipeline, bindGroup, indirectBuffer, indirectOffset, "attention");
  } else {
    kernel.record(recorder, pipeline, bindGroup, plan.workgroups);
  }
  return createTensor(outputBuf, outputDtype, [seqLen, numHeads, headDim], "attention_output");
}

// src/gpu/kernels/rmsnorm.ts
init_device();
init_buffer_pool();
init_tensor();
init_dispatch();
init_utils();
init_debug();
init_schema();
function canUseF16(input, residual) {
  if (input.dtype !== "f16")
    return false;
  if (residual && residual.dtype !== "f16")
    return false;
  return true;
}
function selectRMSNormKernel(options = {}, isF16 = false) {
  const { residual = null, hiddenSize = null } = options;
  const { smallThreshold } = getKernelThresholds().rmsnorm;
  const caps = getKernelCapabilities();
  const hasSubgroups = caps?.hasSubgroups ?? false;
  if (isF16) {
    if (hiddenSize !== null && hiddenSize <= smallThreshold) {
      return "small_f16";
    }
    return "default_f16";
  }
  if (residual) {
    if (hiddenSize !== null && hiddenSize <= smallThreshold) {
      return "residual_small";
    }
    return "residual";
  }
  if (hasSubgroups) {
    if (hiddenSize !== null && hiddenSize <= smallThreshold) {
      return "small_subgroup";
    }
    return "subgroup";
  }
  if (hiddenSize !== null && hiddenSize <= smallThreshold) {
    return "small";
  }
  return "default";
}
async function runRMSNorm(input, weight, eps = 1e-5, options = {}) {
  const device2 = getDevice();
  const { batchSize = 1, hiddenSize, residual = null, outputBuffer = null } = options;
  const isF16 = canUseF16(input, residual);
  const variant = selectRMSNormKernel(options, isF16);
  trace.kernels(`RMSNorm: input.dtype=${input.dtype}, isF16=${isF16}, variant=${variant}`);
  if (residual) {
    trace.kernels(`RMSNorm: Using residual variant, residual.size=${residual.buffer.size}, inferredHiddenSize=${hiddenSize || weight.size / 4}, batchSize=${batchSize}`);
  }
  const pipeline = await getPipelineFast("rmsnorm", variant);
  const inferredHiddenSize = hiddenSize || weight.size / 4;
  const bytesPerElement = isF16 ? 2 : 4;
  const outputSize = batchSize * inferredHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, void 0, "rmsnorm_output");
  const hasResidualFlag = residual ? 1 : 0;
  const uniformBuffer = createUniformBufferWithView(
    "rmsnorm_uniforms",
    16,
    (view) => {
      view.setUint32(0, inferredHiddenSize, true);
      view.setUint32(4, batchSize, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, hasResidualFlag, true);
    },
    null,
    device2
  );
  if (hasResidualFlag) {
    trace.kernels(`RMSNorm: Uniform hasResidual=${hasResidualFlag}, hiddenSize=${inferredHiddenSize}, batchSize=${batchSize}`);
  }
  const residualBuffer = residual?.buffer || device2.createBuffer({
    label: "rmsnorm_residual_placeholder",
    size: 4,
    usage: GPUBufferUsage.STORAGE
  });
  const bindGroup = device2.createBindGroup({
    label: "rmsnorm_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: outputBuf } },
      { binding: 4, resource: { buffer: residualBuffer } }
    ]
  });
  dispatch(device2, pipeline, bindGroup, batchSize, "rmsnorm");
  uniformBuffer.destroy();
  if (!residual)
    residualBuffer.destroy();
  return createTensor(outputBuf, input.dtype, [batchSize, inferredHiddenSize], "rmsnorm_output");
}
async function recordRMSNorm(recorder, input, weight, eps = 1e-5, options = {}) {
  const device2 = recorder.device;
  const {
    batchSize = 1,
    hiddenSize = null,
    residual = null,
    outputBuffer = null
  } = options;
  const inferredHiddenSize = hiddenSize || weight.size / 4;
  const isF16 = canUseF16(input, residual);
  const variant = selectRMSNormKernel(options, isF16);
  const bytesPerElement = isF16 ? 2 : 4;
  const outputSize = batchSize * inferredHiddenSize * bytesPerElement;
  const pipeline = await getPipelineFast("rmsnorm", variant);
  const outputBuf = outputBuffer || acquireBuffer(outputSize, void 0, "rmsnorm_output");
  const uniformBuffer = createUniformBufferWithView(
    "rmsnorm_uniforms",
    16,
    (view) => {
      view.setUint32(0, inferredHiddenSize, true);
      view.setUint32(4, batchSize, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, residual ? 1 : 0, true);
    },
    recorder
  );
  const residualBuffer = residual?.buffer || device2.createBuffer({
    label: "rmsnorm_residual_placeholder",
    size: 4,
    usage: GPUBufferUsage.STORAGE
  });
  const bindGroup = device2.createBindGroup({
    label: "rmsnorm_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: outputBuf } },
      { binding: 4, resource: { buffer: residualBuffer } }
    ]
  });
  recordDispatch(recorder, pipeline, bindGroup, batchSize, "rmsnorm");
  if (!residual) {
    recorder.trackTemporaryBuffer(residualBuffer);
  }
  return createTensor(outputBuf, input.dtype, [batchSize, inferredHiddenSize], "rmsnorm_output");
}

// src/gpu/kernels/softmax.ts
init_device();
init_buffer_pool();
init_tensor();
init_dispatch();
init_utils();
init_debug();
var SOFTMAX_SMALL_THRESHOLD = 256;
function selectSoftmaxVariant(innerSize) {
  const caps = getKernelCapabilities();
  const hasSubgroups = caps?.hasSubgroups ?? false;
  if (hasSubgroups) {
    if (innerSize <= SOFTMAX_SMALL_THRESHOLD) {
      return "small_subgroup";
    }
    return "subgroup";
  }
  if (innerSize <= SOFTMAX_SMALL_THRESHOLD) {
    return "small";
  }
  return "default";
}
async function runSoftmax(input, axis, options = {}) {
  const device2 = getDevice();
  const { batchSize = 1, size, temperature = 1, outputBuffer = null } = options;
  const bytesPerElement = input.dtype === "f16" ? 2 : 4;
  const inferredSize = size || input.buffer.size / (batchSize * bytesPerElement);
  const variant = selectSoftmaxVariant(inferredSize);
  trace.kernels(`Softmax: size=${inferredSize}, variant=${variant}`);
  const pipeline = await createPipeline("softmax", variant);
  const outputSize = batchSize * inferredSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "softmax_output");
  const uniformBuffer = createUniformBufferWithView(
    "softmax_uniforms",
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
      view.setUint32(4, batchSize, true);
      view.setFloat32(8, temperature, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "softmax_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  dispatch(device2, pipeline, bindGroup, batchSize, "softmax");
  uniformBuffer.destroy();
  return createTensor(output, input.dtype, [batchSize, inferredSize], "softmax_output");
}
async function runSoftmaxTopK(logits, numTokens, numExperts, topK, options = {}) {
  const device2 = getDevice();
  const { normalize = true } = options;
  const pipeline = await createPipeline("topk", "fused");
  const indicesSize = numTokens * topK * 4;
  const weightsSize = numTokens * topK * 4;
  const indices = acquireBuffer(indicesSize, void 0, "softmax_topk_indices");
  const weights = acquireBuffer(weightsSize, void 0, "softmax_topk_weights");
  const uniformBuffer = createUniformBufferWithView(
    "softmax_topk_uniforms",
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, numExperts, true);
      view.setUint32(8, topK, true);
      view.setUint32(12, normalize ? 1 : 0, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "softmax_topk_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } }
    ]
  });
  dispatch(device2, pipeline, bindGroup, numTokens, "softmax_topk");
  uniformBuffer.destroy();
  return { indices, weights };
}
async function recordSoftmax(recorder, input, axis, options = {}) {
  const device2 = recorder.device;
  const {
    batchSize = 1,
    seqLen = null,
    outputBuffer = null
  } = options;
  const bytesPerElement = input.dtype === "f16" ? 2 : 4;
  const inferredSeqLen = seqLen || input.buffer.size / (batchSize * bytesPerElement);
  const pipeline = await createPipeline("softmax", "default");
  const outputSize = batchSize * inferredSeqLen * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "softmax_output");
  const uniformBuffer = createUniformBufferWithView(
    "softmax_uniforms",
    16,
    (view) => {
      view.setUint32(0, inferredSeqLen, true);
      view.setUint32(4, batchSize, true);
      view.setFloat32(8, 1, true);
    },
    recorder
  );
  const bindGroup = device2.createBindGroup({
    label: "softmax_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  recordDispatch(recorder, pipeline, bindGroup, batchSize, "softmax");
  return createTensor(output, input.dtype, [batchSize, inferredSeqLen], "softmax_output");
}

// src/gpu/kernels/rope.ts
init_device();
init_tensor();
init_constants();
init_dispatch();
init_utils();
init_schema();
var getRopeDefaults = () => getKernelThresholds().rope;
async function runRoPE(input, freqsCos, freqsSin, seqLen, options = {}) {
  const device2 = getDevice();
  const ropeDefaults = getRopeDefaults();
  const {
    numHeads = 1,
    headDim = 64,
    ropeTheta = ropeDefaults.defaultTheta
  } = options;
  const pipeline = await getPipelineFast("rope", "default");
  const uniformBuffer = createUniformBufferWithView(
    "rope_uniforms",
    ropeDefaults.uniformSize,
    (view) => {
      view.setUint32(0, seqLen, true);
      view.setUint32(4, numHeads, true);
      view.setUint32(8, headDim, true);
      view.setUint32(12, options.startPos || 0, true);
      view.setFloat32(16, ropeTheta, true);
      view.setFloat32(20, 1, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "rope_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: freqsCos } },
      { binding: 3, resource: { buffer: freqsSin } }
    ]
  });
  if (headDim % 2 !== 0) {
    throw new Error(`RoPE headDim must be even, got ${headDim}`);
  }
  const halfDim = headDim / 2;
  const workgroups = Math.ceil(seqLen * numHeads * halfDim / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "rope");
  uniformBuffer.destroy();
  return createTensor(input.buffer, input.dtype, [...input.shape], "rope_output");
}
async function recordRoPE(recorder, input, freqsCos, freqsSin, seqLen, options = {}) {
  const device2 = recorder.device;
  const ropeDefaults = getRopeDefaults();
  const {
    numHeads = 1,
    headDim = 64,
    ropeTheta = ropeDefaults.defaultTheta
  } = options;
  const pipeline = await getPipelineFast("rope", "default");
  const uniformBuffer = createUniformBufferWithView(
    "rope_uniforms",
    ropeDefaults.uniformSize,
    (view) => {
      view.setUint32(0, seqLen, true);
      view.setUint32(4, numHeads, true);
      view.setUint32(8, headDim, true);
      view.setUint32(12, options.startPos || 0, true);
      view.setFloat32(16, ropeTheta, true);
      view.setFloat32(20, 1, true);
    },
    recorder
  );
  const bindGroup = device2.createBindGroup({
    label: "rope_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: freqsCos } },
      { binding: 3, resource: { buffer: freqsSin } }
    ]
  });
  if (headDim % 2 !== 0) {
    throw new Error(`RoPE headDim must be even, got ${headDim}`);
  }
  const halfDim = headDim / 2;
  const workgroups = Math.ceil(seqLen * numHeads * halfDim / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "rope");
  return createTensor(input.buffer, input.dtype, [...input.shape], "rope_output");
}

// src/gpu/kernels/silu.ts
init_device();
init_buffer_pool();
init_tensor();
init_constants();
init_dispatch();
init_utils();
var SILU_VARIANTS = {
  "default/false": "default",
  "default/true": "default_f16",
  "vec4/false": "vec4",
  "vec4/true": "vec4_f16",
  "gate/false": "gate",
  "gate/true": "gate_f16",
  "gate_rowsplit/false": "gate_rowsplit",
  "gate_rowsplit/true": "gate_rowsplit_f16",
  "geglu_rowsplit/false": "geglu_rowsplit",
  "geglu_rowsplit/true": "geglu_rowsplit_f16"
};
function selectSiLUVariant(base, isF16) {
  const key = `${base}/${isF16}`;
  return SILU_VARIANTS[key] ?? base;
}
function canUseF162(input) {
  return input.dtype === "f16";
}
function createSiLUBindGroupEntries(uniformBuffer, input, output, gate) {
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: input.buffer } },
    { binding: 2, resource: { buffer: output } }
  ];
  if (gate) {
    entries.push({ binding: 3, resource: { buffer: gate.buffer } });
  }
  return entries;
}
async function runSiLU(input, options = {}) {
  const device2 = getDevice();
  const { size, gate = null, outputBuffer = null, useVec4 = false } = options;
  const isF16 = canUseF162(input);
  const bytesPerElement = dtypeBytes(input.dtype);
  const baseVariant = gate ? "gate" : useVec4 ? "vec4" : "default";
  const variant = selectSiLUVariant(baseVariant, isF16);
  const pipeline = await getPipelineFast("silu", variant);
  const inferredSize = size || input.buffer.size / bytesPerElement;
  const outputSize = inferredSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "silu_output");
  const uniformBuffer = createUniformBufferWithView(
    "silu_uniforms",
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
    },
    null,
    device2
  );
  const entries = createSiLUBindGroupEntries(uniformBuffer, input, output, gate);
  const bindGroup = device2.createBindGroup({
    label: "silu_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries
  });
  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "silu");
  uniformBuffer.destroy();
  return createTensor(output, input.dtype, [inferredSize], "silu_output");
}
async function runSwiGLURowsplitBias(input, bias, numTokens, dim, options = {}) {
  const device2 = getDevice();
  const { outputBuffer = null, biasOffset = 0 } = options;
  const pipeline = await getPipelineFast("swiglu", "rowsplit_bias");
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = numTokens * dim * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "swiglu_output");
  const uniformBuffer = createUniformBufferWithView(
    "swiglu_uniforms",
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, dim, true);
      view.setUint32(8, biasOffset, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "swiglu_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: bias.buffer } },
      { binding: 3, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(numTokens * dim / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "swiglu");
  uniformBuffer.destroy();
  return createTensor(output, input.dtype, [numTokens, dim], "swiglu_output");
}
async function runSiLURowSplit(input, options) {
  const device2 = getDevice();
  const { numTokens, dim, activation = "silu", outputBuffer = null } = options;
  const isF16 = canUseF162(input);
  const bytesPerElement = dtypeBytes(input.dtype);
  let variant = activation === "gelu" ? "geglu_rowsplit" : "gate_rowsplit";
  if (isF16) {
    variant = variant + "_f16";
  }
  const pipeline = await getPipelineFast("silu", variant);
  const outputSize = numTokens * dim * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "silu_rowsplit_output");
  const uniformBuffer = createUniformBufferWithView(
    "silu_rowsplit_uniforms",
    16,
    (view) => {
      view.setUint32(0, numTokens * dim, true);
      view.setUint32(4, dim, true);
      view.setUint32(8, 0, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "silu_rowsplit_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(numTokens * dim / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "silu_rowsplit");
  uniformBuffer.destroy();
  return createTensor(output, input.dtype, [numTokens, dim], "silu_rowsplit_output");
}
async function recordSiLURowSplit(recorder, input, options) {
  const device2 = recorder.device;
  const { numTokens, dim, activation = "silu", outputBuffer = null } = options;
  const isF16 = canUseF162(input);
  const bytesPerElement = dtypeBytes(input.dtype);
  let variant = activation === "gelu" ? "geglu_rowsplit" : "gate_rowsplit";
  if (isF16) {
    variant = variant + "_f16";
  }
  const pipeline = await getPipelineFast("silu", variant);
  const outputSize = numTokens * dim * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "silu_rowsplit_output");
  const uniformBuffer = createUniformBufferWithView(
    "silu_rowsplit_uniforms",
    16,
    (view) => {
      view.setUint32(0, numTokens * dim, true);
      view.setUint32(4, dim, true);
    },
    recorder
  );
  const bindGroup = device2.createBindGroup({
    label: "silu_rowsplit_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(numTokens * dim / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "silu_rowsplit");
  return createTensor(output, input.dtype, [numTokens, dim], "silu_rowsplit_output");
}
async function recordSiLU(recorder, input, options = {}) {
  const device2 = recorder.device;
  const { size, gate = null, outputBuffer = null } = options;
  const isF16 = canUseF162(input);
  const bytesPerElement = dtypeBytes(input.dtype);
  const baseVariant = gate ? "gate" : "default";
  const variant = selectSiLUVariant(baseVariant, isF16);
  const pipeline = await getPipelineFast("silu", variant);
  const inferredSize = size || input.buffer.size / bytesPerElement;
  const outputSize = inferredSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "silu_output");
  const uniformBuffer = createUniformBufferWithView(
    "silu_uniforms",
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
    },
    recorder
  );
  const entries = createSiLUBindGroupEntries(uniformBuffer, input, output, gate);
  const bindGroup = device2.createBindGroup({
    label: "silu_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries
  });
  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "silu");
  return createTensor(output, input.dtype, [inferredSize], "silu_output");
}

// src/gpu/kernels/gelu.ts
init_device();
init_buffer_pool();
init_tensor();
init_constants();
init_dispatch();
init_utils();
function canUseF163(input) {
  return input.dtype === "f16";
}
async function runGeLU(input, options = {}) {
  const device2 = getDevice();
  const { size, gate = null, outputBuffer = null } = options;
  const isF16 = canUseF163(input);
  const bytesPerElement = dtypeBytes(input.dtype);
  let variant;
  if (gate) {
    variant = isF16 ? "geglu_rowsplit_f16" : "geglu";
  } else {
    variant = "gelu";
  }
  const pipeline = await createPipeline("silu", variant);
  const inferredSize = size || input.buffer.size / bytesPerElement;
  const outputSize = inferredSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "gelu_output");
  const uniformBuffer = createUniformBufferWithView(
    "gelu_uniforms",
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
    },
    null,
    device2
  );
  const gateBuffer = gate ? gate.buffer : input.buffer;
  const bindGroup = device2.createBindGroup({
    label: "gelu_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: gateBuffer } }
    ]
  });
  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "gelu");
  uniformBuffer.destroy();
  return createTensor(output, input.dtype, [inferredSize], "gelu_output");
}
async function recordGeLU(recorder, input, options = {}) {
  const device2 = recorder.device;
  const { size, gate = null, outputBuffer = null } = options;
  const isF16 = canUseF163(input);
  const bytesPerElement = dtypeBytes(input.dtype);
  let variant;
  if (gate) {
    variant = isF16 ? "geglu_rowsplit_f16" : "geglu";
  } else {
    variant = "gelu";
  }
  const pipeline = await createPipeline("silu", variant);
  const inferredSize = size || input.buffer.size / bytesPerElement;
  const outputSize = inferredSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "gelu_output");
  const uniformBuffer = createUniformBufferWithView(
    "gelu_uniforms",
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
    },
    recorder
  );
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: input.buffer } },
    { binding: 2, resource: { buffer: output } }
  ];
  if (gate) {
    entries.push({ binding: 3, resource: { buffer: gate.buffer } });
  }
  const bindGroup = device2.createBindGroup({
    label: "gelu_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries
  });
  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "gelu");
  return createTensor(output, input.dtype, [inferredSize], "gelu_output");
}

// src/gpu/kernels/scale.ts
init_device();
init_buffer_pool();
init_tensor();
init_constants();
init_dispatch();
init_utils();
async function runScale(input, scale, options = {}) {
  const device2 = getDevice();
  const { count, outputBuffer = null, inplace = false } = options;
  const bytesPerElement = dtypeBytes(input.dtype);
  const inferredCount = count ?? Math.floor(input.buffer.size / bytesPerElement);
  const variant = inplace ? "inplace" : "default";
  const pipeline = await createPipeline("scale", variant);
  const outputSize = inferredCount * bytesPerElement;
  const outputBuf = inplace ? input.buffer : outputBuffer || acquireBuffer(outputSize, void 0, "scale_output");
  const uniformBuffer = createUniformBufferWithView(
    "scale_uniforms",
    16,
    (view) => {
      view.setUint32(0, inferredCount, true);
      view.setFloat32(4, scale, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "scale_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: outputBuf } }
    ]
  });
  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "scale");
  uniformBuffer.destroy();
  return createTensor(outputBuf, input.dtype, [...input.shape], "scale_output");
}
async function recordScale(recorder, input, scale, options = {}) {
  const device2 = recorder.device;
  const { count, outputBuffer = null, inplace = false } = options;
  const bytesPerElement = dtypeBytes(input.dtype);
  const inferredCount = count ?? Math.floor(input.buffer.size / bytesPerElement);
  const variant = inplace ? "inplace" : "default";
  const pipeline = await createPipeline("scale", variant);
  const outputSize = inferredCount * bytesPerElement;
  const outputBuf = inplace ? input.buffer : outputBuffer || acquireBuffer(outputSize, void 0, "scale_output");
  const uniformBuffer = createUniformBufferWithView(
    "scale_uniforms",
    16,
    (view) => {
      view.setUint32(0, inferredCount, true);
      view.setFloat32(4, scale, true);
    },
    recorder
  );
  const bindGroup = device2.createBindGroup({
    label: "scale_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: outputBuf } }
    ]
  });
  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "scale");
  return createTensor(outputBuf, input.dtype, [...input.shape], "scale_output");
}

// src/gpu/kernels/gather.ts
init_device();
init_buffer_pool();
init_constants();
init_dispatch();
init_utils();
init_debug();
init_tensor();
init_schema();
var GATHER_VARIANTS = {
  "false/false/false": "default",
  "false/false/true": "vec4",
  "true/false/false": "f16",
  "true/false/true": "f16_vec4",
  "false/true/false": "f16_out",
  "false/true/true": "vec4_f16_out",
  "true/true/false": "f16_f16_out",
  "true/true/true": "f16_vec4_f16_out"
};
function selectGatherVariant(useF16Input, useF16Output, useVec4) {
  const key = `${useF16Input}/${useF16Output}/${useVec4}`;
  const variant = GATHER_VARIANTS[key];
  if (!variant) {
    throw new Error(`Unknown gather variant combination: ${key}`);
  }
  return variant;
}
function getOutputBinding(variant, useF16Output) {
  if (!useF16Output) {
    return 3;
  }
  const config2 = getKernelConfig("gather", variant);
  return config2.variantMetadata?.outputBinding ?? 4;
}
async function runGather(indices, embeddings, numTokens, hiddenSize, vocabSize, options = {}) {
  const device2 = getDevice();
  const {
    useVec4 = true,
    outputBuffer = null,
    embeddingDtype,
    outputDtype = "f32",
    transpose = false,
    indirectBuffer = null,
    indirectOffset = 0
  } = options;
  const caps = getKernelCapabilities();
  const detectedDtype = embeddingDtype || "f32";
  const useF16Input = detectedDtype === "f16" && caps.hasF16;
  const useF16Output = outputDtype === "f16" && caps.hasF16;
  trace.embed(`Gather: numTokens=${numTokens}, hiddenSize=${hiddenSize}, vocabSize=${vocabSize}, transpose=${transpose}, detectedDtype=${detectedDtype}, useF16Input=${useF16Input}, useF16Output=${useF16Output}`);
  const variant = selectGatherVariant(useF16Input, useF16Output, useVec4);
  trace.embed(`Gather variant: ${variant}`);
  const pipeline = await getPipelineFast("gather", variant);
  const outputDtypeKey = useF16Output ? "f16" : "f32";
  const bytesPerElement = DTYPE_SIZES[outputDtypeKey];
  const outputSize = numTokens * hiddenSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "gather_output");
  const uniformBuffer = createUniformBufferWithView(
    "gather_uniforms",
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, vocabSize, true);
      view.setUint32(12, transpose ? 1 : 0, true);
    },
    null,
    device2
  );
  const outputBinding = getOutputBinding(variant, useF16Output);
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: indices } },
    { binding: 2, resource: { buffer: embeddings } },
    { binding: outputBinding, resource: { buffer: output } }
  ];
  const bindGroup = device2.createBindGroup({
    label: "gather_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries
  });
  const workgroups = useVec4 ? Math.ceil(numTokens * hiddenSize / VEC4_ELEMENTS_PER_WG) : Math.ceil(numTokens * hiddenSize / WORKGROUP_SIZES.DEFAULT);
  if (indirectBuffer) {
    dispatchIndirect(device2, pipeline, bindGroup, indirectBuffer, indirectOffset, "gather");
  } else {
    dispatch(device2, pipeline, bindGroup, workgroups, "gather");
  }
  uniformBuffer.destroy();
  const actualDtype = useF16Output ? "f16" : "f32";
  return createTensor(output, actualDtype, [numTokens, hiddenSize], "gather_output");
}
async function recordGather(recorder, indices, embeddings, numTokens, hiddenSize, vocabSize, options = {}) {
  const device2 = recorder.device;
  const {
    useVec4 = true,
    outputBuffer = null,
    embeddingDtype,
    outputDtype = "f32",
    transpose = false,
    indirectBuffer = null,
    indirectOffset = 0
  } = options;
  const caps = getKernelCapabilities();
  const detectedDtype = embeddingDtype || "f32";
  const useF16Input = detectedDtype === "f16" && caps.hasF16;
  const useF16Output = outputDtype === "f16" && caps.hasF16;
  const variant = selectGatherVariant(useF16Input, useF16Output, useVec4);
  trace.embed(`Gather variant: ${variant}`);
  const pipeline = await getPipelineFast("gather", variant);
  const outputDtypeKey = useF16Output ? "f16" : "f32";
  const bytesPerElement = DTYPE_SIZES[outputDtypeKey];
  const outputSize = numTokens * hiddenSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "gather_output");
  const uniformBuffer = createUniformBufferWithView(
    "gather_uniforms",
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, vocabSize, true);
      view.setUint32(12, transpose ? 1 : 0, true);
    },
    recorder
  );
  const outputBinding = getOutputBinding(variant, useF16Output);
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: indices } },
    { binding: 2, resource: { buffer: embeddings } },
    { binding: outputBinding, resource: { buffer: output } }
  ];
  const bindGroup = device2.createBindGroup({
    label: "gather_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries
  });
  const workgroups = useVec4 ? Math.ceil(numTokens * hiddenSize / VEC4_ELEMENTS_PER_WG) : Math.ceil(numTokens * hiddenSize / WORKGROUP_SIZES.DEFAULT);
  if (indirectBuffer) {
    recordDispatchIndirect(recorder, pipeline, bindGroup, indirectBuffer, indirectOffset, "gather");
  } else {
    recordDispatch(recorder, pipeline, bindGroup, workgroups, "gather");
  }
  const actualDtype = useF16Output ? "f16" : "f32";
  return createTensor(output, actualDtype, [numTokens, hiddenSize], "gather_output");
}

// src/gpu/kernels/residual.ts
init_device();
init_buffer_pool();
init_tensor();
init_constants();
init_dispatch();
init_utils();

// src/gpu/kernels/cast.ts
init_device();
init_buffer_pool();
init_tensor();
init_dispatch();
init_utils();
init_constants();
init_debug();
init_schema();
function calculate2DDispatch(workgroups) {
  const maxWorkgroupsPerDim = GPU_LIMITS.MAX_WORKGROUPS;
  return workgroups <= maxWorkgroupsPerDim ? [workgroups, 1, 1] : [maxWorkgroupsPerDim, Math.ceil(workgroups / maxWorkgroupsPerDim), 1];
}
function lcm(a, b) {
  const gcd = (x, y) => {
    let a0 = x;
    let b0 = y;
    while (b0 !== 0) {
      const t = b0;
      b0 = a0 % b0;
      a0 = t;
    }
    return a0;
  };
  return a / gcd(a, b) * b;
}
async function castF32ToF16(input, options = {}) {
  const device2 = getDevice();
  const { outputBuffer = null } = options;
  const numElements = input.shape.reduce((a, b) => a * b, 1);
  const pipeline = await createPipeline("cast", "f32_to_f16");
  const outputSize = numElements * DTYPE_SIZES.f16;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "cast_f32_to_f16_output");
  const uniformBuffer = createUniformBufferWithView(
    "cast_f32_to_f16_uniforms",
    16,
    (view) => {
      view.setUint32(0, numElements, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "cast_f32_to_f16_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(numElements / WORKGROUP_SIZES.DEFAULT);
  const dispatchSize = calculate2DDispatch(workgroups);
  dispatch(device2, pipeline, bindGroup, dispatchSize, "cast_f32_to_f16");
  await device2.queue.onSubmittedWorkDone();
  uniformBuffer.destroy();
  return createTensor(output, "f16", [...input.shape], input.label ? `${input.label}_f16` : "cast_f32_to_f16_output");
}
async function castF16ToF32(input, options = {}) {
  const device2 = getDevice();
  const { outputBuffer = null } = options;
  const numElements = input.shape.reduce((a, b) => a * b, 1);
  const pipeline = await createPipeline("cast", "f16_to_f32");
  const outputSize = numElements * DTYPE_SIZES.f32;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "cast_f16_to_f32_output");
  const uniformBuffer = createUniformBufferWithView(
    "cast_f16_to_f32_uniforms",
    16,
    (view) => {
      view.setUint32(0, numElements, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "cast_f16_to_f32_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(numElements / WORKGROUP_SIZES.DEFAULT);
  const dispatchSize = calculate2DDispatch(workgroups);
  dispatch(device2, pipeline, bindGroup, dispatchSize, "cast_f16_to_f32");
  await device2.queue.onSubmittedWorkDone();
  uniformBuffer.destroy();
  return createTensor(output, "f32", [...input.shape], input.label ? `${input.label}_f32` : "cast_f16_to_f32_output");
}
async function recordCastF32ToF16(recorder, input, options = {}) {
  const device2 = recorder.device;
  const { outputBuffer = null } = options;
  const numElements = input.shape.reduce((a, b) => a * b, 1);
  const pipeline = await createPipeline("cast", "f32_to_f16");
  const outputSize = numElements * DTYPE_SIZES.f16;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "cast_f32_to_f16_output");
  const uniformBuffer = createUniformBufferWithView(
    "cast_f32_to_f16_uniforms",
    16,
    (view) => {
      view.setUint32(0, numElements, true);
    },
    recorder
  );
  const bindGroup = device2.createBindGroup({
    label: "cast_f32_to_f16_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(numElements / WORKGROUP_SIZES.DEFAULT);
  const dispatchSize = calculate2DDispatch(workgroups);
  recordDispatch(recorder, pipeline, bindGroup, dispatchSize, "cast_f32_to_f16");
  return createTensor(output, "f16", [...input.shape], input.label ? `${input.label}_f16` : "cast_f32_to_f16_output");
}
async function recordCastF16ToF32(recorder, input, options = {}) {
  const device2 = recorder.device;
  const { outputBuffer = null } = options;
  const numElements = input.shape.reduce((a, b) => a * b, 1);
  const pipeline = await createPipeline("cast", "f16_to_f32");
  const outputSize = numElements * DTYPE_SIZES.f32;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "cast_f16_to_f32_output");
  const uniformBuffer = createUniformBufferWithView(
    "cast_f16_to_f32_uniforms",
    16,
    (view) => {
      view.setUint32(0, numElements, true);
    },
    recorder
  );
  const bindGroup = device2.createBindGroup({
    label: "cast_f16_to_f32_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(numElements / WORKGROUP_SIZES.DEFAULT);
  const dispatchSize = calculate2DDispatch(workgroups);
  recordDispatch(recorder, pipeline, bindGroup, dispatchSize, "cast_f16_to_f32");
  return createTensor(output, "f32", [...input.shape], input.label ? `${input.label}_f32` : "cast_f16_to_f32_output");
}
async function runBF16ToF32(input, shape, name = "bf16_to_f32_output") {
  const numElements = shape.reduce((a, b) => a * b, 1);
  trace.kernels(`BF16ToF32: Entry numElements=${numElements}, name=${name}, inputSize=${input.size}`);
  const device2 = getDevice();
  const limits = device2.limits;
  const maxBufferSize = limits.maxBufferSize;
  const maxBindingSize = limits.maxStorageBufferBindingSize;
  const outputSize = numElements * DTYPE_SIZES.f32;
  trace.kernels(`BF16ToF32: outputSize=${outputSize}, maxBufferSize=${maxBufferSize}, maxBindingSize=${maxBindingSize}`);
  if (outputSize > maxBufferSize) {
    throw new Error(
      `BF16\u2192F32 output (${outputSize} bytes) exceeds device maxBufferSize (${maxBufferSize}). This often happens for large-vocab models when converting embeddings/LM head. Enable F16 and use BF16\u2192F16 weights, or run on a device with a higher maxBufferSize.`
    );
  }
  if (outputSize > maxBindingSize) {
    return runBF16ToF32Chunked(input, shape, name, maxBindingSize);
  }
  const pipeline = await createPipeline("bf16_to_f32", "default");
  trace.kernels("BF16ToF32: Pipeline created");
  const output = acquireBuffer(outputSize, void 0, name);
  trace.kernels(`BF16ToF32: Output buffer acquired, size=${output.size}`);
  const uniformBuffer = createUniformBufferWithView(
    "bf16_to_f32_uniforms",
    16,
    (view) => {
      view.setUint32(0, numElements, true);
    },
    null,
    device2
  );
  trace.kernels(`BF16ToF32: Uniform numElements=${numElements}`);
  const bindGroup = device2.createBindGroup({
    label: "bf16_to_f32_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  trace.kernels("BF16ToF32: BindGroup created");
  const numPairs = Math.ceil(numElements / 2);
  const workgroups = Math.ceil(numPairs / WORKGROUP_SIZES.DEFAULT);
  const dispatchSize = calculate2DDispatch(workgroups);
  trace.kernels(`BF16ToF32: Dispatching ${dispatchSize[0]}x${dispatchSize[1]} workgroups for ${numPairs} pairs (${numElements} elements)`);
  dispatch(device2, pipeline, bindGroup, dispatchSize, "bf16_to_f32");
  await device2.queue.onSubmittedWorkDone();
  trace.kernels("BF16ToF32: GPU work completed");
  uniformBuffer.destroy();
  return createTensor(output, "f32", [...shape], name);
}
async function runBF16ToF16(input, shape, name = "bf16_to_f16_output") {
  const numElements = shape.reduce((a, b) => a * b, 1);
  const device2 = getDevice();
  const pipeline = await createPipeline("bf16_to_f16", "default");
  const limits = device2.limits;
  const maxBufferSize = limits.maxBufferSize;
  const maxBindingSize = limits.maxStorageBufferBindingSize;
  const outputSize = numElements * DTYPE_SIZES.f16;
  if (outputSize > maxBufferSize) {
    throw new Error(
      `BF16\u2192F16 output (${outputSize} bytes) exceeds device maxBufferSize (${maxBufferSize}).`
    );
  }
  if (outputSize > maxBindingSize) {
    throw new Error(
      `BF16\u2192F16 output (${outputSize} bytes) exceeds device maxStorageBufferBindingSize (${maxBindingSize}).`
    );
  }
  const output = acquireBuffer(outputSize, void 0, name);
  const uniformBuffer = createUniformBufferWithView(
    "bf16_to_f16_uniforms",
    16,
    (view) => {
      view.setUint32(0, numElements, true);
      view.setUint32(4, 0, true);
      view.setUint32(8, 0, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "bf16_to_f16_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const numPairs = Math.ceil(numElements / 2);
  const workgroups = Math.ceil(numPairs / WORKGROUP_SIZES.DEFAULT);
  const dispatchSize = calculate2DDispatch(workgroups);
  dispatch(device2, pipeline, bindGroup, dispatchSize, "bf16_to_f16");
  await device2.queue.onSubmittedWorkDone();
  uniformBuffer.destroy();
  return createTensor(output, "f16", [...shape], name);
}
async function runBF16ToF32Chunked(input, shape, name, maxBindingSize) {
  const numElements = shape.reduce((a, b) => a * b, 1);
  const device2 = getDevice();
  const pipeline = await createPipeline("bf16_to_f32", "default");
  const alignmentBytes = device2.limits.minStorageBufferOffsetAlignment;
  const inElemAlign = Math.max(1, Math.floor(alignmentBytes / DTYPE_SIZES.bf16));
  const outElemAlign = Math.max(1, Math.floor(alignmentBytes / DTYPE_SIZES.f32));
  const elemAlign = lcm(inElemAlign, outElemAlign);
  let maxElementsPerChunk = Math.floor(maxBindingSize / DTYPE_SIZES.f32);
  maxElementsPerChunk -= maxElementsPerChunk % elemAlign;
  if (maxElementsPerChunk <= 0) {
    throw new Error(`BF16\u2192F32 chunk size underflow (maxBindingSize=${maxBindingSize}, alignment=${alignmentBytes})`);
  }
  const numChunks = Math.ceil(numElements / maxElementsPerChunk);
  const outputSize = numElements * DTYPE_SIZES.f32;
  const output = acquireBuffer(outputSize, void 0, name);
  trace.kernels(`BF16ToF32: Chunking ${numElements} elements in ${numChunks} chunks`);
  for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
    const chunkStart = chunkIdx * maxElementsPerChunk;
    const chunkEnd = Math.min((chunkIdx + 1) * maxElementsPerChunk, numElements);
    const chunkSize = chunkEnd - chunkStart;
    const uniformBuffer = createUniformBufferWithView(
      `bf16_to_f32_chunk${chunkIdx}_uniforms`,
      16,
      (view) => {
        view.setUint32(0, chunkSize, true);
        view.setUint32(4, 0, true);
        view.setUint32(8, 0, true);
      },
      null,
      device2
    );
    const inputOffsetBytes = chunkStart * DTYPE_SIZES.bf16;
    const outputOffsetBytes = chunkStart * DTYPE_SIZES.f32;
    const inputPairs = Math.ceil(chunkSize / 2);
    const inputSizeBytes = inputPairs * DTYPE_SIZES.f32;
    const outputSizeBytes = chunkSize * DTYPE_SIZES.f32;
    const bindGroup = device2.createBindGroup({
      label: `bf16_to_f32_chunk${chunkIdx}_bind_group`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: input, offset: inputOffsetBytes, size: inputSizeBytes } },
        { binding: 2, resource: { buffer: output, offset: outputOffsetBytes, size: outputSizeBytes } }
      ]
    });
    const numPairs = Math.ceil(chunkSize / 2);
    const workgroups = Math.ceil(numPairs / WORKGROUP_SIZES.DEFAULT);
    const dispatchSize = calculate2DDispatch(workgroups);
    dispatch(device2, pipeline, bindGroup, dispatchSize, `bf16_to_f32_chunk${chunkIdx}`);
    uniformBuffer.destroy();
  }
  return createTensor(output, "f32", [...shape], name);
}

// src/gpu/kernels/residual.ts
async function alignResidualInputs(a, b, recorder) {
  if (a.dtype === b.dtype) {
    return { a, b, temps: [] };
  }
  if (a.dtype === "f16" && b.dtype === "f32") {
    const casted = recorder ? await recordCastF16ToF32(recorder, a) : await castF16ToF32(a);
    return { a: casted, b, temps: [casted.buffer] };
  }
  if (a.dtype === "f32" && b.dtype === "f16") {
    const casted = recorder ? await recordCastF16ToF32(recorder, b) : await castF16ToF32(b);
    return { a, b: casted, temps: [casted.buffer] };
  }
  return { a, b, temps: [] };
}
async function alignBiasTensor(data, bias, recorder) {
  if (data.dtype === bias.dtype) {
    return { bias, temps: [] };
  }
  if (data.dtype === "f16" && bias.dtype === "f32") {
    const casted = recorder ? await recordCastF32ToF16(recorder, bias) : await castF32ToF16(bias);
    return { bias: casted, temps: [casted.buffer] };
  }
  if (data.dtype === "f32" && bias.dtype === "f16") {
    const casted = recorder ? await recordCastF16ToF32(recorder, bias) : await castF16ToF32(bias);
    return { bias: casted, temps: [casted.buffer] };
  }
  return { bias, temps: [] };
}
async function runResidualAdd(a, b, size, options = {}) {
  const device2 = getDevice();
  const { useVec4 = true, outputBuffer = null } = options;
  const { a: aAligned, b: bAligned, temps } = await alignResidualInputs(a, b);
  const outputDtype = inferOutputDtype(aAligned, bAligned);
  const bytesPerElement = dtypeBytes(outputDtype);
  const variant = useVec4 ? outputDtype === "f16" ? "vec4_f16" : "vec4" : outputDtype === "f16" ? "default_f16" : "default";
  const pipeline = await getPipelineFast("residual", variant);
  const outputSize = size * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "residual_output");
  const uniformBuffer = createUniformBufferWithView(
    "residual_uniforms",
    16,
    (view) => {
      view.setUint32(0, size, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "residual_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: aAligned.buffer } },
      { binding: 2, resource: { buffer: bAligned.buffer } },
      { binding: 3, resource: { buffer: output } }
    ]
  });
  const workgroups = useVec4 ? Math.ceil(size / VEC4_ELEMENTS_PER_WG) : Math.ceil(size / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "residual");
  uniformBuffer.destroy();
  for (const temp of temps) {
    releaseBuffer(temp);
  }
  return createTensor(output, outputDtype, [size], "residual_output");
}
async function runBiasAdd(data, bias, numTokens, dim, options = {}) {
  const device2 = getDevice();
  const { dataOffset = 0, biasOffset = 0 } = options;
  const { bias: biasAligned, temps } = await alignBiasTensor(data, bias);
  const variant = data.dtype === "f16" && biasAligned.dtype === "f16" ? "f16" : "default";
  const pipeline = await getPipelineFast("bias_add", variant);
  const uniformBuffer = createUniformBufferWithView(
    "bias_add_uniforms",
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, dim, true);
      view.setUint32(8, dataOffset, true);
      view.setUint32(12, biasOffset, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "bias_add_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: data.buffer } },
      { binding: 2, resource: { buffer: biasAligned.buffer } }
    ]
  });
  const workgroups = Math.ceil(numTokens * dim / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "bias_add");
  uniformBuffer.destroy();
  for (const temp of temps) {
    releaseBuffer(temp);
  }
  return createTensor(data.buffer, data.dtype, [numTokens, dim], "bias_add_output");
}
async function recordResidualAdd(recorder, a, b, size, options = {}) {
  const device2 = recorder.device;
  const { outputBuffer = null, useVec4 = true } = options;
  const { a: aAligned, b: bAligned, temps } = await alignResidualInputs(a, b, recorder);
  const outputDtype = inferOutputDtype(aAligned, bAligned);
  const bytesPerElement = dtypeBytes(outputDtype);
  const variant = useVec4 ? outputDtype === "f16" ? "vec4_f16" : "vec4" : outputDtype === "f16" ? "default_f16" : "default";
  const pipeline = await getPipelineFast("residual", variant);
  const outputSize = size * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "residual_output");
  const uniformBuffer = createUniformBufferWithView(
    "residual_uniforms",
    16,
    (view) => {
      view.setUint32(0, size, true);
    },
    recorder
  );
  const bindGroup = device2.createBindGroup({
    label: "residual_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: aAligned.buffer } },
      { binding: 2, resource: { buffer: bAligned.buffer } },
      { binding: 3, resource: { buffer: output } }
    ]
  });
  const workgroups = useVec4 ? Math.ceil(size / VEC4_ELEMENTS_PER_WG) : Math.ceil(size / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "residual");
  for (const temp of temps) {
    recorder.trackTemporaryBuffer(temp);
  }
  return createTensor(output, outputDtype, [size], "residual_output");
}
async function recordBiasAdd(recorder, data, bias, numTokens, dim, options = {}) {
  const device2 = recorder.device;
  const { dataOffset = 0, biasOffset = 0 } = options;
  const { bias: biasAligned, temps } = await alignBiasTensor(data, bias, recorder);
  const variant = data.dtype === "f16" && biasAligned.dtype === "f16" ? "f16" : "default";
  const pipeline = await getPipelineFast("bias_add", variant);
  const uniformBuffer = createUniformBufferWithView(
    "bias_add_uniforms",
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, dim, true);
      view.setUint32(8, dataOffset, true);
      view.setUint32(12, biasOffset, true);
    },
    recorder
  );
  const bindGroup = device2.createBindGroup({
    label: "bias_add_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: data.buffer } },
      { binding: 2, resource: { buffer: biasAligned.buffer } }
    ]
  });
  const workgroups = Math.ceil(numTokens * dim / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "bias_add");
  for (const temp of temps) {
    recorder.trackTemporaryBuffer(temp);
  }
  return createTensor(data.buffer, data.dtype, [numTokens, dim], "bias_add_output");
}

// src/gpu/kernels/moe.ts
init_device();
init_buffer_pool();
init_tensor();
init_constants();
init_dispatch();
init_utils();
async function runTopK(probs, numTokens, numExperts, topK, options = {}) {
  const device2 = getDevice();
  const { normalize = true } = options;
  const pipeline = await createPipeline("topk", "default");
  const indicesSize = numTokens * topK * 4;
  const weightsSize = numTokens * topK * 4;
  const indices = acquireBuffer(indicesSize, void 0, "topk_indices");
  const weights = acquireBuffer(weightsSize, void 0, "topk_weights");
  const uniformBuffer = createUniformBufferWithView(
    "topk_uniforms",
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, numExperts, true);
      view.setUint32(8, topK, true);
      view.setUint32(12, normalize ? 1 : 0, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "topk_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: probs } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } }
    ]
  });
  dispatch(device2, pipeline, bindGroup, numTokens, "topk");
  uniformBuffer.destroy();
  return { indices, weights };
}
var moeGatherBindGroupLayout = null;
function getMoEGatherBindGroupLayout(device2) {
  if (moeGatherBindGroupLayout)
    return moeGatherBindGroupLayout;
  moeGatherBindGroupLayout = device2.createBindGroupLayout({
    label: "moe_gather_explicit_layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
    ]
  });
  return moeGatherBindGroupLayout;
}
async function runMoEGather(hiddenStates, expertIndices, numTokens, hiddenSize, numExperts, topK, options = {}) {
  const device2 = getDevice();
  const { maxTokensPerExpert = numTokens } = options;
  const explicitLayout = getMoEGatherBindGroupLayout(device2);
  const countPipeline = await createPipeline("moe_gather", "count", explicitLayout);
  const gatherPipeline = await createPipeline("moe_gather", "gather", explicitLayout);
  const bytesPerElement = hiddenStates.dtype === "f16" ? 2 : 4;
  const gatheredSize = numExperts * maxTokensPerExpert * hiddenSize * bytesPerElement;
  const tokenCountsSize = numExperts * 4;
  const tokenMapSize = numExperts * maxTokensPerExpert * 2 * 4;
  const gatheredBuffer = acquireBuffer(gatheredSize, void 0, "moe_gathered");
  const tokenCounts = acquireBuffer(tokenCountsSize, void 0, "moe_token_counts");
  const tokenMap = acquireBuffer(tokenMapSize, void 0, "moe_token_map");
  const uniformBuffer = createUniformBufferWithView(
    "moe_gather_uniforms",
    32,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, numExperts, true);
      view.setUint32(12, topK, true);
      view.setUint32(16, maxTokensPerExpert, true);
      view.setUint32(20, 0, true);
      view.setUint32(24, 0, true);
      view.setUint32(28, 0, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "moe_gather_bind_group",
    layout: explicitLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: hiddenStates.buffer } },
      { binding: 2, resource: { buffer: expertIndices } },
      { binding: 3, resource: { buffer: gatheredBuffer } },
      { binding: 4, resource: { buffer: tokenCounts } },
      { binding: 5, resource: { buffer: tokenMap } }
    ]
  });
  const encoder = device2.createCommandEncoder({ label: "moe_gather_encoder" });
  encoder.clearBuffer(tokenCounts);
  const countPass = encoder.beginComputePass({ label: "moe_gather_count_pass" });
  countPass.setPipeline(countPipeline);
  countPass.setBindGroup(0, bindGroup);
  const countWorkgroups = Math.ceil(numTokens * topK / WORKGROUP_SIZES.DEFAULT);
  countPass.dispatchWorkgroups(countWorkgroups);
  countPass.end();
  const gatherPass = encoder.beginComputePass({ label: "moe_gather_gather_pass" });
  gatherPass.setPipeline(gatherPipeline);
  gatherPass.setBindGroup(0, bindGroup);
  const gatherWorkgroups = Math.ceil(numExperts * maxTokensPerExpert * hiddenSize / WORKGROUP_SIZES.DEFAULT);
  gatherPass.dispatchWorkgroups(gatherWorkgroups);
  gatherPass.end();
  device2.queue.submit([encoder.finish()]);
  uniformBuffer.destroy();
  const gathered = createTensor(
    gatheredBuffer,
    hiddenStates.dtype,
    [numExperts, maxTokensPerExpert, hiddenSize],
    "moe_gathered"
  );
  return { gathered, tokenCounts, tokenMap, maxTokensPerExpert };
}
async function runScatterAdd(expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK, options = {}) {
  const device2 = getDevice();
  const { outputBuffer = null } = options;
  const pipeline = await createPipeline("scatter_add", "default");
  const bytesPerElement = expertOutputs.dtype === "f16" ? 2 : 4;
  const outputSize = numTokens * hiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, void 0, "scatter_add_output");
  const uniformBuffer = createUniformBufferWithView(
    "scatter_add_uniforms",
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, topK, true);
      view.setUint32(12, numExperts, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "scatter_add_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: expertOutputs.buffer } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } },
      { binding: 4, resource: { buffer: outputBuf } }
    ]
  });
  const encoder = device2.createCommandEncoder({ label: "scatter_add_encoder" });
  encoder.clearBuffer(outputBuf);
  const pass = encoder.beginComputePass({ label: "scatter_add_pass" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  const workgroups = Math.ceil(numTokens * hiddenSize / WORKGROUP_SIZES.DEFAULT);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  device2.queue.submit([encoder.finish()]);
  uniformBuffer.destroy();
  return createTensor(outputBuf, expertOutputs.dtype, [numTokens, hiddenSize], "scatter_add_output");
}
async function runScatterAddDynamic(expertOutputs, indices, weights, tokenOffsets, numTokens, hiddenSize, topK, options = {}) {
  const device2 = getDevice();
  const { outputBuffer = null } = options;
  const pipeline = await createPipeline("scatter_add", "dynamic");
  const bytesPerElement = expertOutputs.dtype === "f16" ? 2 : 4;
  const outputSize = numTokens * hiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, void 0, "scatter_add_dynamic_output");
  const uniformBuffer = createUniformBufferWithView(
    "scatter_add_dynamic_uniforms",
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, topK, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "scatter_add_dynamic_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: expertOutputs.buffer } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } },
      { binding: 4, resource: { buffer: tokenOffsets } },
      { binding: 5, resource: { buffer: outputBuf } }
    ]
  });
  const encoder = device2.createCommandEncoder({ label: "scatter_add_dynamic_encoder" });
  encoder.clearBuffer(outputBuf);
  const pass = encoder.beginComputePass({ label: "scatter_add_dynamic_pass" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  const workgroups = Math.ceil(numTokens * topK * hiddenSize / WORKGROUP_SIZES.DEFAULT);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  device2.queue.submit([encoder.finish()]);
  uniformBuffer.destroy();
  return createTensor(outputBuf, expertOutputs.dtype, [numTokens, hiddenSize], "scatter_add_dynamic_output");
}

// src/gpu/kernels/sample.ts
init_device();
init_buffer_pool();
init_constants();
init_utils();
init_perf_guards();
init_runtime();
function getSampleBindGroupLayout(device2) {
  return getOrCreateBindGroupLayout(
    "sample_bind_group_layout",
    [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
    ],
    device2
  );
}
async function createSamplePipeline(device2, entryPoint) {
  return createPipeline("sample", entryPoint, getSampleBindGroupLayout(device2));
}
async function runArgmax(logits, vocabSize, options = {}) {
  if (!allowReadback("sample.runArgmax")) {
    throw new Error("[Sample] GPU readback disabled for argmax");
  }
  const device2 = getDevice();
  if (!device2)
    throw new Error("GPU device not initialized");
  const argmaxPipeline = await createSamplePipeline(device2, "argmax");
  const reducePipeline = await createSamplePipeline(device2, "argmax_reduce");
  const workgroupSize = WORKGROUP_SIZES.DEFAULT;
  const numWorkgroups = Math.min(workgroupSize, Math.ceil(vocabSize / workgroupSize));
  const tempLogits = acquireBuffer(workgroupSize * 4, void 0, "argmax_temp_logits");
  const tempIndices = acquireBuffer(workgroupSize * 4, void 0, "argmax_temp_indices");
  const outputBuffer = acquireBuffer(4, void 0, "argmax_output");
  const padTokenId = options.padTokenId ?? 4294967295;
  const logitSoftcap = options.logitSoftcap ?? 0;
  const uniformBuffer = createUniformBufferWithView(
    "argmax_uniforms",
    32,
    (view) => {
      view.setUint32(0, vocabSize, true);
      view.setUint32(4, 1, true);
      view.setFloat32(8, 1, true);
      view.setFloat32(12, 0, true);
      view.setUint32(16, padTokenId, true);
      view.setFloat32(20, logitSoftcap, true);
    },
    null,
    device2
  );
  const bindGroupLayout = getSampleBindGroupLayout(device2);
  const argmaxBindGroup = device2.createBindGroup({
    label: "argmax_bind_group",
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: tempIndices } },
      { binding: 4, resource: { buffer: tempLogits } }
    ]
  });
  const reduceBindGroup = device2.createBindGroup({
    label: "argmax_reduce_bind_group",
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      // Shader may not use, but layout requires
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: tempIndices } },
      { binding: 4, resource: { buffer: tempLogits } }
    ]
  });
  const encoder = device2.createCommandEncoder({ label: "argmax_encoder" });
  const pass1 = encoder.beginComputePass({ label: "argmax_pass1" });
  pass1.setPipeline(argmaxPipeline);
  pass1.setBindGroup(0, argmaxBindGroup);
  pass1.dispatchWorkgroups(numWorkgroups);
  pass1.end();
  const pass2 = encoder.beginComputePass({ label: "argmax_pass2" });
  pass2.setPipeline(reducePipeline);
  pass2.setBindGroup(0, reduceBindGroup);
  pass2.dispatchWorkgroups(1);
  pass2.end();
  device2.queue.submit([encoder.finish()]);
  const stagingBuffer = device2.createBuffer({
    label: "argmax_staging",
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });
  const copyEncoder = device2.createCommandEncoder({ label: "argmax_copy" });
  copyEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, 4);
  device2.queue.submit([copyEncoder.finish()]);
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const tokenId = new Uint32Array(stagingBuffer.getMappedRange())[0];
  stagingBuffer.unmap();
  stagingBuffer.destroy();
  uniformBuffer.destroy();
  releaseBuffer(tempLogits);
  releaseBuffer(tempIndices);
  releaseBuffer(outputBuffer);
  return tokenId;
}
async function runGPUSample(logits, vocabSize, options = {}) {
  if (!allowReadback("sample.runGPUSample")) {
    throw new Error("[Sample] GPU readback disabled for sampling");
  }
  const samplingDefaults = getRuntimeConfig().inference.sampling;
  const {
    temperature = samplingDefaults.temperature,
    topK = samplingDefaults.topK,
    randomSeed,
    padTokenId,
    logitSoftcap = 0
  } = options;
  const { greedyThreshold } = getRuntimeConfig().inference.sampling;
  if (temperature < greedyThreshold) {
    return runArgmax(logits, vocabSize, { padTokenId, logitSoftcap });
  }
  const device2 = getDevice();
  if (!device2)
    throw new Error("GPU device not initialized");
  const randomValue = randomSeed !== void 0 ? seededRandom(randomSeed) : Math.random();
  const phase1Pipeline = await createSamplePipeline(device2, "find_topk_phase1");
  const phase2Pipeline = await createSamplePipeline(device2, "find_topk_phase2");
  const phase3Pipeline = await createSamplePipeline(device2, "softmax_and_sample");
  const workgroupSize = WORKGROUP_SIZES.DEFAULT;
  const numWorkgroups = Math.min(workgroupSize, Math.ceil(vocabSize / workgroupSize));
  const topkLogits = acquireBuffer(workgroupSize * 4, void 0, "topk_logits");
  const topkIndices = acquireBuffer(workgroupSize * 4, void 0, "topk_indices");
  const outputBuffer = acquireBuffer(4, void 0, "sample_output");
  const uniformBuffer = createUniformBufferWithView(
    "sample_uniforms",
    32,
    (view) => {
      view.setUint32(0, vocabSize, true);
      view.setUint32(4, topK, true);
      view.setFloat32(8, temperature, true);
      view.setFloat32(12, randomValue, true);
      view.setUint32(16, padTokenId ?? 4294967295, true);
      view.setFloat32(20, logitSoftcap, true);
    },
    null,
    device2
  );
  const bindGroupLayout = getSampleBindGroupLayout(device2);
  const bindGroup = device2.createBindGroup({
    label: "sample_bind_group",
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: topkIndices } },
      { binding: 4, resource: { buffer: topkLogits } }
    ]
  });
  const encoder = device2.createCommandEncoder({ label: "sample_encoder" });
  const pass1 = encoder.beginComputePass({ label: "sample_phase1" });
  pass1.setPipeline(phase1Pipeline);
  pass1.setBindGroup(0, bindGroup);
  pass1.dispatchWorkgroups(numWorkgroups);
  pass1.end();
  const pass2 = encoder.beginComputePass({ label: "sample_phase2" });
  pass2.setPipeline(phase2Pipeline);
  pass2.setBindGroup(0, bindGroup);
  pass2.dispatchWorkgroups(1);
  pass2.end();
  const pass3 = encoder.beginComputePass({ label: "sample_phase3" });
  pass3.setPipeline(phase3Pipeline);
  pass3.setBindGroup(0, bindGroup);
  pass3.dispatchWorkgroups(1);
  pass3.end();
  device2.queue.submit([encoder.finish()]);
  const stagingBuffer = device2.createBuffer({
    label: "sample_staging",
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });
  const copyEncoder = device2.createCommandEncoder({ label: "sample_copy" });
  copyEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, 4);
  device2.queue.submit([copyEncoder.finish()]);
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const tokenId = new Uint32Array(stagingBuffer.getMappedRange())[0];
  stagingBuffer.unmap();
  stagingBuffer.destroy();
  uniformBuffer.destroy();
  releaseBuffer(topkLogits);
  releaseBuffer(topkIndices);
  releaseBuffer(outputBuffer);
  return tokenId;
}
async function recordArgmax(recorder, logits, vocabSize, options = {}) {
  const device2 = recorder.device;
  const argmaxPipeline = await createSamplePipeline(device2, "argmax");
  const reducePipeline = await createSamplePipeline(device2, "argmax_reduce");
  const numWorkgroups = Math.min(WORKGROUP_SIZES.DEFAULT, Math.ceil(vocabSize / WORKGROUP_SIZES.DEFAULT));
  const tempLogits = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, void 0, "argmax_temp_logits");
  const tempIndices = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, void 0, "argmax_temp_indices");
  const outputBuffer = acquireBuffer(4, void 0, "argmax_output");
  const padTokenId = options.padTokenId ?? 4294967295;
  const logitSoftcap = options.logitSoftcap ?? 0;
  const uniformBuffer = createUniformBufferWithView(
    "argmax_uniforms",
    32,
    (view) => {
      view.setUint32(0, vocabSize, true);
      view.setUint32(4, 1, true);
      view.setFloat32(8, 1, true);
      view.setFloat32(12, 0, true);
      view.setUint32(16, padTokenId, true);
      view.setFloat32(20, logitSoftcap, true);
    },
    recorder
  );
  const bindGroupLayout = getSampleBindGroupLayout(device2);
  const bindGroup = device2.createBindGroup({
    label: "argmax_bind_group",
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: tempIndices } },
      { binding: 4, resource: { buffer: tempLogits } }
    ]
  });
  const pass1 = recorder.beginComputePass("argmax_phase1");
  pass1.setPipeline(argmaxPipeline);
  pass1.setBindGroup(0, bindGroup);
  pass1.dispatchWorkgroups(numWorkgroups);
  pass1.end();
  const reduceBindGroup = device2.createBindGroup({
    label: "argmax_reduce_bind_group",
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: tempIndices } },
      { binding: 4, resource: { buffer: tempLogits } }
    ]
  });
  const pass2 = recorder.beginComputePass("argmax_phase2");
  pass2.setPipeline(reducePipeline);
  pass2.setBindGroup(0, reduceBindGroup);
  pass2.dispatchWorkgroups(1);
  pass2.end();
  recorder.trackTemporaryBuffer(tempLogits);
  recorder.trackTemporaryBuffer(tempIndices);
  return outputBuffer;
}
function seededRandom(seed) {
  const x = Math.sin(seed) * 1e4;
  return x - Math.floor(x);
}
function isGPUSamplingAvailable() {
  return getDevice() !== null;
}

// src/gpu/kernels/fused_ffn.ts
init_device();
init_buffer_pool();
init_tensor();
init_utils();
init_debug();
init_weight_buffer();
var FusedFFNKernel = class extends KernelBase {
  async getPipeline(variant) {
    return this.getPipelineFor("fused_ffn", variant);
  }
  dispatch(pipeline, bindGroup, workgroupsX, workgroupsY = 1) {
    this.dispatchKernel(pipeline, bindGroup, [workgroupsX, workgroupsY, 1], "fused_ffn");
  }
  record(recorder, pipeline, bindGroup, workgroupsX, workgroupsY = 1) {
    this.recordKernel(recorder, pipeline, bindGroup, [workgroupsX, workgroupsY, 1], "fused_ffn");
  }
};
function selectFFNVariant(batchSize, weightDtype, intermediateSize) {
  if (weightDtype === "q4k" && !isFusedQ4KDisabled()) {
    return batchSize > 1 ? "q4k_batched" : "q4k";
  }
  if (batchSize > 1) {
    return "batched";
  }
  if (weightDtype === "f16") {
    return "f16";
  }
  if (intermediateSize <= 1024) {
    return "multi";
  }
  return "default";
}
function createFFNUniformBuffer(device2, recorder, params) {
  return createUniformBufferWithView(
    "fused_ffn_uniforms",
    32,
    (view) => {
      view.setUint32(0, params.M, true);
      view.setUint32(4, params.hiddenSize, true);
      view.setUint32(8, params.intermediateSize, true);
      view.setFloat32(12, params.alpha, true);
      view.setUint32(16, params.activation === "silu" ? 0 : 1, true);
      if (params.isQ4K) {
        view.setUint32(20, Math.floor(params.hiddenSize / 256), true);
      }
    },
    recorder,
    device2
  );
}
async function runFusedFFN(input, W_gate, W_up, hiddenSize, intermediateSize, options = {}) {
  const device2 = getDevice();
  const {
    batchSize = 1,
    activation = "silu",
    alpha = 1,
    outputBuffer = null
  } = options;
  if (input.dtype !== "f32") {
    throw new Error("Fused FFN requires f32 activations");
  }
  const gateDtype = getWeightDtype(W_gate) ?? "f32";
  const upDtype = getWeightDtype(W_up) ?? "f32";
  if (gateDtype !== upDtype) {
    throw new Error(`Fused FFN requires matching gate/up dtypes (gate=${gateDtype}, up=${upDtype})`);
  }
  if (gateDtype !== "f16" && gateDtype !== "f32" && gateDtype !== "q4k") {
    throw new Error(`Fused FFN does not support ${gateDtype} weights`);
  }
  const isQ4K = gateDtype === "q4k";
  const variant = selectFFNVariant(batchSize, gateDtype, intermediateSize);
  trace.kernels(`FusedFFN: variant=${variant}, batch=${batchSize}, hidden=${hiddenSize}, intermediate=${intermediateSize}, activation=${activation}, isQ4K=${isQ4K}`);
  const kernel = new FusedFFNKernel(device2);
  const pipeline = await kernel.getPipeline(variant);
  const outputSize = batchSize * intermediateSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "fused_ffn_output");
  const uniformBuffer = createFFNUniformBuffer(device2, null, {
    M: batchSize,
    hiddenSize,
    intermediateSize,
    alpha,
    activation,
    isQ4K
  });
  const bindGroup = device2.createBindGroup({
    label: "fused_ffn_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: getBuffer(W_gate) } },
      { binding: 3, resource: { buffer: getBuffer(W_up) } },
      { binding: 4, resource: { buffer: output } }
    ]
  });
  let workgroupsX;
  let workgroupsY = 1;
  if (variant === "multi") {
    const outputsPerWg = 4;
    workgroupsX = Math.ceil(intermediateSize / outputsPerWg);
  } else if (variant === "q4k" || variant === "q4k_batched") {
    const colsPerWg = 32;
    workgroupsX = Math.ceil(intermediateSize / colsPerWg);
    workgroupsY = variant === "q4k_batched" ? batchSize : 1;
  } else if (variant === "batched") {
    workgroupsX = intermediateSize;
    workgroupsY = batchSize;
  } else {
    workgroupsX = intermediateSize;
  }
  kernel.dispatch(pipeline, bindGroup, workgroupsX, workgroupsY);
  uniformBuffer.destroy();
  return createTensor(output, "f32", [batchSize, intermediateSize], "fused_ffn_output");
}
async function recordFusedFFN(recorder, input, W_gate, W_up, hiddenSize, intermediateSize, options = {}) {
  const device2 = recorder.device;
  const {
    batchSize = 1,
    activation = "silu",
    alpha = 1,
    outputBuffer = null
  } = options;
  if (input.dtype !== "f32") {
    throw new Error("Fused FFN requires f32 activations");
  }
  const gateDtype = getWeightDtype(W_gate) ?? "f32";
  const upDtype = getWeightDtype(W_up) ?? "f32";
  if (gateDtype !== upDtype) {
    throw new Error(`Fused FFN requires matching gate/up dtypes (gate=${gateDtype}, up=${upDtype})`);
  }
  if (gateDtype !== "f16" && gateDtype !== "f32" && gateDtype !== "q4k") {
    throw new Error(`Fused FFN does not support ${gateDtype} weights`);
  }
  const isQ4K = gateDtype === "q4k";
  const variant = selectFFNVariant(batchSize, gateDtype, intermediateSize);
  trace.kernels(`FusedFFN record: variant=${variant}, batch=${batchSize}, hidden=${hiddenSize}, intermediate=${intermediateSize}, activation=${activation}, isQ4K=${isQ4K}`);
  const kernel = new FusedFFNKernel(device2);
  const pipeline = await kernel.getPipeline(variant);
  const outputSize = batchSize * intermediateSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "fused_ffn_output");
  const uniformBuffer = createFFNUniformBuffer(device2, recorder, {
    M: batchSize,
    hiddenSize,
    intermediateSize,
    alpha,
    activation,
    isQ4K
  });
  const bindGroup = device2.createBindGroup({
    label: "fused_ffn_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: getBuffer(W_gate) } },
      { binding: 3, resource: { buffer: getBuffer(W_up) } },
      { binding: 4, resource: { buffer: output } }
    ]
  });
  let workgroupsX;
  let workgroupsY = 1;
  if (variant === "multi") {
    const outputsPerWg = 4;
    workgroupsX = Math.ceil(intermediateSize / outputsPerWg);
  } else if (variant === "q4k" || variant === "q4k_batched") {
    const colsPerWg = 32;
    workgroupsX = Math.ceil(intermediateSize / colsPerWg);
    workgroupsY = variant === "q4k_batched" ? batchSize : 1;
  } else if (variant === "batched") {
    workgroupsX = intermediateSize;
    workgroupsY = batchSize;
  } else {
    workgroupsX = intermediateSize;
  }
  kernel.record(recorder, pipeline, bindGroup, workgroupsX, workgroupsY);
  return createTensor(output, "f32", [batchSize, intermediateSize], "fused_ffn_output");
}
function calculateFusedFFNSavings(batchSize, hiddenSize, intermediateSize) {
  const inputBytes = batchSize * hiddenSize * 4;
  const intermediateBytes = batchSize * intermediateSize * 4;
  const separateBytes = 2 * inputBytes + 3 * intermediateBytes;
  const fusedBytes = inputBytes + intermediateBytes;
  const savingsBytes = separateBytes - fusedBytes;
  const savingsPct = savingsBytes / separateBytes * 100;
  return {
    separateBytes,
    fusedBytes,
    savingsBytes,
    savingsPct
  };
}

// src/gpu/kernels/index.ts
init_fused_matmul_rmsnorm();
init_fused_matmul_rmsnorm();

// src/gpu/kernels/fused_matmul_residual.ts
init_device();
init_buffer_pool();
init_tensor();
init_weight_buffer();
init_dispatch();
init_utils();
init_debug();
function shouldUseFusedMatmulResidual(M) {
  return M === 1;
}
async function runMatmulResidualFused(input, weight, residual, options) {
  const device2 = getDevice();
  const {
    N,
    K,
    alpha = 1,
    outputBuffer = null
  } = options;
  const weightBuffer = getBuffer(weight);
  const outputDtype = input.dtype;
  trace.kernels(`MatmulResidualFused: N=${N}, K=${K}, alpha=${alpha}, dtype=${outputDtype}`);
  const pipeline = await getPipelineFast("fused_matmul_residual", "default");
  const output = outputBuffer || acquireBuffer(N * dtypeBytes(outputDtype), void 0, "matmul_residual_output");
  const uniformBuffer = createUniformBufferWithView(
    "matmul_residual_uniforms",
    32,
    // 8 u32s
    (view) => {
      view.setUint32(0, 1, true);
      view.setUint32(4, N, true);
      view.setUint32(8, K, true);
      view.setFloat32(12, alpha, true);
      view.setUint32(16, 1, true);
      view.setUint32(20, 0, true);
      view.setUint32(24, 0, true);
      view.setUint32(28, 0, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "matmul_residual_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weightBuffer } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: residual.buffer } }
    ]
  });
  const workgroups = N;
  dispatch(device2, pipeline, bindGroup, workgroups, "matmul_residual_fused");
  uniformBuffer.destroy();
  return createTensor(output, outputDtype, [1, N], "matmul_residual_output");
}
async function recordMatmulResidualFused(recorder, input, weight, residual, options) {
  const device2 = recorder.device;
  const {
    N,
    K,
    alpha = 1,
    outputBuffer = null
  } = options;
  const weightBuffer = getBuffer(weight);
  const outputDtype = input.dtype;
  const pipeline = await getPipelineFast("fused_matmul_residual", "default");
  const output = outputBuffer || acquireBuffer(N * dtypeBytes(outputDtype), void 0, "matmul_residual_output");
  const uniformBuffer = createUniformBufferWithView(
    "matmul_residual_uniforms",
    32,
    (view) => {
      view.setUint32(0, 1, true);
      view.setUint32(4, N, true);
      view.setUint32(8, K, true);
      view.setFloat32(12, alpha, true);
      view.setUint32(16, 1, true);
      view.setUint32(20, 0, true);
      view.setUint32(24, 0, true);
      view.setUint32(28, 0, true);
    },
    recorder
  );
  const bindGroup = device2.createBindGroup({
    label: "matmul_residual_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weightBuffer } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: residual.buffer } }
    ]
  });
  const workgroups = N;
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "matmul_residual_fused");
  return createTensor(output, outputDtype, [1, N], "matmul_residual_output");
}

// src/gpu/command-recorder.ts
init_device();
init_perf_guards();
init_uniform_cache();
init_debug();
var CommandRecorder = class _CommandRecorder {
  device;
  label;
  encoder;
  /** Temporary buffers to destroy after submit */
  tempBuffers;
  cleanupPromise = null;
  /** Track if already submitted */
  submitted;
  /** Operation count for debugging */
  opCount;
  // Profiling state
  profilingEnabled;
  querySet = null;
  queryBuffer = null;
  readbackBuffer = null;
  profileEntries = [];
  nextQueryIndex = 0;
  static MAX_QUERIES = 512;
  // 256 kernel pairs
  /**
   * @param device - GPU device (auto-detected if not provided)
   * @param label - Label for debugging
   * @param options - Recorder options (profiling, etc.)
   */
  constructor(device2 = null, label = "command_recorder", options = {}) {
    this.device = device2 || getDevice();
    if (!this.device) {
      throw new Error("[CommandRecorder] No GPU device available");
    }
    this.label = label;
    this.encoder = this.device.createCommandEncoder({ label });
    this.tempBuffers = [];
    this.cleanupPromise = null;
    this.submitted = false;
    this.opCount = 0;
    this.profilingEnabled = options.profile === true && hasFeature(FEATURES.TIMESTAMP_QUERY);
    if (this.profilingEnabled) {
      this._initProfiling();
    }
  }
  /**
   * Initialize GPU timestamp query resources for profiling.
   * @private
   */
  _initProfiling() {
    try {
      this.querySet = this.device.createQuerySet({
        type: "timestamp",
        count: _CommandRecorder.MAX_QUERIES
      });
      this.queryBuffer = this.device.createBuffer({
        label: `${this.label}_query_buffer`,
        size: _CommandRecorder.MAX_QUERIES * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
      });
      this.readbackBuffer = this.device.createBuffer({
        label: `${this.label}_readback_buffer`,
        size: _CommandRecorder.MAX_QUERIES * 8,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
      });
    } catch (e) {
      log.warn("CommandRecorder", `Failed to initialize profiling: ${e}`);
      this.profilingEnabled = false;
    }
  }
  /**
   * Check if profiling is enabled and available.
   */
  isProfilingEnabled() {
    return this.profilingEnabled;
  }
  /**
   * Create a temporary buffer that will be destroyed after submit.
   * Use for uniform buffers and other per-operation temporaries.
   *
   * @param size - Buffer size in bytes
   * @param usage - Buffer usage flags
   * @param label - Buffer label for debugging
   * @returns GPUBuffer
   */
  createTempBuffer(size, usage, label = "temp_buffer") {
    if (this.submitted) {
      throw new Error("[CommandRecorder] Cannot create buffers after submit");
    }
    const buffer = this.device.createBuffer({
      label: `${this.label}_${label}_${this.tempBuffers.length}`,
      size,
      usage
    });
    trackAllocation(size, label);
    this.tempBuffers.push(buffer);
    return buffer;
  }
  /**
   * Create an indirect dispatch buffer initialized with workgroup counts.
   * Buffer usage includes STORAGE so GPU kernels can update counts.
   */
  createIndirectDispatchBuffer(workgroups = [0, 0, 0], label = "indirect_dispatch") {
    const data = workgroups instanceof Uint32Array ? workgroups : new Uint32Array(workgroups);
    const size = Math.max(12, data.byteLength);
    const buffer = this.createTempBuffer(
      size,
      GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label
    );
    const source = data.buffer;
    this.device.queue.writeBuffer(buffer, 0, source, data.byteOffset, data.byteLength);
    return buffer;
  }
  /**
   * Update an indirect dispatch buffer with new workgroup counts.
   */
  writeIndirectDispatchBuffer(buffer, workgroups, offset = 0) {
    if (this.submitted) {
      throw new Error("[CommandRecorder] Cannot write buffers after submit");
    }
    const data = workgroups instanceof Uint32Array ? workgroups : new Uint32Array(workgroups);
    const source = data.buffer;
    this.device.queue.writeBuffer(buffer, offset, source, data.byteOffset, data.byteLength);
  }
  /**
   * Create a uniform buffer, write data, and track for cleanup.
   * Uses content-addressed caching for identical uniform data.
   *
   * @param data - Data to write
   * @param label - Buffer label
   * @returns GPUBuffer
   */
  createUniformBuffer(data, label = "uniforms") {
    const arrayBuffer = data instanceof ArrayBuffer ? data : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
    return getUniformCache().getOrCreate(arrayBuffer, label);
  }
  /**
   * Begin a compute pass on the encoder.
   * When profiling is enabled, injects timestampWrites to measure GPU execution time.
   *
   * @param label - Pass label for debugging (used as key in profile results)
   * @returns GPUComputePassEncoder
   */
  beginComputePass(label = "compute_pass") {
    if (this.submitted) {
      throw new Error("[CommandRecorder] Cannot begin pass after submit");
    }
    this.opCount++;
    const passLabel = `${this.label}_${label}_${this.opCount}`;
    if (this.profilingEnabled && this.querySet && this.nextQueryIndex + 2 <= _CommandRecorder.MAX_QUERIES) {
      const startIndex = this.nextQueryIndex;
      const endIndex = startIndex + 1;
      this.nextQueryIndex += 2;
      this.profileEntries.push({
        label,
        startQueryIndex: startIndex,
        endQueryIndex: endIndex
      });
      return this.encoder.beginComputePass({
        label: passLabel,
        timestampWrites: {
          querySet: this.querySet,
          beginningOfPassWriteIndex: startIndex,
          endOfPassWriteIndex: endIndex
        }
      });
    }
    return this.encoder.beginComputePass({
      label: passLabel
    });
  }
  /**
   * Get the raw encoder for advanced use cases.
   * @returns GPUCommandEncoder
   */
  getEncoder() {
    if (this.submitted) {
      throw new Error("[CommandRecorder] Cannot access encoder after submit");
    }
    return this.encoder;
  }
  /**
   * Track an externally created buffer for cleanup after submit.
   * Use for buffers created outside the recorder that need cleanup.
   *
   * @param buffer - Buffer to track for destruction
   */
  trackTemporaryBuffer(buffer) {
    if (this.submitted) {
      throw new Error("[CommandRecorder] Cannot track buffers after submit");
    }
    this.tempBuffers.push(buffer);
  }
  /**
   * Submit all recorded commands and clean up temporary buffers.
   * After calling this, the recorder cannot be reused.
   */
  submit() {
    if (this.submitted) {
      throw new Error("[CommandRecorder] Already submitted");
    }
    this.device.queue.submit([this.encoder.finish()]);
    this.submitted = true;
    const buffersToDestroy = this.tempBuffers;
    this.tempBuffers = [];
    this.cleanupPromise = this.device.queue.onSubmittedWorkDone().then(() => {
      for (const buffer of buffersToDestroy) {
        buffer.destroy();
      }
      getUniformCache().flushPendingDestruction();
    }).catch((err) => {
      log.warn("CommandRecorder", `Deferred cleanup failed: ${err.message}`);
    });
  }
  /**
   * Submit and wait for GPU to complete (useful for debugging/profiling).
   * Also flushes the uniform cache's pending destruction queue to clean up
   * any evicted buffers that were referenced by this command buffer.
   * @returns Promise that resolves when GPU work is done
   */
  async submitAndWait() {
    this.submit();
    if (this.cleanupPromise) {
      await this.cleanupPromise;
    } else {
      await this.device.queue.onSubmittedWorkDone();
      getUniformCache().flushPendingDestruction();
    }
  }
  /**
   * Get statistics about recorded operations.
   * @returns Statistics object
   */
  getStats() {
    return {
      opCount: this.opCount,
      tempBufferCount: this.tempBuffers.length,
      submitted: this.submitted
    };
  }
  /**
   * Abort recording without submitting (cleanup only).
   * Use if an error occurs during recording.
   */
  abort() {
    if (this.submitted)
      return;
    for (const buffer of this.tempBuffers) {
      buffer.destroy();
    }
    this.tempBuffers = [];
    this._destroyProfilingResources();
    this.submitted = true;
  }
  /**
   * Resolve profiling timestamps and return per-kernel timings.
   * Must be called after submit() and GPU work is done.
   *
   * Returns a map of kernel label to execution time in milliseconds.
   * Labels with multiple invocations are aggregated (e.g., 'matmul' across all layers).
   *
   * @returns Promise resolving to timing map, or null if profiling not enabled
   */
  async resolveProfileTimings() {
    if (!this.profilingEnabled || !this.querySet || !this.queryBuffer || !this.readbackBuffer) {
      return null;
    }
    if (!this.submitted) {
      throw new Error("[CommandRecorder] Must submit before resolving timings");
    }
    if (this.profileEntries.length === 0) {
      return {};
    }
    await this.device.queue.onSubmittedWorkDone();
    const maxIndex = Math.max(...this.profileEntries.map((e) => e.endQueryIndex)) + 1;
    const resolveEncoder = this.device.createCommandEncoder({ label: "profile_resolve" });
    resolveEncoder.resolveQuerySet(this.querySet, 0, maxIndex, this.queryBuffer, 0);
    resolveEncoder.copyBufferToBuffer(this.queryBuffer, 0, this.readbackBuffer, 0, maxIndex * 8);
    this.device.queue.submit([resolveEncoder.finish()]);
    if (!allowReadback("CommandRecorder.resolveProfileTimings")) {
      return null;
    }
    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const timestamps = new BigUint64Array(this.readbackBuffer.getMappedRange());
    const timings = {};
    for (const entry of this.profileEntries) {
      const startNs = timestamps[entry.startQueryIndex];
      const endNs = timestamps[entry.endQueryIndex];
      const durationMs = Number(endNs - startNs) / 1e6;
      if (durationMs < 0 || durationMs > 6e4) {
        continue;
      }
      if (timings[entry.label] !== void 0) {
        timings[entry.label] += durationMs;
      } else {
        timings[entry.label] = durationMs;
      }
    }
    this.readbackBuffer.unmap();
    this._destroyProfilingResources();
    return timings;
  }
  /**
   * Get a formatted profiling report.
   * Must be called after resolveProfileTimings().
   *
   * @param timings - Timings from resolveProfileTimings()
   * @returns Formatted string report
   */
  static formatProfileReport(timings) {
    const entries = Object.entries(timings).sort((a, b) => b[1] - a[1]);
    const total = entries.reduce((sum, [, t]) => sum + t, 0);
    let report = "GPU Profile Report\n";
    report += "\u2500".repeat(50) + "\n";
    report += "Kernel".padEnd(25) + "Time (ms)".padStart(12) + "%".padStart(8) + "\n";
    report += "\u2500".repeat(50) + "\n";
    for (const [label, time] of entries) {
      const pct = (time / total * 100).toFixed(1);
      report += label.padEnd(25) + time.toFixed(2).padStart(12) + pct.padStart(8) + "\n";
    }
    report += "\u2500".repeat(50) + "\n";
    report += "TOTAL".padEnd(25) + total.toFixed(2).padStart(12) + "100.0".padStart(8) + "\n";
    return report;
  }
  /**
   * Clean up profiling resources.
   * @private
   */
  _destroyProfilingResources() {
    if (this.querySet) {
      this.querySet.destroy();
      this.querySet = null;
    }
    if (this.queryBuffer) {
      this.queryBuffer.destroy();
      this.queryBuffer = null;
    }
    if (this.readbackBuffer) {
      this.readbackBuffer.destroy();
      this.readbackBuffer = null;
    }
    this.profileEntries = [];
  }
};
function createCommandRecorder(label = "command_recorder", options) {
  return new CommandRecorder(null, label, options);
}
function createProfilingRecorder(label = "profiled_recorder") {
  return new CommandRecorder(null, label, { profile: true });
}

// src/gpu/kernel-benchmark.ts
init_device();
init_buffer_pool();
init_tensor();
init_constants();
init_debug();
var DEFAULT_CONFIG2 = {
  warmupIterations: PERFORMANCE.WARMUP_RUNS,
  timedIterations: PERFORMANCE.TIMED_RUNS,
  modelConfig: {
    hiddenSize: 1152,
    // Gemma 3 1B
    intermediateSize: 6912,
    // Gemma 3 1B
    numHeads: 4,
    // Gemma 3 1B
    numKVHeads: 1,
    // Gemma 3 1B (GQA)
    headDim: 256,
    // Gemma 3 1B
    vocabSize: 262144,
    // Gemma 3 1B
    numLayers: 26
    // Gemma 3 1B
  }
};
function calculateStats(times) {
  const sorted = [...times].sort((a, b) => a - b);
  const n = sorted.length;
  const mean = times.reduce((a, b) => a + b, 0) / n;
  const variance = times.reduce((sum, t) => sum + (t - mean) ** 2, 0) / n;
  const stddev = Math.sqrt(variance);
  return {
    median: sorted[Math.floor(n / 2)],
    min: sorted[0],
    max: sorted[n - 1],
    p95: sorted[Math.floor(n * 0.95)],
    p99: sorted[Math.floor(n * 0.99)],
    stddev,
    mean
  };
}
function estimateFLOPS(kernel, config2, latencyMs) {
  let flops = 0;
  switch (kernel) {
    case "matmul":
      flops = 2 * (config2.M || 1) * (config2.N || 1) * (config2.K || 1);
      break;
    case "attention":
      const qk = 2 * (config2.seqLen || 1) * (config2.kvLen || 1) * (config2.headDim || 128) * (config2.numHeads || 1);
      const sm = 5 * (config2.seqLen || 1) * (config2.kvLen || 1) * (config2.numHeads || 1);
      const v = 2 * (config2.seqLen || 1) * (config2.kvLen || 1) * (config2.headDim || 128) * (config2.numHeads || 1);
      flops = qk + sm + v;
      break;
    case "rmsnorm":
      flops = 4 * (config2.size || 1);
      break;
    case "silu":
      flops = 2 * (config2.size || 1);
      break;
    default:
      flops = config2.size || config2.elements || 1;
  }
  const gflops = flops / (latencyMs * 1e6);
  const theoretical = 1e4;
  return { gflops, theoretical };
}
function createTestBuffer(size, label) {
  const buffer = acquireBuffer(size, void 0, label);
  const device2 = getDevice();
  const data = new Float32Array(size / 4);
  for (let i = 0; i < data.length; i++) {
    data[i] = (Math.random() - 0.5) * 2;
  }
  device2.queue.writeBuffer(buffer, 0, data);
  return buffer;
}
async function benchmarkKernel(name, variant, config2, runFn, warmupIterations, timedIterations) {
  const device2 = getDevice();
  for (let i = 0; i < warmupIterations; i++) {
    await runFn();
    await device2.queue.onSubmittedWorkDone();
  }
  const times = [];
  for (let i = 0; i < timedIterations; i++) {
    const start = performance.now();
    await runFn();
    await device2.queue.onSubmittedWorkDone();
    times.push(performance.now() - start);
  }
  const stats = calculateStats(times);
  const flopsInfo = estimateFLOPS(name, config2, stats.median);
  const readBytes = config2.readBytes || config2.size * 4 || 0;
  const writeBytes = config2.writeBytes || config2.size * 4 || 0;
  const totalBytes = readBytes + writeBytes;
  const gbPerSec = totalBytes / (stats.median * 1e6);
  return {
    kernel: name,
    variant,
    config: config2,
    latency: {
      median_ms: stats.median,
      min_ms: stats.min,
      max_ms: stats.max,
      p95_ms: stats.p95,
      p99_ms: stats.p99,
      stddev_ms: stats.stddev
    },
    throughput: {
      gb_per_sec: gbPerSec,
      elements_per_sec: (config2.size || config2.elements || 1) / (stats.median / 1e3)
    },
    flops: {
      gflops: flopsInfo.gflops,
      theoretical_gflops: flopsInfo.theoretical,
      efficiency_pct: flopsInfo.gflops / flopsInfo.theoretical * 100
    },
    memory: {
      read_bytes: readBytes,
      write_bytes: writeBytes,
      total_bytes: totalBytes
    },
    iterations: timedIterations,
    warmup_iterations: warmupIterations,
    timestamp: (/* @__PURE__ */ new Date()).toISOString()
  };
}
async function benchmarkMatmul(M, N, K, options = {}) {
  const config2 = { ...DEFAULT_CONFIG2, ...options };
  const device2 = getDevice();
  const A = createTestBuffer(M * K * 4, "bench_A");
  const B = createTestBuffer(K * N * 4, "bench_B");
  const tensorA = createTensor(A, "f32", [M, K], "bench_A");
  const result = await benchmarkKernel(
    "matmul",
    "f32",
    {
      M,
      N,
      K,
      readBytes: (M * K + K * N) * 4,
      writeBytes: M * N * 4
    },
    async () => {
      const C = await runMatmul(tensorA, B, M, N, K);
      releaseBuffer(C.buffer);
    },
    config2.warmupIterations,
    config2.timedIterations
  );
  releaseBuffer(A);
  releaseBuffer(B);
  return result;
}
async function benchmarkAttentionDecode(numHeads, headDim, kvLen, options = {}) {
  const config2 = { ...DEFAULT_CONFIG2, ...options };
  const QBuf = createTestBuffer(numHeads * headDim * 4, "bench_Q");
  const KBuf = createTestBuffer(kvLen * numHeads * headDim * 4, "bench_K");
  const VBuf = createTestBuffer(kvLen * numHeads * headDim * 4, "bench_V");
  const Q = createTensor(QBuf, "f32", [1, numHeads * headDim], "bench_Q");
  const K = createTensor(KBuf, "f32", [kvLen, numHeads * headDim], "bench_K");
  const V = createTensor(VBuf, "f32", [kvLen, numHeads * headDim], "bench_V");
  const result = await benchmarkKernel(
    "attention",
    "decode",
    {
      seqLen: 1,
      kvLen,
      numHeads,
      headDim,
      readBytes: (numHeads * headDim + 2 * kvLen * numHeads * headDim) * 4,
      writeBytes: numHeads * headDim * 4
    },
    async () => {
      const out = await runAttention(Q, K, V, null, numHeads, headDim, {
        seqLen: 1,
        kvLen,
        numKVHeads: numHeads
      });
      releaseBuffer(out.buffer);
    },
    config2.warmupIterations,
    config2.timedIterations
  );
  releaseBuffer(QBuf);
  releaseBuffer(KBuf);
  releaseBuffer(VBuf);
  return result;
}
async function benchmarkRMSNorm(batchSize, hiddenSize, options = {}) {
  const config2 = { ...DEFAULT_CONFIG2, ...options };
  const size = batchSize * hiddenSize;
  const input = createTestBuffer(size * 4, "bench_input");
  const weight = createTestBuffer(hiddenSize * 4, "bench_weight");
  const inputTensor = createTensor(input, "f32", [batchSize, hiddenSize], "bench_input");
  const result = await benchmarkKernel(
    "rmsnorm",
    "default",
    {
      batchSize,
      hiddenSize,
      size,
      readBytes: (size + hiddenSize) * 4,
      writeBytes: size * 4
    },
    async () => {
      const out = await runRMSNorm(inputTensor, weight, 1e-6, { batchSize, hiddenSize });
      releaseBuffer(out.buffer);
    },
    config2.warmupIterations,
    config2.timedIterations
  );
  releaseBuffer(input);
  releaseBuffer(weight);
  return result;
}
async function benchmarkSiLU(size, options = {}) {
  const config2 = { ...DEFAULT_CONFIG2, ...options };
  const input = createTestBuffer(size * 4, "bench_input");
  const inputTensor = createTensor(input, "f32", [size], "bench_input");
  const result = await benchmarkKernel(
    "silu",
    "default",
    {
      size,
      readBytes: size * 4,
      writeBytes: size * 4
    },
    async () => {
      const out = await runSiLU(inputTensor, { size });
      releaseBuffer(out.buffer);
    },
    config2.warmupIterations,
    config2.timedIterations
  );
  releaseBuffer(input);
  return result;
}
async function benchmarkMatmulRMSNormFused(N, K, options = {}) {
  const config2 = { ...DEFAULT_CONFIG2, ...options };
  const { runMatmulRMSNormFused: runMatmulRMSNormFused2, shouldUseFusedMatmulRMSNorm: shouldUseFusedMatmulRMSNorm2 } = await Promise.resolve().then(() => (init_fused_matmul_rmsnorm(), fused_matmul_rmsnorm_exports));
  if (!shouldUseFusedMatmulRMSNorm2(1, N)) {
    throw new Error(`Fused kernel not supported for N=${N} with current thresholds.`);
  }
  const input = createTestBuffer(K * 4, "bench_input");
  const weight = createTestBuffer(K * N * 4, "bench_weight");
  const normWeight = createTestBuffer(N * 4, "bench_norm_weight");
  const residual = createTestBuffer(N * 4, "bench_residual");
  const inputTensor = createTensor(input, "f32", [1, K], "bench_input");
  const residualTensor = createTensor(residual, "f32", [1, N], "bench_residual");
  const separateResult = await benchmarkKernel(
    "matmul+rmsnorm",
    "separate",
    {
      M: 1,
      N,
      K,
      readBytes: (K + K * N + N + N) * 4,
      // input + weight + norm_weight + residual
      writeBytes: N * 4
    },
    async () => {
      const matmulOut = await runMatmul(inputTensor, weight, 1, N, K);
      const normOut = await runRMSNorm(matmulOut, normWeight, 1e-6, {
        batchSize: 1,
        hiddenSize: N,
        residual: residualTensor
      });
      releaseBuffer(matmulOut.buffer);
      releaseBuffer(normOut.buffer);
    },
    config2.warmupIterations,
    config2.timedIterations
  );
  const fusedResult = await benchmarkKernel(
    "matmul+rmsnorm",
    "fused",
    {
      M: 1,
      N,
      K,
      readBytes: (K + K * N + N + N) * 4,
      writeBytes: N * 4
    },
    async () => {
      const out = await runMatmulRMSNormFused2(inputTensor, weight, normWeight, {
        N,
        K,
        eps: 1e-6,
        residual
        // fused kernel still takes GPUBuffer for residual
      });
      releaseBuffer(out.buffer);
    },
    config2.warmupIterations,
    config2.timedIterations
  );
  releaseBuffer(input);
  releaseBuffer(weight);
  releaseBuffer(normWeight);
  releaseBuffer(residual);
  const speedup = separateResult.latency.median_ms / fusedResult.latency.median_ms;
  const comparison = {
    baseline: separateResult,
    optimized: fusedResult,
    speedup,
    latency_reduction_pct: (1 - fusedResult.latency.median_ms / separateResult.latency.median_ms) * 100,
    throughput_increase_pct: (speedup - 1) * 100
  };
  trace.perf(`Benchmark: Matmul+RMSNorm (N=${N}, K=${K}): Separate: ${separateResult.latency.median_ms.toFixed(3)}ms, Fused: ${fusedResult.latency.median_ms.toFixed(3)}ms, Speedup: ${speedup.toFixed(2)}x`);
  return { separate: separateResult, fused: fusedResult, comparison };
}
async function benchmarkDecodePass(options = {}) {
  const config2 = { ...DEFAULT_CONFIG2.modelConfig, ...options.modelConfig };
  const device2 = getDevice();
  const limits = getDeviceLimits();
  const caps = getKernelCapabilities();
  const results = [];
  trace.perf(`Benchmark: Starting decode pass benchmark... Model config: hidden=${config2.hiddenSize}, intermediate=${config2.intermediateSize}, heads=${config2.numHeads}`);
  trace.perf("Benchmark: Running RMSNorm...");
  results.push(await benchmarkRMSNorm(1, config2.hiddenSize, options));
  trace.perf("Benchmark: Running QKV projection...");
  const qkvDim = (config2.numHeads + 2 * config2.numKVHeads) * config2.headDim;
  results.push(await benchmarkMatmul(1, qkvDim, config2.hiddenSize, options));
  trace.perf("Benchmark: Running Attention decode...");
  const kvLen = 512;
  results.push(await benchmarkAttentionDecode(config2.numHeads, config2.headDim, kvLen, options));
  trace.perf("Benchmark: Running output projection...");
  results.push(await benchmarkMatmul(1, config2.hiddenSize, config2.numHeads * config2.headDim, options));
  trace.perf("Benchmark: Running FFN gate+up...");
  results.push(await benchmarkMatmul(1, config2.intermediateSize * 2, config2.hiddenSize, options));
  trace.perf("Benchmark: Running SiLU...");
  results.push(await benchmarkSiLU(config2.intermediateSize, options));
  trace.perf("Benchmark: Running FFN down...");
  results.push(await benchmarkMatmul(1, config2.hiddenSize, config2.intermediateSize, options));
  trace.perf("Benchmark: Running final RMSNorm...");
  results.push(await benchmarkRMSNorm(1, config2.hiddenSize, options));
  trace.perf("Benchmark: Running LM head...");
  results.push(await benchmarkMatmul(1, config2.vocabSize, config2.hiddenSize, options));
  const perLayerLatency = results.slice(0, 8).reduce((sum, r) => sum + r.latency.median_ms, 0);
  const lmHeadLatency = results[8].latency.median_ms;
  const totalDecodeLatency = perLayerLatency * config2.numLayers + lmHeadLatency;
  const tokPerSec = 1e3 / totalDecodeLatency;
  const sortedByLatency = [...results].sort((a, b) => b.latency.median_ms - a.latency.median_ms);
  const bottleneck = sortedByLatency[0];
  const bottleneckPct = bottleneck.latency.median_ms / totalDecodeLatency * 100;
  const report = {
    device_info: {
      vendor: "WebGPU",
      architecture: "Unknown",
      max_workgroup_size: limits?.maxComputeInvocationsPerWorkgroup || 256,
      max_shared_memory: limits?.maxComputeWorkgroupStorageSize || 16384,
      has_f16: caps.hasF16,
      has_subgroups: caps.hasSubgroups
    },
    model_config: {
      name: "Gemma 3 1B",
      hidden_size: config2.hiddenSize,
      intermediate_size: config2.intermediateSize,
      num_heads: config2.numHeads,
      num_kv_heads: config2.numKVHeads,
      head_dim: config2.headDim,
      num_layers: config2.numLayers,
      vocab_size: config2.vocabSize
    },
    results,
    comparisons: [],
    summary: {
      total_decode_latency_ms: totalDecodeLatency,
      estimated_tok_per_sec: tokPerSec,
      bottleneck_kernel: `${bottleneck.kernel}/${bottleneck.variant}`,
      bottleneck_percentage: bottleneckPct
    },
    generated_at: (/* @__PURE__ */ new Date()).toISOString()
  };
  trace.perf(`Benchmark Summary: Total decode latency: ${totalDecodeLatency.toFixed(2)}ms, Estimated tokens/sec: ${tokPerSec.toFixed(1)}, Bottleneck: ${bottleneck.kernel} (${bottleneckPct.toFixed(1)}%)`);
  return report;
}
function compareBenchmarks(baseline, optimized) {
  const speedup = baseline.latency.median_ms / optimized.latency.median_ms;
  const latencyReduction = (baseline.latency.median_ms - optimized.latency.median_ms) / baseline.latency.median_ms * 100;
  const throughputIncrease = (optimized.throughput.gb_per_sec - baseline.throughput.gb_per_sec) / baseline.throughput.gb_per_sec * 100;
  return {
    baseline,
    optimized,
    speedup,
    latency_reduction_pct: latencyReduction,
    throughput_increase_pct: throughputIncrease
  };
}
function exportBenchmarkJSON(report) {
  return JSON.stringify(report, null, 2);
}
function printBenchmarkReport(report) {
  log.info("Benchmark", "=".repeat(60));
  log.info("Benchmark", "KERNEL BENCHMARK REPORT");
  log.info("Benchmark", "=".repeat(60));
  log.info("Benchmark", "Device Info:");
  log.info("Benchmark", `  Max Workgroup Size: ${report.device_info.max_workgroup_size}`);
  log.info("Benchmark", `  Max Shared Memory: ${(report.device_info.max_shared_memory / 1024).toFixed(1)}KB`);
  log.info("Benchmark", `  F16 Support: ${report.device_info.has_f16}`);
  log.info("Benchmark", `  Subgroup Support: ${report.device_info.has_subgroups}`);
  log.info("Benchmark", "Model Config:");
  log.info("Benchmark", `  Name: ${report.model_config.name}`);
  log.info("Benchmark", `  Hidden Size: ${report.model_config.hidden_size}`);
  log.info("Benchmark", `  Intermediate Size: ${report.model_config.intermediate_size}`);
  log.info("Benchmark", `  Heads: ${report.model_config.num_heads} (KV: ${report.model_config.num_kv_heads})`);
  log.info("Benchmark", "Kernel Results:");
  log.info("Benchmark", "-".repeat(60));
  log.info("Benchmark", "Kernel           | Latency (ms) | GB/s    | GFLOPS");
  log.info("Benchmark", "-".repeat(60));
  for (const r of report.results) {
    log.info(
      "Benchmark",
      `${(r.kernel + "/" + r.variant).padEnd(16)} | ${r.latency.median_ms.toFixed(3).padStart(12)} | ${r.throughput.gb_per_sec.toFixed(2).padStart(7)} | ${r.flops.gflops.toFixed(1).padStart(7)}`
    );
  }
  log.info("Benchmark", "-".repeat(60));
  log.info("Benchmark", "Summary:");
  log.info("Benchmark", `  Total Decode Latency: ${report.summary.total_decode_latency_ms.toFixed(2)}ms`);
  log.info("Benchmark", `  Estimated Tokens/sec: ${report.summary.estimated_tok_per_sec.toFixed(1)}`);
  log.info("Benchmark", `  Bottleneck: ${report.summary.bottleneck_kernel} (${report.summary.bottleneck_percentage.toFixed(1)}%)`);
  if (report.comparisons.length > 0) {
    log.info("Benchmark", "Comparisons:");
    for (const c of report.comparisons) {
      log.info("Benchmark", `  ${c.baseline.kernel}: ${c.speedup.toFixed(2)}x speedup`);
    }
  }
  log.info("Benchmark", "=".repeat(60));
}

// src/gpu/kernels/split_qkv.ts
init_device();
init_buffer_pool();
init_tensor();
init_constants();
init_dispatch();
init_utils();
async function runSplitQKV(qkvTensor, options) {
  const device2 = getDevice();
  const { numTokens, qSize, kSize, vSize, qTensor = null, kTensor = null, vTensor = null } = options;
  const pipeline = await getPipelineFast("split_qkv", "default");
  const outputDtype = qkvTensor.dtype;
  const bytesPerElement = dtypeBytes(outputDtype);
  const qBuffer = qTensor?.buffer || acquireBuffer(numTokens * qSize * bytesPerElement, void 0, "Q");
  const kBuffer = kTensor?.buffer || acquireBuffer(numTokens * kSize * bytesPerElement, void 0, "K");
  const vBuffer = vTensor?.buffer || acquireBuffer(numTokens * vSize * bytesPerElement, void 0, "V");
  const uniformBuffer = createUniformBufferWithView(
    "split_qkv_uniforms",
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, qSize, true);
      view.setUint32(8, kSize, true);
      view.setUint32(12, vSize, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "split_qkv_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: qkvTensor.buffer } },
      { binding: 2, resource: { buffer: qBuffer } },
      { binding: 3, resource: { buffer: kBuffer } },
      { binding: 4, resource: { buffer: vBuffer } }
    ]
  });
  const totalElements = numTokens * (qSize + kSize + vSize);
  const workgroups = Math.ceil(totalElements / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "split_qkv");
  uniformBuffer.destroy();
  const Q = qTensor || createTensor(qBuffer, outputDtype, [numTokens, qSize], "Q");
  const K = kTensor || createTensor(kBuffer, outputDtype, [numTokens, kSize], "K");
  const V = vTensor || createTensor(vBuffer, outputDtype, [numTokens, vSize], "V");
  return { Q, K, V };
}
async function recordSplitQKV(recorder, qkvTensor, options) {
  const device2 = recorder.device;
  const { numTokens, qSize, kSize, vSize, qTensor = null, kTensor = null, vTensor = null } = options;
  const pipeline = await getPipelineFast("split_qkv", "default");
  const outputDtype = qkvTensor.dtype;
  const bytesPerElement = dtypeBytes(outputDtype);
  const qBuffer = qTensor?.buffer || acquireBuffer(numTokens * qSize * bytesPerElement, void 0, "Q");
  const kBuffer = kTensor?.buffer || acquireBuffer(numTokens * kSize * bytesPerElement, void 0, "K");
  const vBuffer = vTensor?.buffer || acquireBuffer(numTokens * vSize * bytesPerElement, void 0, "V");
  const uniformBuffer = createUniformBufferWithView(
    "split_qkv_uniforms",
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, qSize, true);
      view.setUint32(8, kSize, true);
      view.setUint32(12, vSize, true);
    },
    recorder
  );
  const bindGroup = device2.createBindGroup({
    label: "split_qkv_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: qkvTensor.buffer } },
      { binding: 2, resource: { buffer: qBuffer } },
      { binding: 3, resource: { buffer: kBuffer } },
      { binding: 4, resource: { buffer: vBuffer } }
    ]
  });
  const totalElements = numTokens * (qSize + kSize + vSize);
  const workgroups = Math.ceil(totalElements / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "split_qkv");
  const Q = qTensor || createTensor(qBuffer, outputDtype, [numTokens, qSize], "Q");
  const K = kTensor || createTensor(kBuffer, outputDtype, [numTokens, kSize], "K");
  const V = vTensor || createTensor(vBuffer, outputDtype, [numTokens, vSize], "V");
  return { Q, K, V };
}

// src/gpu/perf-profiler.ts
init_device();
init_debug();
var profilingEnabled = false;
var profileEntries = [];
var profileStartTime = 0;
function isProfilingEnabled() {
  if (typeof window !== "undefined") {
    return Boolean(window.DOPPLER_PROFILE);
  }
  return profilingEnabled;
}
function setProfilingEnabled(enabled) {
  profilingEnabled = enabled;
  if (typeof window !== "undefined") {
    window.DOPPLER_PROFILE = enabled;
  }
}
function clearProfile() {
  profileEntries = [];
  profileStartTime = 0;
}
function startProfileSession() {
  clearProfile();
  profileStartTime = performance.now();
}
function recordProfileEntry(name, category, startTime, endTime, metadata) {
  if (!isProfilingEnabled())
    return;
  profileEntries.push({
    name,
    category,
    startTime,
    endTime,
    duration: endTime - startTime,
    metadata
  });
}
async function profileAsync(name, category, fn, metadata) {
  if (!isProfilingEnabled()) {
    return fn();
  }
  const startTime = performance.now();
  try {
    const result = await fn();
    const endTime = performance.now();
    recordProfileEntry(name, category, startTime, endTime, metadata);
    return result;
  } catch (error) {
    const endTime = performance.now();
    recordProfileEntry(name, category, startTime, endTime, { ...metadata, error: true });
    throw error;
  }
}
function profileSync(name, category, fn, metadata) {
  if (!isProfilingEnabled()) {
    return fn();
  }
  const startTime = performance.now();
  try {
    const result = fn();
    const endTime = performance.now();
    recordProfileEntry(name, category, startTime, endTime, metadata);
    return result;
  } catch (error) {
    const endTime = performance.now();
    recordProfileEntry(name, category, startTime, endTime, { ...metadata, error: true });
    throw error;
  }
}
async function profileKernel(name, dispatchFn, metadata) {
  if (!isProfilingEnabled()) {
    dispatchFn();
    return;
  }
  const device2 = getDevice();
  const startTime = performance.now();
  dispatchFn();
  await device2.queue.onSubmittedWorkDone();
  const endTime = performance.now();
  recordProfileEntry(name, "kernel", startTime, endTime, metadata);
}
function getProfileReport() {
  const entries = [...profileEntries];
  const totalTime = entries.reduce((sum, e) => sum + e.duration, 0);
  const kernelEntries = entries.filter((e) => e.category === "kernel");
  const memoryEntries = entries.filter((e) => e.category === "memory");
  const syncEntries = entries.filter((e) => e.category === "sync");
  const otherEntries = entries.filter((e) => e.category === "other");
  const kernelTime = kernelEntries.reduce((sum, e) => sum + e.duration, 0);
  const memoryTime = memoryEntries.reduce((sum, e) => sum + e.duration, 0);
  const syncTime = syncEntries.reduce((sum, e) => sum + e.duration, 0);
  const otherTime = otherEntries.reduce((sum, e) => sum + e.duration, 0);
  const byName = /* @__PURE__ */ new Map();
  for (const entry of entries) {
    const existing = byName.get(entry.name) || { totalTime: 0, count: 0 };
    existing.totalTime += entry.duration;
    existing.count += 1;
    byName.set(entry.name, existing);
  }
  const breakdown = Array.from(byName.entries()).map(([name, stats]) => ({
    name,
    totalTime: stats.totalTime,
    count: stats.count,
    avgTime: stats.totalTime / stats.count,
    pctOfTotal: stats.totalTime / totalTime * 100
  })).sort((a, b) => b.totalTime - a.totalTime);
  const bottlenecks = [];
  if (syncEntries.length > entries.length * 0.1) {
    bottlenecks.push({
      name: "Excessive GPU Syncs",
      impact: syncTime / totalTime,
      suggestion: "Use CommandRecorder to batch operations and reduce syncs"
    });
  }
  if (memoryTime > kernelTime) {
    bottlenecks.push({
      name: "Memory Bandwidth Bound",
      impact: memoryTime / totalTime,
      suggestion: "Consider kernel fusion to reduce memory traffic"
    });
  }
  const smallKernels = kernelEntries.filter((e) => e.duration < 0.1);
  if (smallKernels.length > kernelEntries.length * 0.5) {
    const smallKernelTime = smallKernels.reduce((sum, e) => sum + e.duration, 0);
    bottlenecks.push({
      name: "Kernel Launch Overhead",
      impact: smallKernelTime / totalTime,
      suggestion: "Batch small kernels or increase work per kernel"
    });
  }
  for (const item of breakdown.slice(0, 3)) {
    if (item.pctOfTotal > 30) {
      bottlenecks.push({
        name: `${item.name} dominates (${item.pctOfTotal.toFixed(1)}%)`,
        impact: item.pctOfTotal / 100,
        suggestion: `Optimize ${item.name} or check if it's using optimal variant`
      });
    }
  }
  return {
    entries,
    summary: {
      totalTime,
      kernelTime,
      memoryTime,
      syncTime,
      otherTime,
      kernelCount: kernelEntries.length,
      memoryOps: memoryEntries.length,
      syncOps: syncEntries.length
    },
    breakdown,
    bottlenecks
  };
}
function printProfileReport(report) {
  const r = report || getProfileReport();
  log.info("Profile", "=".repeat(60));
  log.info("Profile", "PERFORMANCE PROFILE REPORT");
  log.info("Profile", "=".repeat(60));
  log.info("Profile", "Summary:");
  log.info("Profile", `  Total Time: ${r.summary.totalTime.toFixed(2)}ms`);
  log.info("Profile", `  Kernel Time: ${r.summary.kernelTime.toFixed(2)}ms (${(r.summary.kernelTime / r.summary.totalTime * 100).toFixed(1)}%)`);
  log.info("Profile", `  Memory Time: ${r.summary.memoryTime.toFixed(2)}ms (${(r.summary.memoryTime / r.summary.totalTime * 100).toFixed(1)}%)`);
  log.info("Profile", `  Sync Time: ${r.summary.syncTime.toFixed(2)}ms (${(r.summary.syncTime / r.summary.totalTime * 100).toFixed(1)}%)`);
  log.info("Profile", `  Kernel Count: ${r.summary.kernelCount}`);
  log.info("Profile", "Top Operations:");
  log.info("Profile", "-".repeat(60));
  log.info("Profile", "Operation                    | Time (ms) | Count | % Total");
  log.info("Profile", "-".repeat(60));
  for (const item of r.breakdown.slice(0, 10)) {
    log.info(
      "Profile",
      `${item.name.padEnd(28)} | ${item.totalTime.toFixed(2).padStart(9)} | ${item.count.toString().padStart(5)} | ${item.pctOfTotal.toFixed(1).padStart(7)}%`
    );
  }
  if (r.bottlenecks.length > 0) {
    log.info("Profile", "Bottlenecks:");
    log.info("Profile", "-".repeat(60));
    for (const b of r.bottlenecks) {
      log.info("Profile", `  [${(b.impact * 100).toFixed(0)}%] ${b.name}`);
      log.info("Profile", `       Fix: ${b.suggestion}`);
    }
  }
  log.info("Profile", "=".repeat(60));
}
function exportProfileJSON(report) {
  return JSON.stringify(report || getProfileReport(), null, 2);
}
function analyzeDecodePerformance(tokensGenerated, totalTimeMs, targetTokPerSec = 40) {
  const currentTokPerSec = tokensGenerated / totalTimeMs * 1e3;
  const gap = targetTokPerSec / currentTokPerSec;
  const suggestions = [];
  if (gap > 5) {
    suggestions.push("Critical: Enable CommandRecorder for batched execution");
    suggestions.push("Critical: Verify GEMV kernels are being used for M=1 matmuls");
    suggestions.push("Critical: Check if subgroups are available and enabled");
  }
  if (gap > 3) {
    suggestions.push("Use fused FFN kernel to reduce memory bandwidth");
    suggestions.push("Enable optimized decode attention kernel");
    suggestions.push("Profile individual kernels to find dominant operation");
  }
  if (gap > 1.5) {
    suggestions.push("Consider F16 KV cache to reduce memory traffic");
    suggestions.push("Tune workgroup sizes for your GPU");
    suggestions.push("Check for unnecessary GPU syncs");
  }
  return {
    currentTokPerSec,
    targetTokPerSec,
    gap,
    suggestions
  };
}

// kernel-tests/src/reference/index.ts
var reference_exports = {};
__export(reference_exports, {
  argmaxRef: () => argmaxRef,
  attentionRef: () => attentionRef,
  batchGatherRef: () => batchGatherRef,
  batchMatmulRef: () => batchMatmulRef,
  computeRopeFreqs: () => computeRopeFreqs,
  createCausalMask: () => createCausalMask,
  dequantInt4Ref: () => dequantInt4Ref,
  dequantInt8Ref: () => dequantInt8Ref,
  dequantQ4_0Ref: () => dequantQ4_0Ref,
  dequantQ4_KRef: () => dequantQ4_KRef,
  dequantizeQ4_KBlockRef: () => dequantizeQ4_KBlockRef,
  flashAttentionRef: () => flashAttentionRef,
  float32ToFloat16: () => float32ToFloat16,
  gatherRef: () => gatherRef,
  gatherWithPosRef: () => gatherWithPosRef,
  logSoftmaxRef: () => logSoftmaxRef,
  matmulRef: () => matmulRef,
  matvecRef: () => matvecRef,
  moeComputeAssignmentsRef: () => moeComputeAssignmentsRef,
  moeGatherRef: () => moeGatherRef,
  mqaRef: () => mqaRef,
  quantizeQ4_KBlockRef: () => quantizeQ4_KBlockRef,
  quantizeQ4_KRef: () => quantizeQ4_KRef,
  residualAddInplaceRef: () => residualAddInplaceRef,
  residualAddRef: () => residualAddRef,
  rmsNormNoWeightRef: () => rmsNormNoWeightRef,
  rmsNormRef: () => rmsNormRef,
  ropeInterleavedRef: () => ropeInterleavedRef,
  ropeRef: () => ropeRef,
  sampleTopKRef: () => sampleTopKRef,
  scaledResidualAddRef: () => scaledResidualAddRef,
  scatterAddAccumulateRef: () => scatterAddAccumulateRef,
  scatterAddRef: () => scatterAddRef,
  seededRandom: () => seededRandom2,
  siluFusedRef: () => siluFusedRef,
  siluGatedRef: () => siluGatedRef,
  siluInplaceRef: () => siluInplaceRef,
  siluRef: () => siluRef,
  softmaxInplaceRef: () => softmaxInplaceRef,
  softmaxRef: () => softmaxRef,
  softmaxTopkRef: () => softmaxTopkRef,
  softmaxWithTemp: () => softmaxWithTemp,
  topkArgmaxRef: () => topkArgmaxRef,
  topkRef: () => topkRef
});

// kernel-tests/src/reference/matmul.ts
function matmulRef(A, B, M, N, K, alpha = 1) {
  const C = new Float32Array(M * N);
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = sum * alpha;
    }
  }
  return C;
}
function batchMatmulRef(A, B, batch, M, N, K) {
  const C = new Float32Array(batch * M * N);
  const strideA = M * K;
  const strideB = K * N;
  const strideC = M * N;
  for (let b = 0; b < batch; b++) {
    for (let m = 0; m < M; m++) {
      for (let n = 0; n < N; n++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          sum += A[b * strideA + m * K + k] * B[b * strideB + k * N + n];
        }
        C[b * strideC + m * N + n] = sum;
      }
    }
  }
  return C;
}
function matvecRef(A, x, M, K) {
  const y = new Float32Array(M);
  for (let m = 0; m < M; m++) {
    let sum = 0;
    for (let k = 0; k < K; k++) {
      sum += A[m * K + k] * x[k];
    }
    y[m] = sum;
  }
  return y;
}

// kernel-tests/src/reference/softmax.ts
function softmaxRef(input, innerSize, outerSize, temperature = 1) {
  const output = new Float32Array(input.length);
  for (let row = 0; row < outerSize; row++) {
    const offset = row * innerSize;
    let maxVal = -Infinity;
    for (let i = 0; i < innerSize; i++) {
      maxVal = Math.max(maxVal, input[offset + i] / temperature);
    }
    let sum = 0;
    for (let i = 0; i < innerSize; i++) {
      const expVal = Math.exp(input[offset + i] / temperature - maxVal);
      output[offset + i] = expVal;
      sum += expVal;
    }
    for (let i = 0; i < innerSize; i++) {
      output[offset + i] /= sum;
    }
  }
  return output;
}
function logSoftmaxRef(input, innerSize, outerSize, temperature = 1) {
  const output = new Float32Array(input.length);
  for (let row = 0; row < outerSize; row++) {
    const offset = row * innerSize;
    let maxVal = -Infinity;
    for (let i = 0; i < innerSize; i++) {
      maxVal = Math.max(maxVal, input[offset + i] / temperature);
    }
    let logSum = 0;
    for (let i = 0; i < innerSize; i++) {
      logSum += Math.exp(input[offset + i] / temperature - maxVal);
    }
    logSum = Math.log(logSum);
    for (let i = 0; i < innerSize; i++) {
      output[offset + i] = input[offset + i] / temperature - maxVal - logSum;
    }
  }
  return output;
}
function softmaxInplaceRef(input, innerSize, outerSize, temperature = 1) {
  for (let row = 0; row < outerSize; row++) {
    const offset = row * innerSize;
    let maxVal = -Infinity;
    for (let i = 0; i < innerSize; i++) {
      maxVal = Math.max(maxVal, input[offset + i] / temperature);
    }
    let sum = 0;
    for (let i = 0; i < innerSize; i++) {
      input[offset + i] = Math.exp(input[offset + i] / temperature - maxVal);
      sum += input[offset + i];
    }
    for (let i = 0; i < innerSize; i++) {
      input[offset + i] /= sum;
    }
  }
  return input;
}

// kernel-tests/src/reference/silu.ts
function silu(x) {
  return x / (1 + Math.exp(-x));
}
function siluRef(input) {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    output[i] = silu(input[i]);
  }
  return output;
}
function siluGatedRef(gate, up) {
  const output = new Float32Array(gate.length);
  for (let i = 0; i < gate.length; i++) {
    output[i] = silu(gate[i]) * up[i];
  }
  return output;
}
function siluFusedRef(input) {
  const halfSize = input.length / 2;
  const output = new Float32Array(halfSize);
  for (let i = 0; i < halfSize; i++) {
    const gateVal = input[i];
    const upVal = input[halfSize + i];
    output[i] = silu(gateVal) * upVal;
  }
  return output;
}
function siluInplaceRef(input) {
  for (let i = 0; i < input.length; i++) {
    input[i] = silu(input[i]);
  }
  return input;
}

// kernel-tests/src/reference/rmsnorm.ts
function rmsNormRef(input, weight, batchSize, hiddenSize, eps = 1e-6) {
  const output = new Float32Array(input.length);
  for (let b = 0; b < batchSize; b++) {
    const offset = b * hiddenSize;
    let sumSq = 0;
    for (let i = 0; i < hiddenSize; i++) {
      const val = input[offset + i];
      sumSq += val * val;
    }
    const meanSq = sumSq / hiddenSize;
    const scale = 1 / Math.sqrt(meanSq + eps);
    for (let i = 0; i < hiddenSize; i++) {
      output[offset + i] = input[offset + i] * scale * weight[i];
    }
  }
  return output;
}
function rmsNormNoWeightRef(input, batchSize, hiddenSize, eps = 1e-6) {
  const output = new Float32Array(input.length);
  for (let b = 0; b < batchSize; b++) {
    const offset = b * hiddenSize;
    let sumSq = 0;
    for (let i = 0; i < hiddenSize; i++) {
      const val = input[offset + i];
      sumSq += val * val;
    }
    const scale = 1 / Math.sqrt(sumSq / hiddenSize + eps);
    for (let i = 0; i < hiddenSize; i++) {
      output[offset + i] = input[offset + i] * scale;
    }
  }
  return output;
}

// kernel-tests/src/reference/rope.ts
function computeRopeFreqs(dim, maxSeqLen, base = 1e4) {
  const halfDim = dim / 2;
  const cos = new Float32Array(maxSeqLen * halfDim);
  const sin = new Float32Array(maxSeqLen * halfDim);
  for (let pos = 0; pos < maxSeqLen; pos++) {
    for (let i = 0; i < halfDim; i++) {
      const freq = 1 / Math.pow(base, 2 * i / dim);
      const angle = pos * freq;
      cos[pos * halfDim + i] = Math.cos(angle);
      sin[pos * halfDim + i] = Math.sin(angle);
    }
  }
  return { cos, sin };
}
function ropeRef(x, cos, sin, seqLen, numHeads, headDim, startPos = 0) {
  const output = new Float32Array(x.length);
  const halfDim = headDim / 2;
  for (let s = 0; s < seqLen; s++) {
    const pos = s + startPos;
    for (let h = 0; h < numHeads; h++) {
      const offset = s * numHeads * headDim + h * headDim;
      for (let i = 0; i < halfDim; i++) {
        const x0 = x[offset + i];
        const x1 = x[offset + i + halfDim];
        const cosVal = cos[pos * halfDim + i];
        const sinVal = sin[pos * halfDim + i];
        output[offset + i] = x0 * cosVal - x1 * sinVal;
        output[offset + i + halfDim] = x0 * sinVal + x1 * cosVal;
      }
    }
  }
  return output;
}
function ropeInterleavedRef(x, cos, sin, seqLen, numHeads, headDim, startPos = 0) {
  const output = new Float32Array(x.length);
  const halfDim = headDim / 2;
  for (let s = 0; s < seqLen; s++) {
    const pos = s + startPos;
    for (let h = 0; h < numHeads; h++) {
      const offset = s * numHeads * headDim + h * headDim;
      for (let i = 0; i < halfDim; i++) {
        const x0 = x[offset + 2 * i];
        const x1 = x[offset + 2 * i + 1];
        const cosVal = cos[pos * halfDim + i];
        const sinVal = sin[pos * halfDim + i];
        output[offset + 2 * i] = x0 * cosVal - x1 * sinVal;
        output[offset + 2 * i + 1] = x0 * sinVal + x1 * cosVal;
      }
    }
  }
  return output;
}

// kernel-tests/src/reference/attention.ts
function attentionRef(Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim, mask = null) {
  const output = new Float32Array(seqLen * numHeads * headDim);
  const scale = 1 / Math.sqrt(headDim);
  const headsPerKV = numHeads / numKVHeads;
  for (let h = 0; h < numHeads; h++) {
    const kvHead = Math.floor(h / headsPerKV);
    for (let q = 0; q < seqLen; q++) {
      const scores = new Float32Array(kvLen);
      for (let k = 0; k < kvLen; k++) {
        let score = 0;
        for (let d = 0; d < headDim; d++) {
          const qIdx = q * numHeads * headDim + h * headDim + d;
          const kIdx = k * numKVHeads * headDim + kvHead * headDim + d;
          score += Q[qIdx] * K[kIdx];
        }
        scores[k] = score * scale;
        if (mask) {
          scores[k] += mask[q * kvLen + k];
        }
      }
      let maxScore = -Infinity;
      for (let k = 0; k < kvLen; k++) {
        maxScore = Math.max(maxScore, scores[k]);
      }
      let sumExp = 0;
      for (let k = 0; k < kvLen; k++) {
        scores[k] = Math.exp(scores[k] - maxScore);
        sumExp += scores[k];
      }
      for (let k = 0; k < kvLen; k++) {
        scores[k] /= sumExp;
      }
      for (let d = 0; d < headDim; d++) {
        let val = 0;
        for (let k = 0; k < kvLen; k++) {
          const vIdx = k * numKVHeads * headDim + kvHead * headDim + d;
          val += scores[k] * V[vIdx];
        }
        output[q * numHeads * headDim + h * headDim + d] = val;
      }
    }
  }
  return output;
}
function createCausalMask(seqLen, kvLen = null) {
  if (kvLen === null)
    kvLen = seqLen;
  const mask = new Float32Array(seqLen * kvLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < kvLen; j++) {
      const offset = kvLen - seqLen;
      mask[i * kvLen + j] = j <= i + offset ? 0 : -Infinity;
    }
  }
  return mask;
}
function flashAttentionRef(Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim, blockSize = 64) {
  return attentionRef(Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim, createCausalMask(seqLen, kvLen));
}
function mqaRef(Q, K, V, seqLen, kvLen, numHeads, headDim, mask = null) {
  return attentionRef(Q, K, V, seqLen, kvLen, numHeads, 1, headDim, mask);
}

// kernel-tests/src/reference/topk.ts
function topkRef(probs, numTokens, numExperts, topK, normalize = true) {
  const indices = new Uint32Array(numTokens * topK);
  const weights = new Float32Array(numTokens * topK);
  for (let token = 0; token < numTokens; token++) {
    const offset = token * numExperts;
    const pairs = [];
    for (let i = 0; i < numExperts; i++) {
      pairs.push({ prob: probs[offset + i], idx: i });
    }
    pairs.sort((a, b) => b.prob - a.prob);
    let weightSum = 0;
    for (let k = 0; k < topK; k++) {
      indices[token * topK + k] = pairs[k].idx;
      weights[token * topK + k] = pairs[k].prob;
      weightSum += pairs[k].prob;
    }
    if (normalize && weightSum > 0) {
      for (let k = 0; k < topK; k++) {
        weights[token * topK + k] /= weightSum;
      }
    }
  }
  return { indices, weights };
}
function softmaxTopkRef(logits, numTokens, numExperts, topK, normalize = true) {
  const indices = new Uint32Array(numTokens * topK);
  const weights = new Float32Array(numTokens * topK);
  for (let token = 0; token < numTokens; token++) {
    const offset = token * numExperts;
    let maxVal = -Infinity;
    for (let i = 0; i < numExperts; i++) {
      maxVal = Math.max(maxVal, logits[offset + i]);
    }
    const expVals = new Float32Array(numExperts);
    let expSum = 0;
    for (let i = 0; i < numExperts; i++) {
      expVals[i] = Math.exp(logits[offset + i] - maxVal);
      expSum += expVals[i];
    }
    const pairs = [];
    for (let i = 0; i < numExperts; i++) {
      pairs.push({ prob: expVals[i] / expSum, idx: i });
    }
    pairs.sort((a, b) => b.prob - a.prob);
    let weightSum = 0;
    for (let k = 0; k < topK; k++) {
      indices[token * topK + k] = pairs[k].idx;
      weights[token * topK + k] = pairs[k].prob;
      weightSum += pairs[k].prob;
    }
    if (normalize && weightSum > 0) {
      for (let k = 0; k < topK; k++) {
        weights[token * topK + k] /= weightSum;
      }
    }
  }
  return { indices, weights };
}

// kernel-tests/src/reference/scatter-add.ts
function scatterAddRef(expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK) {
  const output = new Float32Array(numTokens * hiddenSize);
  for (let token = 0; token < numTokens; token++) {
    for (let dim = 0; dim < hiddenSize; dim++) {
      let sum = 0;
      for (let k = 0; k < topK; k++) {
        const expertIdx = indices[token * topK + k];
        const weight = weights[token * topK + k];
        const expertOffset = expertIdx * numTokens * hiddenSize + token * hiddenSize + dim;
        sum += weight * expertOutputs[expertOffset];
      }
      output[token * hiddenSize + dim] = sum;
    }
  }
  return output;
}
function scatterAddAccumulateRef(expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK, existingOutput) {
  const output = new Float32Array(existingOutput);
  for (let token = 0; token < numTokens; token++) {
    for (let dim = 0; dim < hiddenSize; dim++) {
      let sum = 0;
      for (let k = 0; k < topK; k++) {
        const expertIdx = indices[token * topK + k];
        const weight = weights[token * topK + k];
        const expertOffset = expertIdx * numTokens * hiddenSize + token * hiddenSize + dim;
        sum += weight * expertOutputs[expertOffset];
      }
      output[token * hiddenSize + dim] += sum;
    }
  }
  return output;
}

// kernel-tests/src/reference/moe-gather.ts
function moeGatherRef(tokens, expertIndices, numTokens, hiddenSize, numExperts, topK) {
  const tokenCounts = new Uint32Array(numExperts);
  for (let t = 0; t < numTokens; t++) {
    for (let k = 0; k < topK; k++) {
      const expertIdx = expertIndices[t * topK + k];
      tokenCounts[expertIdx]++;
    }
  }
  let maxTokensPerExpert = 0;
  for (let e = 0; e < numExperts; e++) {
    maxTokensPerExpert = Math.max(maxTokensPerExpert, tokenCounts[e]);
  }
  const gatheredTokens = new Float32Array(numExperts * maxTokensPerExpert * hiddenSize);
  const tokenIndices = new Uint32Array(numExperts * maxTokensPerExpert);
  tokenIndices.fill(4294967295);
  const currentCounts = new Uint32Array(numExperts);
  for (let t = 0; t < numTokens; t++) {
    for (let k = 0; k < topK; k++) {
      const expertIdx = expertIndices[t * topK + k];
      const slotIdx = currentCounts[expertIdx];
      const srcOffset = t * hiddenSize;
      const dstOffset = expertIdx * maxTokensPerExpert * hiddenSize + slotIdx * hiddenSize;
      for (let d = 0; d < hiddenSize; d++) {
        gatheredTokens[dstOffset + d] = tokens[srcOffset + d];
      }
      tokenIndices[expertIdx * maxTokensPerExpert + slotIdx] = t;
      currentCounts[expertIdx]++;
    }
  }
  return {
    gatheredTokens,
    tokenCounts,
    tokenIndices,
    maxTokensPerExpert
  };
}
function moeComputeAssignmentsRef(expertIndices, numTokens, numExperts, topK) {
  const tokenCounts = new Uint32Array(numExperts);
  const expertOffsets = new Uint32Array(numExperts);
  for (let t = 0; t < numTokens; t++) {
    for (let k = 0; k < topK; k++) {
      const expertIdx = expertIndices[t * topK + k];
      tokenCounts[expertIdx]++;
    }
  }
  let offset = 0;
  for (let e = 0; e < numExperts; e++) {
    expertOffsets[e] = offset;
    offset += tokenCounts[e];
  }
  return { tokenCounts, expertOffsets, totalAssignments: offset };
}

// kernel-tests/src/reference/gather.ts
function gatherRef(embeddings, indices, vocabSize, embedDim) {
  const seqLen = indices.length;
  const output = new Float32Array(seqLen * embedDim);
  for (let i = 0; i < seqLen; i++) {
    const idx = indices[i];
    const srcOffset = idx * embedDim;
    const dstOffset = i * embedDim;
    for (let d = 0; d < embedDim; d++) {
      output[dstOffset + d] = embeddings[srcOffset + d];
    }
  }
  return output;
}
function batchGatherRef(embeddings, indices, batchSize, seqLen, embedDim) {
  const output = new Float32Array(batchSize * seqLen * embedDim);
  for (let b = 0; b < batchSize; b++) {
    for (let s = 0; s < seqLen; s++) {
      const idx = indices[b * seqLen + s];
      const srcOffset = idx * embedDim;
      const dstOffset = (b * seqLen + s) * embedDim;
      for (let d = 0; d < embedDim; d++) {
        output[dstOffset + d] = embeddings[srcOffset + d];
      }
    }
  }
  return output;
}
function gatherWithPosRef(embeddings, posEmbeddings, indices, vocabSize, embedDim, startPos = 0) {
  const seqLen = indices.length;
  const output = new Float32Array(seqLen * embedDim);
  for (let i = 0; i < seqLen; i++) {
    const tokenIdx = indices[i];
    const posIdx = i + startPos;
    const tokenOffset = tokenIdx * embedDim;
    const posOffset = posIdx * embedDim;
    const dstOffset = i * embedDim;
    for (let d = 0; d < embedDim; d++) {
      output[dstOffset + d] = embeddings[tokenOffset + d] + posEmbeddings[posOffset + d];
    }
  }
  return output;
}

// kernel-tests/src/reference/residual.ts
function residualAddRef(x, residual) {
  const output = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    output[i] = x[i] + residual[i];
  }
  return output;
}
function residualAddInplaceRef(x, residual) {
  for (let i = 0; i < x.length; i++) {
    x[i] += residual[i];
  }
  return x;
}
function scaledResidualAddRef(x, residual, scale) {
  const output = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    output[i] = x[i] + scale * residual[i];
  }
  return output;
}

// kernel-tests/src/reference/dequant.ts
function float16ToFloat32(bits) {
  const sign = bits >> 15 & 1;
  const exp = bits >> 10 & 31;
  const frac = bits & 1023;
  if (exp === 0) {
    if (frac === 0)
      return sign ? -0 : 0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  }
  if (exp === 31) {
    return frac ? NaN : sign ? -Infinity : Infinity;
  }
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}
function float32ToFloat16(value) {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);
  floatView[0] = value;
  const f = int32View[0];
  const sign = f >> 31 & 1;
  let exp = f >> 23 & 255;
  let frac = f & 8388607;
  if (exp === 255) {
    return sign << 15 | 31744 | (frac ? 512 : 0);
  }
  if (exp === 0) {
    return sign << 15;
  }
  exp = exp - 127 + 15;
  if (exp >= 31) {
    return sign << 15 | 31744;
  }
  if (exp <= 0) {
    if (exp < -10) {
      return sign << 15;
    }
    frac = (frac | 8388608) >> 1 - exp;
    return sign << 15 | frac >> 13;
  }
  return sign << 15 | exp << 10 | frac >> 13;
}
function dequantInt8Ref(quantized, scales, zeroPoints = null, numChannels = 1, channelSize = 0) {
  const output = new Float32Array(quantized.length);
  if (channelSize === 0) {
    channelSize = quantized.length / numChannels;
  }
  for (let c = 0; c < numChannels; c++) {
    const scale = scales[c];
    const zp = zeroPoints ? zeroPoints[c] : 0;
    for (let i = 0; i < channelSize; i++) {
      const idx = c * channelSize + i;
      output[idx] = (quantized[idx] - zp) * scale;
    }
  }
  return output;
}
function dequantInt4Ref(quantized, scales, numElements, groupSize = 32) {
  const output = new Float32Array(numElements);
  const numGroups = Math.ceil(numElements / groupSize);
  for (let i = 0; i < numElements; i++) {
    const byteIdx = Math.floor(i / 2);
    const groupIdx = Math.floor(i / groupSize);
    const scale = scales[groupIdx];
    let val;
    if (i % 2 === 0) {
      val = quantized[byteIdx] & 15;
    } else {
      val = quantized[byteIdx] >> 4 & 15;
    }
    if (val >= 8) {
      val = val - 16;
    }
    output[i] = val * scale;
  }
  return output;
}
function dequantQ4_0Ref(quantized, numBlocks) {
  const blockSize = 32;
  const output = new Float32Array(numBlocks * blockSize);
  const dataView = new DataView(quantized.buffer);
  for (let block = 0; block < numBlocks; block++) {
    const blockOffset = block * 18;
    const scaleBytes = dataView.getUint16(blockOffset, true);
    const scale = float16ToFloat32(scaleBytes);
    for (let i = 0; i < 16; i++) {
      const byte = quantized[blockOffset + 2 + i];
      const low = (byte & 15) - 8;
      const high = (byte >> 4 & 15) - 8;
      output[block * blockSize + i * 2] = low * scale;
      output[block * blockSize + i * 2 + 1] = high * scale;
    }
  }
  return output;
}
var Q4K_K = 256;
var Q4K_BLOCK_SIZE = 144;
function findMinMax(data, offset, length) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < length; i++) {
    const val = data[offset + i];
    if (val < min)
      min = val;
    if (val > max)
      max = val;
  }
  return { min, max };
}
function quantizeQ4_KBlockRef(data, offset) {
  const block = new Uint8Array(Q4K_BLOCK_SIZE);
  const view = new DataView(block.buffer);
  const scales = new Float32Array(8);
  const minOffsets = new Float32Array(8);
  const qs = new Uint8Array(256);
  for (let sb = 0; sb < 8; sb++) {
    const sbOffset = offset + sb * 32;
    const { min, max } = findMinMax(data, sbOffset, 32);
    minOffsets[sb] = -min;
    const range = max - min;
    scales[sb] = range > 0 ? range / 15 : 0;
    const invScale = scales[sb] > 0 ? 1 / scales[sb] : 0;
    for (let i = 0; i < 32; i++) {
      const val = data[sbOffset + i];
      let q = Math.round((val - min) * invScale);
      q = Math.max(0, Math.min(15, q));
      qs[sb * 32 + i] = q;
    }
  }
  let maxScale = 0;
  let maxMinOffset = 0;
  for (let i = 0; i < 8; i++) {
    if (scales[i] > maxScale)
      maxScale = scales[i];
    if (minOffsets[i] > maxMinOffset)
      maxMinOffset = minOffsets[i];
    if (minOffsets[i] < 0)
      minOffsets[i] = 0;
  }
  const d = maxScale / 63;
  const dmin = maxMinOffset / 63;
  view.setUint16(0, float32ToFloat16(d), true);
  view.setUint16(2, float32ToFloat16(dmin), true);
  const invD = d > 0 ? 1 / d : 0;
  const invDmin = dmin > 0 ? 1 / dmin : 0;
  const scaleBits = new Uint8Array(8);
  const minBits = new Uint8Array(8);
  for (let i = 0; i < 8; i++) {
    scaleBits[i] = Math.min(63, Math.round(scales[i] * invD));
    minBits[i] = Math.min(63, Math.round(Math.max(0, minOffsets[i]) * invDmin));
  }
  for (let i = 0; i < 4; i++) {
    const scaleLo = scaleBits[i] & 63;
    const scaleHi2 = scaleBits[i + 4] >> 4 & 3;
    block[4 + i] = scaleLo | scaleHi2 << 6;
  }
  for (let i = 0; i < 4; i++) {
    const minLo = minBits[i] & 63;
    const minHi2 = minBits[i + 4] >> 4 & 3;
    block[4 + 4 + i] = minLo | minHi2 << 6;
  }
  for (let i = 0; i < 4; i++) {
    const scaleLo4 = scaleBits[i + 4] & 15;
    const minLo4 = minBits[i + 4] & 15;
    block[4 + 8 + i] = scaleLo4 | minLo4 << 4;
  }
  for (let chunk = 0; chunk < 4; chunk++) {
    const chunkBase = chunk * 64;
    const byteBase = 16 + chunk * 32;
    for (let i = 0; i < 32; i++) {
      const lo = qs[chunkBase + i] & 15;
      const hi = qs[chunkBase + 32 + i] & 15;
      block[byteBase + i] = lo | hi << 4;
    }
  }
  return block;
}
function quantizeQ4_KRef(values, numBlocks) {
  const out = new Uint8Array(numBlocks * Q4K_BLOCK_SIZE);
  for (let b = 0; b < numBlocks; b++) {
    const block = quantizeQ4_KBlockRef(values, b * Q4K_K);
    out.set(block, b * Q4K_BLOCK_SIZE);
  }
  return out;
}
function dequantizeQ4_KBlockRef(block) {
  const view = new DataView(block.buffer, block.byteOffset);
  const out = new Float32Array(Q4K_K);
  const d = float16ToFloat32(view.getUint16(0, true));
  const dmin = float16ToFloat32(view.getUint16(2, true));
  const scaleBits = new Uint8Array(8);
  const minBits = new Uint8Array(8);
  for (let i = 0; i < 4; i++) {
    scaleBits[i] = block[4 + i] & 63;
    scaleBits[i + 4] = (block[4 + i] >> 6 & 3) << 4;
  }
  for (let i = 0; i < 4; i++) {
    minBits[i] = block[4 + 4 + i] & 63;
    minBits[i + 4] = (block[4 + 4 + i] >> 6 & 3) << 4;
  }
  for (let i = 0; i < 4; i++) {
    scaleBits[i + 4] |= block[4 + 8 + i] & 15;
    minBits[i + 4] |= block[4 + 8 + i] >> 4 & 15;
  }
  const scales = new Float32Array(8);
  const mins = new Float32Array(8);
  for (let i = 0; i < 8; i++) {
    scales[i] = d * scaleBits[i];
    mins[i] = dmin * minBits[i];
  }
  for (let chunk = 0; chunk < 4; chunk++) {
    const chunkBase = chunk * 64;
    const byteBase = 16 + chunk * 32;
    for (let i = 0; i < 32; i++) {
      const byte = block[byteBase + i];
      const lo = byte & 15;
      const hi = byte >> 4 & 15;
      const sb0 = Math.floor((chunkBase + i) / 32);
      const sb1 = Math.floor((chunkBase + 32 + i) / 32);
      out[chunkBase + i] = scales[sb0] * lo - mins[sb0];
      out[chunkBase + 32 + i] = scales[sb1] * hi - mins[sb1];
    }
  }
  return out;
}
function dequantQ4_KRef(quantized, numBlocks) {
  const out = new Float32Array(numBlocks * Q4K_K);
  for (let b = 0; b < numBlocks; b++) {
    const start = b * Q4K_BLOCK_SIZE;
    const block = quantized.subarray(start, start + Q4K_BLOCK_SIZE);
    out.set(dequantizeQ4_KBlockRef(block), b * Q4K_K);
  }
  return out;
}

// kernel-tests/src/reference/sample.ts
function argmaxRef(logits) {
  let maxIdx = 0;
  let maxVal = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > maxVal) {
      maxVal = logits[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}
function topkArgmaxRef(logits, k) {
  const indexed = Array.from(logits).map((val, idx) => ({ val, idx }));
  indexed.sort((a, b) => b.val - a.val);
  const topK = indexed.slice(0, k);
  return {
    indices: topK.map((x) => x.idx),
    values: topK.map((x) => x.val)
  };
}
function softmaxWithTemp(logits, temperature) {
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    scaled[i] = logits[i] / temperature;
  }
  let max = scaled[0];
  for (let i = 1; i < scaled.length; i++) {
    if (scaled[i] > max)
      max = scaled[i];
  }
  let sum = 0;
  for (let i = 0; i < scaled.length; i++) {
    scaled[i] = Math.exp(scaled[i] - max);
    sum += scaled[i];
  }
  for (let i = 0; i < scaled.length; i++) {
    scaled[i] /= sum;
  }
  return scaled;
}
function sampleTopKRef(logits, temperature, topK, randomValue) {
  if (temperature < 0.01) {
    return argmaxRef(logits);
  }
  const { indices, values } = topkArgmaxRef(logits, topK);
  const scaledValues = values.map((v) => v / temperature);
  const max = Math.max(...scaledValues);
  const expValues = scaledValues.map((v) => Math.exp(v - max));
  const sum = expValues.reduce((a, b) => a + b, 0);
  const probs = expValues.map((v) => v / sum);
  let cumProb = 0;
  for (let i = 0; i < probs.length; i++) {
    cumProb += probs[i];
    if (cumProb >= randomValue) {
      return indices[i];
    }
  }
  return indices[indices.length - 1];
}
function seededRandom2(seed) {
  const x = Math.sin(seed) * 1e4;
  return x - Math.floor(x);
}

// kernel-tests/src/harness/tolerance.ts
var KERNEL_TOLERANCES = {
  matmul_f32: { rtol: 1e-5, atol: 1e-6 },
  matmul_f16: { rtol: 0.01, atol: 1e-3 },
  // FP16 has ~3 decimal digits
  attention: { rtol: 1e-4, atol: 1e-5 },
  // Softmax accumulation
  softmax: { rtol: 1e-5, atol: 1e-7 },
  // Must sum to 1
  rmsnorm: { rtol: 1e-5, atol: 1e-6 },
  rope: { rtol: 1e-5, atol: 1e-6 },
  // Sin/cos operations
  silu: { rtol: 1e-5, atol: 1e-6 },
  topk: {
    indices: { exact: true },
    // Indices must match exactly
    weights: { rtol: 1e-5, atol: 1e-7 }
  },
  scatter_add: { rtol: 1e-5, atol: 1e-6 },
  moe_gather: { rtol: 1e-5, atol: 1e-6 },
  gather: { exact: true },
  // Embedding lookup is exact
  residual: { rtol: 1e-6, atol: 1e-8 },
  // Simple addition
  dequant: { rtol: 1e-4, atol: 1e-5 }
  // Quantization introduces error
};
function compareArrays(expected, actual, options = {}) {
  const { rtol = 1e-5, atol = 1e-8 } = options;
  if (expected.length !== actual.length) {
    return {
      passed: false,
      error: `Length mismatch: expected ${expected.length}, got ${actual.length}`,
      maxError: Infinity,
      avgError: Infinity,
      mismatchCount: expected.length
    };
  }
  let maxError = 0;
  let sumError = 0;
  let mismatchCount = 0;
  const mismatches = [];
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i];
    const a = actual[i];
    const error = Math.abs(e - a);
    const threshold = atol + rtol * Math.abs(e);
    maxError = Math.max(maxError, error);
    sumError += error;
    if (error > threshold) {
      mismatchCount++;
      if (mismatches.length < 10) {
        mismatches.push({ index: i, expected: e, actual: a, error, threshold });
      }
    }
  }
  return {
    passed: mismatchCount === 0,
    maxError,
    avgError: sumError / expected.length,
    mismatchCount,
    mismatchRatio: mismatchCount / expected.length,
    firstMismatches: mismatches
  };
}
function generateTestData(size, seed = 42, options = {}) {
  const { min = -1, max = 1, dtype = "float32" } = options;
  let data;
  switch (dtype) {
    case "uint32":
      data = new Uint32Array(size);
      break;
    case "int32":
      data = new Int32Array(size);
      break;
    default:
      data = new Float32Array(size);
  }
  let state = seed;
  const range = max - min;
  for (let i = 0; i < size; i++) {
    state = state * 1103515245 + 12345 & 2147483647;
    const normalized = state / 2147483647;
    if (dtype === "float32") {
      data[i] = min + normalized * range;
    } else {
      data[i] = Math.floor(min + normalized * range);
    }
  }
  return data;
}

// kernel-tests/browser/test-page.ts
var {
  runMatmul: runMatmul2 = null,
  runSoftmax: runSoftmax2 = null,
  runTopK: runTopK2 = null,
  runSoftmaxTopK: runSoftmaxTopK2 = null,
  runScatterAdd: runScatterAdd2 = null,
  runMoEGather: runMoEGather2 = null,
  runRMSNorm: runRMSNorm2 = null,
  runRoPE: runRoPE2 = null,
  runSiLU: runSiLU2 = null,
  runSwiGLURowsplitBias: runSwiGLURowsplitBias2 = null,
  runScale: runScale2 = null,
  runGather: runGather2 = null,
  runResidualAdd: runResidualAdd2 = null,
  runBiasAdd: runBiasAdd2 = null,
  runAttention: runAttention2 = null,
  dequantize: dequantize2 = null,
  dequantizeQ6K: dequantizeQ6K2 = null,
  runBF16ToF32: runBF16ToF322 = null,
  runBF16ToF16: runBF16ToF162 = null,
  castF32ToF16: castF32ToF162 = null
} = kernel_selector_exports;
var bufferPool = null;
try {
  bufferPool = await Promise.resolve().then(() => (init_buffer_pool(), buffer_pool_exports));
} catch (e) {
  console.warn("Buffer pool not available:", e.message);
}
var device = null;
var initialized = false;
function f16ToF322(h) {
  const sign = (h & 32768) >> 15;
  const exponent = (h & 31744) >> 10;
  const mantissa = h & 1023;
  if (exponent === 0) {
    if (mantissa === 0)
      return sign ? -0 : 0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mantissa / 1024);
  } else if (exponent === 31) {
    return mantissa === 0 ? sign ? -Infinity : Infinity : NaN;
  }
  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
}
async function initGPU() {
  if (device)
    return device;
  setPlatformsBaseUrl("/config/platforms/");
  setRegistryUrl("/config/kernels/registry.json");
  device = await initDevice();
  if (!device) {
    throw new Error("WebGPU not available");
  }
  setActiveKernelPath(resolveKernelPath("q4k-fused"), "runtime");
  initialized = true;
  return device;
}
async function getGPU() {
  if (!device) {
    await initGPU();
  }
  return { device, queue: device.queue };
}
function makeBuffer(data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) {
  const byteLength = data instanceof ArrayBuffer ? data.byteLength : data.byteLength;
  const buffer = device.createBuffer({
    size: byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true
  });
  const mappedRange = buffer.getMappedRange();
  if (data instanceof Float32Array) {
    new Float32Array(mappedRange).set(data);
  } else if (data instanceof Uint32Array) {
    new Uint32Array(mappedRange).set(data);
  } else if (data instanceof Int32Array) {
    new Int32Array(mappedRange).set(data);
  } else if (data instanceof Uint16Array) {
    new Uint16Array(mappedRange).set(data);
  } else if (data instanceof Uint8Array) {
    new Uint8Array(mappedRange).set(data);
  } else {
    new Uint8Array(mappedRange).set(new Uint8Array(data));
  }
  buffer.unmap();
  return buffer;
}
async function readBufferData(buffer, size) {
  const stagingBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
  device.queue.submit([encoder.finish()]);
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const data = new Uint8Array(stagingBuffer.getMappedRange()).slice();
  stagingBuffer.unmap();
  stagingBuffer.destroy();
  return data.buffer;
}
var testHarness = {
  // Core
  getGPU,
  device: () => device,
  // Reference implementations
  references: reference_exports,
  softmax: softmaxRef,
  topkRef,
  softmaxTopkRef,
  matmulRef,
  scatterAddRef,
  // Utilities
  generateTestData,
  compareArrays,
  makeBuffer,
  readBufferData,
  KERNEL_TOLERANCES,
  // ============================================================================
  // Kernel Runners (match expected interface from tests)
  // ============================================================================
  /**
   * Run matmul kernel
   */
  async runMatmul(dev, A, B, M, N, K, alpha = 1) {
    if (!runMatmul2) {
      return matmulRef(A, B, M, N, K, alpha);
    }
    const bufA = makeBuffer(A);
    const tensorA = createTensor(bufA, "f32", [M, K], "matmul_a");
    const bufB = makeBuffer(B);
    const resultTensor = await runMatmul2(tensorA, bufB, M, N, K, { alpha, transposeB: false });
    const result = new Float32Array(await readBufferData(resultTensor.buffer, M * N * 4));
    bufA.destroy();
    bufB.destroy();
    resultTensor.buffer.destroy();
    return result;
  },
  /**
   * Run batched matmul kernel
   */
  async runBatchMatmul(dev, A, B, batch, M, N, K) {
    return batchMatmulRef(A, B, batch, M, N, K);
  },
  /**
   * Run matrix-vector multiplication
   */
  async runMatvec(dev, A, x, M, K) {
    return matvecRef(A, x, M, K);
  },
  /**
   * Run Q4_K fused matmul kernel (tests q4_fused/q4_fused_batched)
   * C = A[M,K] @ dequant(B_q4k[N,K])^T = C[M,N]
   */
  async runMatmulQ4K(dev, A, B_q4k, M, N, K, alpha = 1) {
    if (!runMatmul2) {
      throw new Error("runMatmul kernel not available");
    }
    const bufA = makeBuffer(A);
    const tensorA = createTensor(bufA, "f32", [M, K], "matmul_q4k_a");
    const bufB = makeBuffer(B_q4k, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const resultTensor = await runMatmul2(tensorA, bufB, M, N, K, { alpha, bDtype: "q4k" });
    const result = new Float32Array(await readBufferData(resultTensor.buffer, M * N * 4));
    bufA.destroy();
    bufB.destroy();
    resultTensor.buffer.destroy();
    return result;
  },
  /**
   * Run softmax kernel
   */
  async runSoftmax(dev, input, innerSize, outerSize, temperature = 1) {
    if (!runSoftmax2) {
      return softmaxRef(input, innerSize, outerSize, temperature);
    }
    const inputBuf = makeBuffer(input);
    const inputTensor = createTensor(inputBuf, "f32", [outerSize, innerSize], "softmax_input");
    const resultTensor = await runSoftmax2(inputTensor, -1, {
      batchSize: outerSize,
      size: innerSize,
      temperature
    });
    const result = new Float32Array(await readBufferData(resultTensor.buffer, input.length * 4));
    inputBuf.destroy();
    resultTensor.buffer.destroy();
    return result;
  },
  /**
   * Run fused softmax + top-k kernel
   */
  async runSoftmaxTopK(dev, logits, numTokens, numExperts, topK, options = {}) {
    if (!runSoftmaxTopK2) {
      return softmaxTopkRef(logits, numTokens, numExperts, topK, options.normalize !== false);
    }
    const inputBuf = makeBuffer(logits);
    const { indices: indicesBuf, weights: weightsBuf } = await runSoftmaxTopK2(
      inputBuf,
      numTokens,
      numExperts,
      topK,
      { normalize: options.normalize !== false }
    );
    const indices = new Uint32Array(await readBufferData(indicesBuf, numTokens * topK * 4));
    const weights = new Float32Array(await readBufferData(weightsBuf, numTokens * topK * 4));
    inputBuf.destroy();
    indicesBuf.destroy();
    weightsBuf.destroy();
    return { indices, weights };
  },
  /**
   * Run top-k selection (without softmax)
   */
  async runTopK(dev, probs, numTokens, numExperts, topK, options = {}) {
    const inputBuf = makeBuffer(probs);
    const { indices: indicesBuf, weights: weightsBuf } = await runTopK2(
      inputBuf,
      numTokens,
      numExperts,
      topK,
      { normalize: options.normalize !== false }
    );
    const indices = new Uint32Array(await readBufferData(indicesBuf, numTokens * topK * 4));
    const weights = new Float32Array(await readBufferData(weightsBuf, numTokens * topK * 4));
    inputBuf.destroy();
    indicesBuf.destroy();
    weightsBuf.destroy();
    return { indices, weights };
  },
  /**
   * Run scatter-add kernel
   */
  async runScatterAdd(dev, expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK) {
    if (!runScatterAdd2) {
      return scatterAddRef(expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK);
    }
    const expertBuf = makeBuffer(expertOutputs);
    const indicesBuf = makeBuffer(indices);
    const weightsBuf = makeBuffer(weights);
    const expertTensor = createTensor(expertBuf, "f32", [numExperts, numTokens, hiddenSize], "expert_outputs");
    const resultTensor = await runScatterAdd2(
      expertTensor,
      indicesBuf,
      weightsBuf,
      numTokens,
      hiddenSize,
      numExperts,
      topK
    );
    const result = new Float32Array(await readBufferData(resultTensor.buffer, numTokens * hiddenSize * 4));
    expertBuf.destroy();
    indicesBuf.destroy();
    weightsBuf.destroy();
    resultTensor.buffer.destroy();
    return result;
  },
  /**
   * Run RMSNorm kernel
   * kernel-selector API: runRMSNorm(input, weight, eps, options)
   * options: { batchSize, hiddenSize }
   */
  async runRMSNorm(dev, input, weight, numTokens, hiddenSize, eps = 1e-6) {
    if (!runRMSNorm2) {
      return rmsNormRef(input, weight, numTokens, hiddenSize, eps);
    }
    const inputBuf = makeBuffer(input);
    const weightBuf = makeBuffer(weight);
    const inputTensor = createTensor(inputBuf, "f32", [numTokens, hiddenSize], "rmsnorm_input");
    const resultTensor = await runRMSNorm2(inputTensor, weightBuf, eps, {
      batchSize: numTokens,
      hiddenSize
    });
    let result;
    if (resultTensor.dtype === "f16") {
      const rawData = await readBufferData(resultTensor.buffer, numTokens * hiddenSize * 2);
      const u16View = new Uint16Array(rawData);
      result = new Float32Array(u16View.length);
      for (let i = 0; i < u16View.length; i++) {
        result[i] = f16ToF322(u16View[i]);
      }
    } else {
      result = new Float32Array(
        await readBufferData(resultTensor.buffer, numTokens * hiddenSize * 4)
      );
    }
    inputBuf.destroy();
    weightBuf.destroy();
    resultTensor.buffer.destroy();
    return result;
  },
  /**
   * Run RoPE kernel
   */
  async runRoPE(dev, input, seqLen, numHeads, headDim, startPos = 0) {
    const { cos, sin } = computeRopeFreqs(headDim, seqLen + startPos);
    if (!runRoPE2) {
      return ropeRef(input, cos, sin, seqLen, numHeads, headDim, startPos);
    }
    const inputBuf = makeBuffer(input);
    const cosBuf = makeBuffer(cos);
    const sinBuf = makeBuffer(sin);
    await runRoPE2(inputBuf, cosBuf, sinBuf, seqLen, {
      numHeads,
      headDim,
      startPos
    });
    const result = new Float32Array(
      await readBufferData(inputBuf, seqLen * numHeads * headDim * 4)
    );
    inputBuf.destroy();
    cosBuf.destroy();
    sinBuf.destroy();
    return result;
  },
  /**
   * Run SiLU kernel
   */
  async runSiLU(dev, input) {
    if (!runSiLU2) {
      return siluRef(input);
    }
    const inputBuf = makeBuffer(input);
    const resultBuf = await runSiLU2(inputBuf, { size: input.length });
    const result = new Float32Array(await readBufferData(resultBuf, input.length * 4));
    inputBuf.destroy();
    resultBuf.destroy();
    return result;
  },
  /**
   * Run SiLU with gating
   */
  async runSiLUGated(dev, gate, up) {
    if (!runSiLU2) {
      return siluGatedRef(gate, up);
    }
    const gateBuf = makeBuffer(gate);
    const upBuf = makeBuffer(up);
    const resultBuf = await runSiLU2(upBuf, { size: up.length, gate: gateBuf });
    const result = new Float32Array(await readBufferData(resultBuf, up.length * 4));
    gateBuf.destroy();
    upBuf.destroy();
    resultBuf.destroy();
    return result;
  },
  /**
   * Run gather/embedding lookup
   * kernel-selector API: runGather(indices, embeddings, numTokens, hiddenSize, vocabSize, options)
   */
  async runGather(dev, embeddings, indices, vocabSize, embedDim) {
    if (!runGather2) {
      return gatherRef(embeddings, indices, vocabSize, embedDim);
    }
    const embBuf = makeBuffer(embeddings);
    const idxBuf = makeBuffer(indices);
    const numTokens = indices.length;
    const resultTensor = await runGather2(idxBuf, embBuf, numTokens, embedDim, vocabSize, { transpose: false });
    let result;
    if (resultTensor.dtype === "f16") {
      const rawData = await readBufferData(resultTensor.buffer, numTokens * embedDim * 2);
      const u16View = new Uint16Array(rawData);
      result = new Float32Array(u16View.length);
      for (let i = 0; i < u16View.length; i++) {
        result[i] = f16ToF322(u16View[i]);
      }
    } else {
      result = new Float32Array(
        await readBufferData(resultTensor.buffer, numTokens * embedDim * 4)
      );
    }
    embBuf.destroy();
    idxBuf.destroy();
    resultTensor.buffer.destroy();
    return result;
  },
  /**
   * Run residual add
   * kernel-selector API: runResidualAdd(a, b, size, options)
   */
  async runResidual(dev, x, residual) {
    if (!runResidualAdd2) {
      return residualAddRef(x, residual);
    }
    const xBuf = makeBuffer(x);
    const resBuf = makeBuffer(residual);
    const size = x.length;
    const resultBuf = await runResidualAdd2(xBuf, resBuf, size);
    const result = new Float32Array(await readBufferData(resultBuf, size * 4));
    xBuf.destroy();
    resBuf.destroy();
    resultBuf.destroy();
    return result;
  },
  /**
   * Run bias add (row-wise)
   */
  async runBiasAdd(dev, data, bias, numTokens, dim) {
    if (!runBiasAdd2) {
      const result2 = new Float32Array(data);
      for (let t = 0; t < numTokens; t++) {
        const rowOffset = t * dim;
        for (let d = 0; d < dim; d++) {
          result2[rowOffset + d] += bias[d];
        }
      }
      return result2;
    }
    const dataBuf = makeBuffer(data);
    const biasBuf = makeBuffer(bias);
    const dataTensor = createTensor(dataBuf, "f32", [numTokens, dim], "bias_add_data");
    const biasTensor = createTensor(biasBuf, "f32", [dim], "bias_add_bias");
    const resultTensor = await runBiasAdd2(dataTensor, biasTensor, numTokens, dim);
    const result = new Float32Array(await readBufferData(resultTensor.buffer, numTokens * dim * 4));
    dataBuf.destroy();
    biasBuf.destroy();
    if (resultTensor.buffer !== dataBuf) {
      resultTensor.buffer.destroy();
    }
    return result;
  },
  /**
   * Run Q4_K dequantization (Q4_K_M) on GPU
   * kernel-selector API: dequantize(quantized, numBlocks, options)
   */
  async runDequantQ4K(dev, quantized, numBlocks) {
    if (!dequantize2) {
      throw new Error("dequantize kernel not available");
    }
    const qBuf = makeBuffer(quantized, GPUBufferUsage.STORAGE);
    const outTensor = await dequantize2(qBuf, numBlocks, { outputDtype: "f32", useVec4: false });
    const out = new Float32Array(await readBufferData(outTensor.buffer, numBlocks * 256 * 4));
    qBuf.destroy();
    outTensor.buffer.destroy();
    return out;
  },
  /**
   * Run Q4_K dequantization to F16 output (production path)
   * This matches what the loader uses during model loading
   */
  async runDequantQ4K_F16(dev, quantized, numBlocks) {
    if (!dequantize2) {
      throw new Error("dequantize kernel not available");
    }
    const qBuf = makeBuffer(quantized, GPUBufferUsage.STORAGE);
    const outTensor = await dequantize2(qBuf, numBlocks, { outputDtype: "f16", useVec4: true });
    const f16Bytes = numBlocks * 256 * 2;
    const rawData = await readBufferData(outTensor.buffer, f16Bytes);
    const u16 = new Uint16Array(rawData);
    const out = new Float32Array(numBlocks * 256);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i];
      const sign = h >> 15 & 1;
      const exp = h >> 10 & 31;
      const mant = h & 1023;
      let f;
      if (exp === 0) {
        f = mant === 0 ? 0 : Math.pow(2, -14) * (mant / 1024);
      } else if (exp === 31) {
        f = mant === 0 ? Infinity : NaN;
      } else {
        f = Math.pow(2, exp - 15) * (1 + mant / 1024);
      }
      out[i] = sign ? -f : f;
    }
    qBuf.destroy();
    outTensor.buffer.destroy();
    return out;
  },
  /**
   * Run attention kernel
   * kernel-selector API: runAttention(Q, K, V, mask, numHeads, headDim, options)
   * options: { seqLen, kvLen, numKVHeads, scale, causal }
   *
   * Uses production kernel selector which automatically chooses appropriate tier
   * (subgroup, tiled_small, streaming) based on device capabilities.
   */
  async runAttention(dev, Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim, mask = null) {
    if (!runAttention2) {
      return attentionRef(Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim, mask);
    }
    const qBuf = makeBuffer(Q);
    const kBuf = makeBuffer(K);
    const vBuf = makeBuffer(V);
    const maskBuf = mask ? makeBuffer(mask) : null;
    const isCausal = !!mask;
    const outBuf = await runAttention2(qBuf, kBuf, vBuf, maskBuf, numHeads, headDim, {
      seqLen,
      kvLen,
      numKVHeads,
      scale: 1 / Math.sqrt(headDim),
      causal: isCausal
    });
    const out = new Float32Array(await readBufferData(outBuf, seqLen * numHeads * headDim * 4));
    qBuf.destroy();
    kBuf.destroy();
    vBuf.destroy();
    maskBuf?.destroy();
    outBuf.destroy();
    return out;
  },
  /**
   * Run MoE gather - dispatches tokens to experts
   * Now uses the fixed two-phase GPU kernel (count_and_map + gather_tokens)
   */
  async runMoEGather(dev, tokens, expertIndices, numTokens, hiddenSize, numExperts, topK) {
    if (!runMoEGather2) {
      const result2 = moeGatherRef(tokens, expertIndices, numTokens, hiddenSize, numExperts, topK);
      return {
        gatheredTokens: result2.gatheredTokens,
        tokenCounts: result2.tokenCounts
      };
    }
    const tokensBuf = makeBuffer(tokens);
    const indicesBuf = makeBuffer(expertIndices);
    const tokensTensor = createTensor(tokensBuf, "f32", [numTokens, hiddenSize], "moe_input");
    const result = await runMoEGather2(tokensTensor, indicesBuf, numTokens, hiddenSize, numExperts, topK);
    const maxTokensPerExpert = result.maxTokensPerExpert;
    const gatheredTokens = new Float32Array(await readBufferData(result.gathered.buffer, numExperts * maxTokensPerExpert * hiddenSize * 4));
    const tokenCounts = new Uint32Array(await readBufferData(result.tokenCounts, numExperts * 4));
    tokensBuf.destroy();
    indicesBuf.destroy();
    result.gathered.buffer.destroy();
    result.tokenCounts.destroy();
    result.tokenMap.destroy();
    return {
      gatheredTokens,
      tokenCounts
    };
  },
  /**
   * Run GPU argmax (greedy decoding)
   */
  async runArgmax(dev, logits) {
    const logitsBuf = makeBuffer(logits);
    const tokenId = await runArgmax(logitsBuf, logits.length);
    logitsBuf.destroy();
    return tokenId;
  },
  /**
   * Run GPU top-k sampling with temperature
   */
  async runSampleTopK(dev, logits, temperature, topK, randomValue) {
    const logitsBuf = makeBuffer(logits);
    const tokenId = await runGPUSample(logitsBuf, logits.length, {
      temperature,
      topK,
      randomSeed: randomValue * 1e4
      // Convert to seed
    });
    logitsBuf.destroy();
    return tokenId;
  },
  /**
   * Run SwiGLU activation: output = SiLU(gate) * up
   * Tests the gated SiLU variant from the silu kernel
   */
  async runSwiGLU(dev, gate, up, gateBias, upBias) {
    const size = gate.length;
    const gateWithBias = new Float32Array(size);
    const upWithBias = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      gateWithBias[i] = gate[i] + gateBias[i];
      upWithBias[i] = up[i] + upBias[i];
    }
    if (!runSiLU2) {
      const result2 = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        const silu2 = gateWithBias[i] / (1 + Math.exp(-gateWithBias[i]));
        result2[i] = silu2 * upWithBias[i];
      }
      return result2;
    }
    const gateBuf = makeBuffer(gateWithBias);
    const upBuf = makeBuffer(upWithBias);
    const resultBuf = await runSiLU2(upBuf, { size, gate: gateBuf });
    const result = new Float32Array(await readBufferData(resultBuf, size * 4));
    gateBuf.destroy();
    upBuf.destroy();
    resultBuf.destroy();
    return result;
  },
  /**
   * Run scale kernel: output[i] = input[i] * scale
   */
  async runScale(dev, input, scale) {
    if (!runScale2) {
      const result2 = new Float32Array(input.length);
      for (let i = 0; i < input.length; i++) {
        result2[i] = input[i] * scale;
      }
      return result2;
    }
    const inputBuf = makeBuffer(input);
    const resultBuf = await runScale2(inputBuf, scale, { count: input.length });
    const result = new Float32Array(await readBufferData(resultBuf, input.length * 4));
    inputBuf.destroy();
    resultBuf.destroy();
    return result;
  },
  /**
   * Run BF16 → F32 cast
   */
  async runBF16ToF32(dev, input) {
    if (!runBF16ToF322) {
      const out2 = new Float32Array(input.length);
      for (let i = 0; i < input.length; i++) {
        const view = new DataView(new ArrayBuffer(4));
        view.setUint32(0, input[i] << 16, true);
        out2[i] = view.getFloat32(0, true);
      }
      return out2;
    }
    const inputBuf = makeBuffer(input, GPUBufferUsage.STORAGE);
    const outTensor = await runBF16ToF322(inputBuf, [input.length], "bf16_to_f32_test");
    const out = new Float32Array(await readBufferData(outTensor.buffer, input.length * 4));
    inputBuf.destroy();
    outTensor.buffer.destroy();
    return out;
  },
  /**
   * Run F32 → F16 cast
   */
  async runF32ToF16(dev, input) {
    if (!castF32ToF162) {
      const out2 = new Uint16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        const view = new DataView(new ArrayBuffer(4));
        view.setFloat32(0, input[i], true);
        const bits = view.getUint32(0, true);
        const sign = bits >> 31 & 1;
        const exp = bits >> 23 & 255;
        const mant = bits & 8388607;
        let hExp = 0;
        let hMant = 0;
        if (exp === 255) {
          hExp = 31;
          hMant = mant ? 512 : 0;
        } else if (exp !== 0) {
          const newExp = exp - 127 + 15;
          if (newExp >= 31) {
            hExp = 31;
          } else if (newExp > 0) {
            hExp = newExp;
            hMant = mant >> 13;
          }
        }
        out2[i] = sign << 15 | hExp << 10 | hMant;
      }
      return out2;
    }
    const inputBuf = makeBuffer(input);
    const inputTensor = createTensor(inputBuf, "f32", [input.length], "f32_to_f16_input");
    const outTensor = await castF32ToF162(inputTensor);
    const out = new Uint16Array(await readBufferData(outTensor.buffer, input.length * 2));
    inputBuf.destroy();
    outTensor.buffer.destroy();
    return out;
  },
  /**
   * Run BF16 → F16 cast
   */
  async runBF16ToF16(dev, input) {
    if (!runBF16ToF162) {
      const out2 = new Uint16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        const view = new DataView(new ArrayBuffer(4));
        view.setUint32(0, input[i] << 16, true);
        const bits = view.getUint32(0, true);
        const sign = bits >> 31 & 1;
        const exp = bits >> 23 & 255;
        const mant = bits & 8388607;
        let hExp = 0;
        let hMant = 0;
        if (exp === 255) {
          hExp = 31;
          hMant = mant ? 512 : 0;
        } else if (exp !== 0) {
          const newExp = exp - 127 + 15;
          if (newExp >= 31) {
            hExp = 31;
          } else if (newExp > 0) {
            hExp = newExp;
            hMant = mant >> 13;
          }
        }
        out2[i] = sign << 15 | hExp << 10 | hMant;
      }
      return out2;
    }
    const inputBuf = makeBuffer(input, GPUBufferUsage.STORAGE);
    const outTensor = await runBF16ToF162(inputBuf, [input.length], "bf16_to_f16_test");
    const out = new Uint16Array(await readBufferData(outTensor.buffer, input.length * 2));
    inputBuf.destroy();
    outTensor.buffer.destroy();
    return out;
  },
  /**
   * Run Q6_K dequantization
   * Note: Q6K outputs f16, which we read as f16 and convert to f32
   */
  async runDequantQ6K(dev, quantized, numBlocks) {
    if (!dequantizeQ6K2) {
      throw new Error("dequantizeQ6K kernel not available");
    }
    const blockSize = 256;
    const qBuf = makeBuffer(quantized, GPUBufferUsage.STORAGE);
    const outBuf = await dequantizeQ6K2(qBuf, numBlocks, { outputOffset: 0 });
    const rawData = await readBufferData(outBuf, numBlocks * blockSize * 2);
    const u16View = new Uint16Array(rawData);
    const out = new Float32Array(u16View.length);
    for (let i = 0; i < u16View.length; i++) {
      out[i] = f16ToF322(u16View[i]);
    }
    qBuf.destroy();
    outBuf.destroy();
    return out;
  },
  /**
   * Run F16 weights matmul (f16w_f32a kernel)
   * Takes F32 activations and F16 weights (as Uint16Array)
   * This tests the exact same kernel path as production
   * C = A[M,K] @ B[N,K]^T = C[M,N]
   */
  async runMatmulF16W(dev, A, B_f16, M, N, K) {
    if (!runMatmul2) {
      throw new Error("runMatmul kernel not available");
    }
    const bufA = makeBuffer(A);
    const tensorA = createTensor(bufA, "f32", [M, K], "matmul_f16w_a");
    const bufB = makeBuffer(B_f16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const resultTensor = await runMatmul2(tensorA, bufB, M, N, K, {
      bDtype: "f16",
      preferF16: true,
      transposeB: true
    });
    const result = new Float32Array(await readBufferData(resultTensor.buffer, M * N * 4));
    bufA.destroy();
    bufB.destroy();
    resultTensor.buffer.destroy();
    return result;
  },
  /**
   * Combined Q4K dequant + F16 matmul (production path)
   * Does dequant to F16 then matmul, all on GPU without CPU round-trip
   * C = A[M,K] @ dequant(B_q4k[N,K])^T = C[M,N]
   */
  async runDequantAndMatmulF16W(dev, A, B_q4k, M, N, K, numBlocks) {
    if (!runMatmul2 || !dequantize2) {
      throw new Error("runMatmul or dequantize kernel not available");
    }
    const qBuf = makeBuffer(B_q4k, GPUBufferUsage.STORAGE);
    const dequantTensor = await dequantize2(qBuf, numBlocks, { outputDtype: "f16", useVec4: true });
    const bufA = makeBuffer(A);
    const tensorA = createTensor(bufA, "f32", [M, K], "dequant_matmul_a");
    const resultTensor = await runMatmul2(tensorA, dequantTensor.buffer, M, N, K, {
      bDtype: "f16",
      preferF16: true,
      transposeB: true
    });
    const result = new Float32Array(await readBufferData(resultTensor.buffer, M * N * 4));
    qBuf.destroy();
    bufA.destroy();
    dequantTensor.buffer.destroy();
    resultTensor.buffer.destroy();
    return result;
  }
};
window.testHarness = testHarness;
window.gpuReady = false;
window.gpuError = void 0;
window.addEventListener("DOMContentLoaded", async () => {
  try {
    await initGPU();
    console.log("WebGPU initialized successfully");
    window.gpuReady = true;
    const status = document.getElementById("status");
    if (status) {
      const caps = getKernelCapabilities();
      status.innerHTML = `
        <strong>WebGPU Ready</strong><br>
        Adapter: ${caps?.adapterInfo || "Unknown"}<br>
        F16 Support: ${caps?.hasF16 ? "Yes" : "No"}<br>
        Subgroups: ${caps?.hasSubgroups ? "Yes" : "No"}
      `;
      status.style.color = "green";
    }
  } catch (e) {
    console.error("Failed to initialize WebGPU:", e);
    window.gpuReady = false;
    window.gpuError = e.message;
    const status = document.getElementById("status");
    if (status) {
      status.innerHTML = `<strong>WebGPU Error:</strong> ${e.message}`;
      status.style.color = "red";
    }
  }
});
export {
  getGPU,
  initGPU,
  testHarness
};
//# sourceMappingURL=test-page.js.map
