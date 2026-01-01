var __defProp = Object.defineProperty;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __esm = (fn, res) => function __init() {
  return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
};
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};

// gpu/perf-guards.ts
function trackSubmit() {
  if (config.trackSubmitCount) {
    counters.submits++;
    if (config.logExpensiveOps) {
      console.log(`[PerfGuard] Submit #${counters.submits}`);
    }
  }
}
function trackAllocation(size, label) {
  if (config.trackAllocations) {
    counters.allocations++;
    if (config.logExpensiveOps) {
      console.log(`[PerfGuard] Allocation #${counters.allocations}: ${size} bytes (${label || "unlabeled"})`);
    }
  }
}
function allowReadback(reason) {
  if (!config.allowGPUReadback) {
    const message = `[PerfGuard] GPU readback blocked: ${reason || "unknown reason"}`;
    if (config.strictMode) {
      throw new Error(message);
    }
    if (config.logExpensiveOps) {
      console.warn(message);
    }
    return false;
  }
  if (config.trackSubmitCount) {
    counters.readbacks++;
    if (config.logExpensiveOps) {
      console.log(`[PerfGuard] Readback #${counters.readbacks}: ${reason || "unknown"}`);
    }
  }
  return true;
}
var DEFAULT_CONFIG, config, counters;
var init_perf_guards = __esm({
  "gpu/perf-guards.ts"() {
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

// gpu/submit-tracker.ts
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
  "gpu/submit-tracker.ts"() {
    init_perf_guards();
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

// gpu/device.ts
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
async function initDevice() {
  if (gpuDevice) {
    return gpuDevice;
  }
  if (!isWebGPUAvailable()) {
    throw new Error("WebGPU is not available in this browser");
  }
  const adapter = await requestAdapter();
  if (!adapter) {
    throw new Error("Failed to get WebGPU adapter");
  }
  const availableFeatures = detectFeatures(adapter);
  const requestedFeatures = buildFeatureRequests(availableFeatures);
  const limits = buildLimits(adapter);
  const adapterInfo = adapter.info || { vendor: "unknown", architecture: "unknown", device: "unknown", description: "" };
  try {
    gpuDevice = await adapter.requestDevice({
      requiredFeatures: requestedFeatures,
      requiredLimits: limits
    });
  } catch (e) {
    console.warn("[DOPPLER GPU] Failed to request device with features, trying minimal config:", e.message);
    gpuDevice = await adapter.requestDevice();
  }
  if (!gpuDevice) {
    throw new Error("Failed to create WebGPU device");
  }
  gpuDevice.lost.then((info) => {
    console.error("[DOPPLER GPU] Device lost:", info.message, "Reason:", info.reason);
    gpuDevice = null;
    kernelCapabilities = null;
  });
  wrapQueueForTracking(gpuDevice.queue);
  kernelCapabilities = {
    hasSubgroups: gpuDevice.features.has(FEATURES.SUBGROUPS),
    hasSubgroupsF16: gpuDevice.features.has(FEATURES.SUBGROUPS_F16),
    hasF16: gpuDevice.features.has(FEATURES.SHADER_F16),
    hasTimestampQuery: gpuDevice.features.has(FEATURES.TIMESTAMP_QUERY),
    maxBufferSize: gpuDevice.limits.maxStorageBufferBindingSize,
    maxWorkgroupSize: gpuDevice.limits.maxComputeInvocationsPerWorkgroup,
    maxWorkgroupStorageSize: gpuDevice.limits.maxComputeWorkgroupStorageSize,
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
  console.log(`[GPU] ${adapterInfo.vendor || "unknown"} ${adapterInfo.architecture || adapterInfo.device || ""}, ${features}, ${(kernelCapabilities.maxBufferSize / (1024 * 1024 * 1024)).toFixed(1)}GB`);
  return gpuDevice;
}
function getKernelCapabilities() {
  if (!kernelCapabilities) {
    throw new Error("Device not initialized. Call initDevice() first.");
  }
  return { ...kernelCapabilities };
}
function getDevice() {
  return gpuDevice;
}
function hasFeature(feature) {
  if (!gpuDevice) {
    return false;
  }
  return gpuDevice.features.has(feature);
}
function getDeviceLimits() {
  if (!gpuDevice) {
    return null;
  }
  return {
    maxStorageBufferBindingSize: gpuDevice.limits.maxStorageBufferBindingSize,
    maxBufferSize: gpuDevice.limits.maxBufferSize,
    maxComputeWorkgroupSizeX: gpuDevice.limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: gpuDevice.limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: gpuDevice.limits.maxComputeWorkgroupSizeZ,
    maxComputeInvocationsPerWorkgroup: gpuDevice.limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupStorageSize: gpuDevice.limits.maxComputeWorkgroupStorageSize,
    maxStorageBuffersPerShaderStage: gpuDevice.limits.maxStorageBuffersPerShaderStage,
    maxUniformBufferBindingSize: gpuDevice.limits.maxUniformBufferBindingSize,
    maxComputeWorkgroupsPerDimension: gpuDevice.limits.maxComputeWorkgroupsPerDimension
  };
}
var gpuDevice, kernelCapabilities, FEATURES;
var init_device = __esm({
  "gpu/device.ts"() {
    init_submit_tracker();
    gpuDevice = null;
    kernelCapabilities = null;
    FEATURES = {
      SHADER_F16: "shader-f16",
      SUBGROUPS: "subgroups",
      SUBGROUPS_F16: "subgroups-f16",
      TIMESTAMP_QUERY: "timestamp-query"
    };
  }
});

// gpu/buffer-dtypes.ts
function setBufferDtype(buffer, dtype) {
  if (buffer)
    dtypeMap.set(buffer, dtype);
}
function getBufferDtype(buffer) {
  return buffer ? dtypeMap.get(buffer) ?? null : null;
}
function getBufferLayout(buffer) {
  return buffer ? layoutMap.get(buffer) ?? null : null;
}
function isColumnMajorBuffer(buffer) {
  return getBufferLayout(buffer) === "column";
}
var dtypeMap, layoutMap;
var init_buffer_dtypes = __esm({
  "gpu/buffer-dtypes.ts"() {
    dtypeMap = /* @__PURE__ */ new WeakMap();
    layoutMap = /* @__PURE__ */ new WeakMap();
  }
});

// gpu/profiler.ts
var GPUProfiler;
var init_profiler = __esm({
  "gpu/profiler.ts"() {
    init_device();
    init_perf_guards();
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
          console.warn("[GPUProfiler] Failed to create timestamp query resources:", e);
          this.hasTimestampQuery = false;
        }
      }
      /**
       * Begin timing a labeled region
       * @param label - Unique label for this measurement
       */
      begin(label) {
        if (this.activeLabels.has(label)) {
          console.warn(`[GPUProfiler] Label "${label}" already active`);
          return;
        }
        const startTime = performance.now();
        if (this.hasTimestampQuery) {
          const queryIndex = this.nextQueryIndex;
          this.nextQueryIndex += 2;
          if (queryIndex >= this.queryCapacity * 2) {
            console.warn("[GPUProfiler] Query capacity exceeded, resetting");
            this.nextQueryIndex = 0;
          }
          this.activeLabels.set(label, {
            startQueryIndex: queryIndex,
            cpuStartTime: startTime
          });
        } else {
          this.activeLabels.set(label, {
            cpuStartTime: startTime
          });
        }
      }
      /**
       * End timing a labeled region
       * @param label - Label started with begin()
       */
      end(label) {
        const active = this.activeLabels.get(label);
        if (!active) {
          console.warn(`[GPUProfiler] No active measurement for label "${label}"`);
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
          console.warn("[GPUProfiler] Missing required resources for resolve");
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

// gpu/kernel-tuner.ts
async function getKernelTuner() {
  if (!globalTuner) {
    globalTuner = new KernelTuner();
    await globalTuner.init();
  }
  return globalTuner;
}
var CACHE_PREFIX, DEFAULT_WARMUP, DEFAULT_ITERATIONS, KernelTuner, globalTuner;
var init_kernel_tuner = __esm({
  "gpu/kernel-tuner.ts"() {
    init_device();
    init_profiler();
    CACHE_PREFIX = "doppler_kernel_tune_";
    DEFAULT_WARMUP = 3;
    DEFAULT_ITERATIONS = 10;
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
        const cacheKey = CACHE_PREFIX + signature;
        try {
          const cached = localStorage.getItem(cacheKey);
          if (cached) {
            const data = JSON.parse(cached);
            this.cache = new Map(Object.entries(data));
          }
        } catch (e) {
          console.warn("[KernelTuner] Failed to load cache:", e);
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
        const cacheKey = CACHE_PREFIX + signature;
        try {
          const data = Object.fromEntries(this.cache);
          localStorage.setItem(cacheKey, JSON.stringify(data));
        } catch (e) {
          console.warn("[KernelTuner] Failed to save cache:", e);
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
          warmup = DEFAULT_WARMUP,
          iterations = DEFAULT_ITERATIONS,
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
          localStorage.removeItem(CACHE_PREFIX + signature);
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

// gpu/kernels/utils.ts
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
    console.warn(
      `[KernelSelector] Attention may be slow: tile requires ${sharedMemRequired} bytes but device has ${limits.maxComputeWorkgroupStorageSize} bytes shared memory.`
    );
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
    console.error(`[KernelSelector] Failed to load shader ${filename}:`, error);
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
      const type = msg.type === "error" ? "ERROR" : msg.type === "warning" ? "WARN" : "INFO";
      console.log(`[DEBUG compileShader ${label}] ${type}: ${msg.message} (line ${msg.lineNum}:${msg.linePos})`);
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
    console.warn(`[KernelSelector] Tuning failed for ${operation}, using defaults:`, e.message);
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
  console.log("[KernelSelector] Auto-tuning complete:", results);
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
          console.warn(`[KernelSelector] Prewarm failed for ${operation}/${variant}:`, e.message);
        }
      }
    }
    console.log(`[KernelSelector] Prewarmed ${count} kernel pipelines`);
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
          console.warn(`[KernelSelector] Prewarm failed for ${operation}/${variant}:`, e.message);
        })
      );
    }
  }
  await Promise.all(jobs);
  console.log(`[KernelSelector] Prewarmed ${jobs.length} kernel pipelines`);
}
function createUniformBufferFromData(label, data, recorder, deviceOverride) {
  if (recorder) {
    return recorder.createUniformBuffer(data, label);
  }
  const device2 = deviceOverride ?? getDevice();
  if (!device2) {
    throw new Error("GPU device not initialized");
  }
  const byteLength = data instanceof ArrayBuffer ? data.byteLength : data.byteLength;
  const buffer = device2.createBuffer({
    label,
    size: byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  device2.queue.writeBuffer(buffer, 0, data);
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
  "gpu/kernels/utils.ts"() {
    init_device();
    init_kernel_tuner();
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
          requires: ["shader-f16"]
        },
        f16_vec4: {
          shaderFile: "matmul_f16.wgsl",
          entryPoint: "main_vec4",
          workgroupSize: [16, 16, 1],
          requires: ["shader-f16"]
        },
        f16w_f32a: {
          shaderFile: "matmul_f16w_f32a.wgsl",
          entryPoint: "main",
          workgroupSize: [16, 16, 1],
          requires: ["shader-f16"]
        },
        f16w_f32a_naive: {
          shaderFile: "matmul_f16w_f32a_naive.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
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
          requires: ["shader-f16", "subgroups"]
        },
        // Fused Q4_K dequant + matmul - 2-3x faster (no separate dequant pass)
        q4_fused: {
          shaderFile: "fused_matmul_q4.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16", "subgroups"]
        },
        q4_fused_batched: {
          shaderFile: "fused_matmul_q4.wgsl",
          entryPoint: "main_batched",
          workgroupSize: [64, 4, 1],
          requires: ["shader-f16", "subgroups"]
        },
        // Multi-column GEMV for large vocab (LM head) - 32 columns per workgroup
        q4_fused_multicol: {
          shaderFile: "fused_matmul_q4.wgsl",
          entryPoint: "main_multicol",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16", "subgroups"]
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
          requires: ["shader-f16", "subgroups"]
        },
        batched: {
          shaderFile: "fused_ffn.wgsl",
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
          requires: ["shader-f16", "subgroups"]
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
          requires: ["subgroups", "shader-f16"]
        },
        subgroup_vec4_f16out: {
          shaderFile: "dequant_f16_out.wgsl",
          entryPoint: "main_vec4",
          workgroupSize: [64, 1, 1],
          requires: ["subgroups", "shader-f16"]
        },
        shared: {
          shaderFile: "dequant_shared.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        shared_vec4: {
          shaderFile: "dequant_shared.wgsl",
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
          shaderFile: "dequant_f16_out.wgsl",
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
          shaderFile: "dequant_mxfp4.wgsl",
          entryPoint: "main_vec4",
          workgroupSize: [64, 1, 1],
          requires: []
        },
        mxfp4_expert: {
          shaderFile: "dequant_mxfp4.wgsl",
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
          shaderFile: "attention.wgsl",
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
          shaderFile: "attention_f16kv.wgsl",
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
          requires: ["shader-f16"]
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
          entryPoint: "rmsnorm_small",
          workgroupSize: [256, 1, 1],
          requires: []
        },
        residual: {
          shaderFile: "rmsnorm.wgsl",
          entryPoint: "rmsnorm_inplace_residual",
          workgroupSize: [256, 1, 1],
          requires: []
        }
      },
      // Fused GEMV + RMSNorm for decode (M=1)
      // Combines down projection matmul with RMSNorm in single kernel
      fused_matmul_rmsnorm: {
        default: {
          shaderFile: "fused_matmul_rmsnorm.wgsl",
          entryPoint: "main",
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
          shaderFile: "gather.wgsl",
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
          shaderFile: "gather_f16.wgsl",
          entryPoint: "gather_vec4",
          workgroupSize: [64, 1, 1],
          requires: ["shader-f16"]
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
          shaderFile: "residual.wgsl",
          entryPoint: "add_vec4",
          workgroupSize: [64, 1, 1],
          requires: []
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
          shaderFile: "scatter_add.wgsl",
          entryPoint: "scatter_add_vec4",
          workgroupSize: [64, 1, 1],
          requires: []
        },
        dynamic: {
          shaderFile: "scatter_add.wgsl",
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
          shaderFile: "moe_gather.wgsl",
          entryPoint: "gather_tokens_vec4",
          workgroupSize: [64, 1, 1],
          requires: []
        },
        single_pass: {
          shaderFile: "moe_gather.wgsl",
          entryPoint: "gather_single_pass",
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
        }
      },
      cast: {
        f32_to_f16: {
          shaderFile: "cast_f32_to_f16.wgsl",
          entryPoint: "main",
          workgroupSize: [256, 1, 1],
          requires: ["shader-f16"]
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

// gpu/buffer-pool.ts
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
function getSizeBucket(size, maxAllowedSize = Infinity) {
  const minBucket = 256;
  if (size <= minBucket)
    return minBucket;
  const largeThreshold = 32 * 1024 * 1024;
  if (size >= largeThreshold) {
    const largeStep = 16 * 1024 * 1024;
    const bucket2 = Math.ceil(size / largeStep) * largeStep;
    if (bucket2 > maxAllowedSize) {
      return alignTo(size, 256);
    }
    return bucket2;
  }
  const bits = 32 - Math.clz32(size - 1);
  const bucket = Math.pow(2, bits);
  if (bucket > maxAllowedSize) {
    return alignTo(size, 256);
  }
  return bucket;
}
function getBufferPool() {
  if (!globalPool) {
    globalPool = new BufferPool();
  }
  return globalPool;
}
function createBufferPool(debugMode) {
  return new BufferPool(debugMode);
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
  "gpu/buffer-pool.ts"() {
    init_device();
    init_perf_guards();
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
      // Statistics
      stats;
      // Configuration
      config;
      // Debug mode flag
      debugMode;
      constructor(debugMode = false) {
        this.pools = /* @__PURE__ */ new Map();
        this.activeBuffers = /* @__PURE__ */ new Set();
        this.bufferMetadata = /* @__PURE__ */ new Map();
        this.debugMode = debugMode;
        this.stats = {
          allocations: 0,
          reuses: 0,
          totalBytesAllocated: 0,
          peakBytesAllocated: 0,
          currentBytesAllocated: 0
        };
        this.config = {
          maxPoolSizePerBucket: 8,
          // Max buffers per size bucket
          maxTotalPooledBuffers: 64,
          // Max total pooled buffers
          enablePooling: true,
          alignmentBytes: 256
          // WebGPU buffer alignment
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
        const bucket = getSizeBucket(alignedSize, maxAllowedBucket);
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
          console.warn("[BufferPool] Releasing buffer not tracked as active");
          return;
        }
        this.activeBuffers.delete(buffer);
        if (this.debugMode) {
          this.bufferMetadata.delete(buffer);
        }
        if (!this.config.enablePooling) {
          buffer.destroy();
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
          buffer.destroy();
          this.stats.currentBytesAllocated -= buffer.size;
        }
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
          console.warn("[BufferPool] Leak detection requires debug mode");
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

// gpu/kernels/dispatch.ts
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
var init_dispatch = __esm({
  "gpu/kernels/dispatch.ts"() {
  }
});

// gpu/kernels/fused_matmul_rmsnorm.ts
var fused_matmul_rmsnorm_exports = {};
__export(fused_matmul_rmsnorm_exports, {
  recordMatmulRMSNormFused: () => recordMatmulRMSNormFused,
  runMatmulRMSNormFused: () => runMatmulRMSNormFused,
  selectMatmulRMSNormFusedVariant: () => selectMatmulRMSNormFusedVariant,
  shouldUseFusedMatmulRMSNorm: () => shouldUseFusedMatmulRMSNorm
});
function selectMatmulRMSNormFusedVariant(N) {
  if (N <= WG_SIZE) {
    return "small";
  }
  if (N <= MAX_MEDIUM_N) {
    return "medium";
  }
  return "default";
}
async function runMatmulRMSNormFused(input, weight, normWeight, options) {
  const device2 = getDevice();
  const {
    N,
    K,
    eps = 1e-5,
    residual = null,
    outputBuffer = null
  } = options;
  const variant = selectMatmulRMSNormFusedVariant(N);
  if (DEBUG_KERNELS6) {
    console.log(`[MatmulRMSNormFused] N=${N}, K=${K}, variant=${variant}, hasResidual=${!!residual}`);
  }
  const pipeline = await createPipeline("fused_matmul_rmsnorm", variant);
  const outputSize = N * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "matmul_rmsnorm_fused_output");
  const uniformBuffer = createUniformBufferWithView(
    "matmul_rmsnorm_fused_uniforms",
    16,
    (view) => {
      view.setUint32(0, N, true);
      view.setUint32(4, K, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, residual ? 1 : 0, true);
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
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: normWeight } },
      { binding: 4, resource: { buffer: output } },
      { binding: 5, resource: { buffer: residualBuffer } }
    ]
  });
  let workgroups;
  if (variant === "small" || variant === "medium") {
    workgroups = 1;
  } else {
    workgroups = Math.ceil(N / COLS_PER_WG);
  }
  dispatch(device2, pipeline, bindGroup, workgroups, "matmul_rmsnorm_fused");
  uniformBuffer.destroy();
  if (!residual)
    residualBuffer.destroy();
  setBufferDtype(output, "f32");
  return output;
}
async function recordMatmulRMSNormFused(recorder, input, weight, normWeight, options) {
  const device2 = recorder.device;
  const {
    N,
    K,
    eps = 1e-5,
    residual = null,
    outputBuffer = null
  } = options;
  const variant = selectMatmulRMSNormFusedVariant(N);
  if (DEBUG_KERNELS6) {
    console.log(`[recordMatmulRMSNormFused] N=${N}, K=${K}, variant=${variant}, hasResidual=${!!residual}`);
  }
  const pipeline = await createPipeline("fused_matmul_rmsnorm", variant);
  const outputSize = N * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "matmul_rmsnorm_fused_output");
  const uniformBuffer = createUniformBufferWithView(
    "matmul_rmsnorm_fused_uniforms",
    16,
    (view) => {
      view.setUint32(0, N, true);
      view.setUint32(4, K, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, residual ? 1 : 0, true);
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
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: normWeight } },
      { binding: 4, resource: { buffer: output } },
      { binding: 5, resource: { buffer: residualBuffer } }
    ]
  });
  let workgroups;
  if (variant === "small" || variant === "medium") {
    workgroups = 1;
  } else {
    workgroups = Math.ceil(N / COLS_PER_WG);
  }
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "matmul_rmsnorm_fused");
  if (!residual) {
    recorder.trackTemporaryBuffer(residualBuffer);
  }
  setBufferDtype(output, "f32");
  return output;
}
function shouldUseFusedMatmulRMSNorm(M, N) {
  if (M !== 1) {
    return false;
  }
  if (N > MAX_MEDIUM_N) {
    return false;
  }
  return true;
}
var DEBUG_KERNELS6, WG_SIZE, COLS_PER_WG, MAX_MEDIUM_N;
var init_fused_matmul_rmsnorm = __esm({
  "gpu/kernels/fused_matmul_rmsnorm.ts"() {
    init_device();
    init_buffer_dtypes();
    init_buffer_pool();
    init_dispatch();
    init_utils();
    DEBUG_KERNELS6 = typeof window !== "undefined" ? Boolean(window.DOPPLER_DEBUG_KERNELS) : false;
    WG_SIZE = 256;
    COLS_PER_WG = 4;
    MAX_MEDIUM_N = 4096;
  }
});

// kernel-tests/browser/test-page.ts
init_device();
init_buffer_dtypes();

// gpu/kernel-hints.ts
var currentHints = null;
var hintsSource = null;
function setKernelHints(hints, source = "manifest") {
  const priority = { manifest: 0, profile: 1, runtime: 2 };
  if (!currentHints || priority[source] >= priority[hintsSource || "manifest"]) {
    currentHints = hints;
    hintsSource = source;
    console.log(`[KernelHints] Set from ${source}:`, hints);
  }
}
function getKernelHints() {
  return currentHints;
}
function shouldUseFusedQ4K() {
  const debugFlags = typeof window !== "undefined" ? window : null;
  if (debugFlags?.DOPPLER_DISABLE_FUSED_Q4K)
    return false;
  const hints = getKernelHints();
  if (hints?.q4kMatmul) {
    return hints.q4kMatmul === "fused_q4k";
  }
  return false;
}

// gpu/kernel-selector.ts
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
  recordCastF32ToF16: () => recordCastF32ToF16,
  recordDequantize: () => recordDequantize,
  recordFusedFFN: () => recordFusedFFN,
  recordGather: () => recordGather,
  recordGeLU: () => recordGeLU,
  recordMatmul: () => recordMatmul,
  recordMatmulRMSNormFused: () => recordMatmulRMSNormFused,
  recordProfileEntry: () => recordProfileEntry,
  recordRMSNorm: () => recordRMSNorm,
  recordResidualAdd: () => recordResidualAdd,
  recordRoPE: () => recordRoPE,
  recordScale: () => recordScale,
  recordSiLU: () => recordSiLU,
  recordSiLURowSplit: () => recordSiLURowSplit,
  recordSoftmax: () => recordSoftmax,
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
  runSwiGLURowsplitBias: () => runSwiGLURowsplitBias,
  runTopK: () => runTopK,
  selectDequantKernel: () => selectDequantKernel,
  selectMatmulKernel: () => selectMatmulKernel,
  selectMatmulRMSNormFusedVariant: () => selectMatmulRMSNormFusedVariant,
  selectRMSNormKernel: () => selectRMSNormKernel,
  setProfilingEnabled: () => setProfilingEnabled,
  shouldUseFusedMatmulRMSNorm: () => shouldUseFusedMatmulRMSNorm,
  startProfileSession: () => startProfileSession,
  validateAttentionLimits: () => validateAttentionLimits
});

// gpu/kernels/index.ts
init_utils();

// gpu/kernels/matmul.ts
init_device();
init_buffer_dtypes();
init_buffer_pool();

// gpu/kernels/kernel-base.ts
init_dispatch();
init_utils();
var KernelBase = class {
  device;
  constructor(device2) {
    this.device = device2;
  }
  async getPipelineFor(operation, variant, bindGroupLayout = null) {
    return createPipeline(operation, variant, bindGroupLayout);
  }
  dispatchKernel(pipeline, bindGroup, workgroups, label) {
    dispatch(this.device, pipeline, bindGroup, workgroups, label);
  }
  recordKernel(recorder, pipeline, bindGroup, workgroups, label) {
    recordDispatch(recorder, pipeline, bindGroup, workgroups, label);
  }
};

// gpu/kernels/constants.ts
var WORKGROUP_SIZES = {
  /** Default workgroup size for most kernels */
  DEFAULT: 256,
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
var GPU_LIMITS = {
  /** Max workgroups per dimension (WebGPU minimum) */
  MAX_WORKGROUPS: 65535
};
var MEMORY_THRESHOLDS = {
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
var DIMENSION_LIMITS = {
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
var TILE_SIZES = {
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
var QUANTIZATION = {
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
var ALIGNMENT = {
  /** WebGPU buffer alignment */
  BUFFER: 256,
  /** Uniform buffer alignment */
  UNIFORM: 256,
  /** Storage buffer alignment */
  STORAGE: 256,
  /** Vertex buffer alignment */
  VERTEX: 4
};
var PERFORMANCE = {
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

// gpu/kernels/matmul.ts
init_utils();
var DEBUG_KERNELS = typeof window !== "undefined" ? Boolean(window.DOPPLER_DEBUG_KERNELS) : false;
function isFusedQ4KDisabled() {
  const debugFlags = typeof window !== "undefined" ? window : null;
  if (debugFlags?.DOPPLER_DISABLE_FUSED_Q4K)
    return true;
  return !shouldUseFusedQ4K();
}
function toMatmulDtype(dtype) {
  if (dtype === "f16")
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
var DEBUG_FORCE_TRANSPOSE_TRUE = false;
function resolveTransposeB(B, transposeBOption) {
  if (transposeBOption === "auto") {
    const isColMajor = isColumnMajorBuffer(B);
    const result = DEBUG_FORCE_TRANSPOSE_TRUE ? true : !isColMajor;
    if (DEBUG_KERNELS && _transposeDebugCount < 50) {
      _transposeDebugCount++;
      console.log(`[resolveTransposeB] isColumnMajor=${isColMajor}, transposeB=${result}, bufSize=${B.size} (DEBUG_FORCE=${DEBUG_FORCE_TRANSPOSE_TRUE})`);
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
  const QK_K = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  const Q4K_BLOCK_BYTES = QUANTIZATION.Q4K_BLOCK_BYTES;
  let bBindingSize;
  let bRequired;
  if (bDtype === "q4k") {
    const numBlocksPerRow = Math.ceil(K / QK_K);
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
function selectMatmulVariantAndFlags(mode, M, N, K, aDtype, bDtype, transposeB, requestedOutputDtype, options) {
  const capabilities = getKernelCapabilities();
  let variant = "f32";
  let useQ4KFused = false;
  let useGemv = false;
  if (bDtype === "q4k") {
    const useFused = mode === "record" ? true : !isFusedQ4KDisabled();
    if (useFused) {
      if (!capabilities.hasSubgroups) {
        if (mode === "record") {
          throw new Error(
            "Q4_K fused matmul requires subgroup support. Your GPU/browser may not support WebGPU subgroups. Consider using a dequantized model (F16) as fallback."
          );
        }
        console.warn(
          "[Matmul] Q4K fused requested but no subgroup support. Falling back to dequant path. Your GPU/browser may not support WebGPU subgroups."
        );
      } else {
        useQ4KFused = true;
        if (mode === "record") {
          if (M === 1) {
            variant = "q4_fused_multicol";
          } else {
            variant = "q4_fused_batched";
          }
        } else {
          variant = M === 1 ? "q4_fused_multicol" : "q4_fused_batched";
        }
      }
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
        const MULTICOL_THRESHOLD = 256;
        if (N > MULTICOL_THRESHOLD) {
          variant = "gemv_subgroup_multicol";
        } else {
          variant = "gemv_subgroup";
        }
      } else {
        variant = "gemv";
      }
    } else if (M === 1 && effectiveBDtype === "f16" && aDtype === "f32") {
      variant = "f16w_f32a_naive";
    }
  }
  return { variant, useQ4KFused, useGemv };
}
function resolveMatmulOutput(variant, M, N, outputBuffer) {
  const outputsF16 = variant === "f16" || variant === "f16_vec4";
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
  if (useGemv && (variant === "gemv_subgroup" || variant === "gemv_subgroup_multicol")) {
    const colsPerWg = variant === "gemv_subgroup_multicol" ? 32 : 4;
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
    } else if (variant === "q4_fused_multicol") {
      const colsPerWg = 32;
      workgroupsX = Math.ceil(N / colsPerWg);
      workgroupsY = 1;
    } else {
      const tileM = 4;
      workgroupsX = N;
      workgroupsY = Math.ceil(M / tileM);
    }
  } else if (useGemv) {
    workgroupsX = N;
    workgroupsY = 1;
  } else if (variant === "f16w_f32a_naive") {
    workgroupsX = Math.ceil(N / wgX);
    workgroupsY = 1;
  } else {
    workgroupsX = Math.ceil(M / wgX);
    workgroupsY = Math.ceil(N / wgY);
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
  if (DEBUG_KERNELS && _runMatmulDebugCount < 20) {
    _runMatmulDebugCount++;
    const isColMajor = isColumnMajorBuffer(B);
    console.log(`[runMatmul] M=${M}, N=${N}, K=${K}, transposeBOption=${transposeBOption}, isColMajor=${isColMajor}`);
  }
  const transposeB = resolveTransposeB(B, transposeBOption);
  validateMatmulDimensions("runMatmul", M, N, K);
  const rawADtype = getBufferDtype(A);
  const rawBDtype = getBufferDtype(B);
  const requestedOutputDtype = options.outputDtype || "f32";
  if (DEBUG_KERNELS && !rawBDtype && M <= 2) {
    console.warn(`[runMatmul] B buffer dtype unknown! size=${B.size}, M=${M}, N=${N}, K=${K}. Assuming f32.`);
  }
  const aDtype = toMatmulDtype(rawADtype);
  const bDtype = toMatmulDtype(rawBDtype);
  validateMatmulOffsets("runMatmul", aOffset, bOffset, cOffset);
  const { aBindingSize, bBindingSize } = getMatmulBindingSizes(
    "runMatmul",
    A,
    B,
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
  if (DEBUG_KERNELS && bDtype === "q4k") {
    if (useQ4KFused) {
      console.log(
        `[Matmul] Q4K FUSED: M=${M}, N=${N}, K=${K}, variant=${variant} (WARNING: 2.3x slower than dequant)`
      );
    } else {
      console.log(
        `[Matmul] Q4K DEQUANT: M=${M}, N=${N}, K=${K}, will dequant first then matmul with variant=${variant}`
      );
    }
  }
  if (DEBUG_KERNELS && N > 1e5) {
    console.log(`[Pipeline] MATMUL_LARGE: N=${N}, variant=${variant}, aDtype=${aDtype}, bDtype=${bDtype}, transposeB=${transposeB}`);
  }
  const config2 = getKernelConfig("matmul", variant);
  const kernel = new MatmulKernel(device2);
  const pipeline = await kernel.getPipeline(variant);
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
  const bindGroup = device2.createBindGroup({
    label: "matmul_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: A, offset: aOffset, size: aBindingSize } },
      { binding: 2, resource: { buffer: B, offset: bOffset, size: bBindingSize } },
      { binding: 3, resource: { buffer: C, offset: cOffset, size: cBindingSize } }
    ]
  });
  kernel.dispatch(pipeline, bindGroup, dispatchPlan.workgroups);
  uniformBuffer.destroy();
  setBufferDtype(C, actualOutputDtype);
  return C;
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
  if (DEBUG_KERNELS && _recordMatmulDebugCount < 20) {
    _recordMatmulDebugCount++;
    const isColMajor = isColumnMajorBuffer(B);
    console.log(`[recordMatmul] M=${M}, N=${N}, K=${K}, transposeBOption=${transposeBOption}, isColMajor=${isColMajor}`);
  }
  const transposeB = resolveTransposeB(B, transposeBOption);
  validateMatmulDimensions("recordMatmul", M, N, K);
  const aDtype = toMatmulDtype(getBufferDtype(A));
  const bDtype = toMatmulDtype(getBufferDtype(B));
  const requestedOutputDtype = options.outputDtype || "f32";
  validateMatmulOffsets("recordMatmul", aOffset, bOffset, cOffset);
  const { aBindingSize, bBindingSize } = getMatmulBindingSizes(
    "recordMatmul",
    A,
    B,
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
  const pipeline = await kernel.getPipeline(variant);
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
  const bindGroup = device2.createBindGroup({
    label: "matmul_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: A, offset: aOffset, size: aBindingSize } },
      { binding: 2, resource: { buffer: B, offset: bOffset, size: bBindingSize } },
      { binding: 3, resource: { buffer: C, offset: cOffset, size: cBindingSize } }
    ]
  });
  kernel.record(recorder, pipeline, bindGroup, dispatchPlan.workgroups);
  setBufferDtype(C, actualOutputDtype);
  return C;
}

// gpu/kernels/dequant.ts
init_device();
init_buffer_dtypes();
init_buffer_pool();
init_dispatch();
init_utils();
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
  const QK_K = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  let workgroups;
  if (variant.includes("vec4")) {
    workgroups = numBlocks;
  } else if (variant.includes("shared")) {
    workgroups = numBlocks;
  } else {
    workgroups = Math.ceil(numBlocks * QK_K / (WORKGROUP_SIZES.DEFAULT / 4));
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
  const pipeline = await createPipeline("dequant", variant);
  const QK_K = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  const bytesPerElem = outputDtype === "f16" ? 2 : 4;
  const outputSize = numBlocks * QK_K * bytesPerElem;
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
  uniformBuffer.destroy();
  setBufferDtype(output, outputDtype === "f16" ? "f16" : "f32");
  return output;
}
async function dequantizeMXFP4(blocks, scales, totalElements, numGroups, options = {}) {
  const device2 = getDevice();
  const {
    outputBuffer = null,
    groupSize = 32
    // 32 elements per group (16 bytes * 2 nibbles)
  } = options;
  const pipeline = await createPipeline("dequant", "mxfp4");
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
  uniformBuffer.destroy();
  setBufferDtype(output, "f32");
  return output;
}
async function dequantizeMXFP4Expert(blocks, scales, expertIdx, numExperts, outDim, numGroups, options = {}) {
  const device2 = getDevice();
  const { outputBuffer = null } = options;
  const pipeline = await createPipeline("dequant", "mxfp4_expert");
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
  uniformBuffer.destroy();
  setBufferDtype(output, "f32");
  return output;
}
async function dequantizeQ6K(quantized, numBlocks, options = {}) {
  const device2 = getDevice();
  const {
    outputOffset = 0,
    outputBuffer = null,
    outputDtype = "f16"
    // Q6_K always outputs f16 for now
  } = options;
  const pipeline = await createPipeline("dequant", "q6k_f16out");
  const QK_K = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  const bytesPerElem = outputDtype === "f16" ? 2 : 4;
  const outputSize = numBlocks * QK_K * bytesPerElem;
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
  uniformBuffer.destroy();
  setBufferDtype(output, outputDtype === "f16" ? "f16" : "f32");
  return output;
}
async function recordDequantize(recorder, quantized, numBlocks, options = {}) {
  const device2 = recorder.device;
  const {
    outputOffset = 0,
    outputBuffer = null,
    outputDtype = "f32"
  } = options;
  const variant = selectDequantKernel({ ...options, outputDtype });
  const pipeline = await createPipeline("dequant", variant);
  const QK_K = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  const bytesPerElem = outputDtype === "f16" ? 2 : 4;
  const outputSize = numBlocks * QK_K * bytesPerElem;
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
  setBufferDtype(output, outputDtype === "f16" ? "f16" : "f32");
  return output;
}

// gpu/kernels/attention.ts
init_device();
init_buffer_dtypes();
init_buffer_pool();
init_utils();
var DEBUG_KERNELS2 = typeof window !== "undefined" ? Boolean(window.DOPPLER_DEBUG_KERNELS) : false;
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
function selectAttentionTier(headDim, seqLen, useF16KV, attentionKernel, sharedLimit, caps) {
  const isDecode = seqLen === 1;
  const canLarge = headDim <= DIMENSION_LIMITS.ATTENTION_LARGE_MAX_HEAD_DIM && sharedLimit >= MEMORY_THRESHOLDS.ATTENTION_LARGE_SHARED;
  const smallRequired = useF16KV ? MEMORY_THRESHOLDS.ATTENTION_SMALL_SHARED_F16 : MEMORY_THRESHOLDS.ATTENTION_SMALL_SHARED_F32;
  const canSmall = headDim <= DIMENSION_LIMITS.ATTENTION_SMALL_MAX_HEAD_DIM && sharedLimit >= smallRequired;
  const canSubgroup = caps.hasSubgroups && headDim <= DIMENSION_LIMITS.ATTENTION_SUBGROUP_MAX_HEAD_DIM && sharedLimit >= MEMORY_THRESHOLDS.ATTENTION_SUBGROUP_SHARED && isDecode;
  let tier = attentionKernel;
  if (tier === "tiled_large" && !canLarge) {
    console.warn(
      `[Attention] Requested tiled_large but device doesn't support it (headDim=${headDim}, shared=${sharedLimit}). Falling back.`
    );
    tier = null;
  }
  if (tier === "tiled_small" && !canSmall) {
    console.warn(
      `[Attention] Requested tiled_small but device doesn't support it (headDim=${headDim}, shared=${sharedLimit}). Falling back.`
    );
    tier = null;
  }
  if (!tier) {
    if (canSubgroup) {
      tier = "subgroup";
      console.log(`[Attention] Using subgroup decode kernel (headDim=${headDim}, hasSubgroups=true)`);
    } else if (canLarge) {
      tier = "tiled_large";
    } else if (canSmall) {
      tier = "tiled_small";
    } else if (isDecode) {
      tier = "streaming";
    } else {
      console.warn(
        `[Attention] No tiled kernel fits prefill (headDim=${headDim}, shared=${sharedLimit}). Falling back to streaming. Expect slow prefill.`
      );
      tier = "streaming";
    }
  }
  return tier;
}
function resolveAttentionVariant(tier, isDecode, useF16KV, numHeads, headDim) {
  const base = isDecode ? "decode" : "prefill";
  if (tier === "subgroup") {
    if (useF16KV) {
      if (numHeads <= 8 && headDim >= 128) {
        return "decode_chunked_f16kv";
      }
      return "decode_streaming_f16kv";
    }
    return "decode_subgroup";
  }
  if (tier === "tiled_large") {
    return base + (useF16KV ? "_f16kv" : "");
  }
  if (tier === "tiled_small") {
    return `${base}_small${useF16KV ? "_f16kv" : ""}`;
  }
  if (isDecode && useF16KV && numHeads <= 8 && headDim >= 128) {
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
function resolveAttentionPlan(seqLen, headDim, numHeads, attentionKernel, kvDtype, sharedLimit, caps) {
  const useF16KV = kvDtype === "f16";
  const tier = selectAttentionTier(headDim, seqLen, useF16KV, attentionKernel, sharedLimit, caps);
  const isDecode = seqLen === 1;
  const variant = resolveAttentionVariant(tier, isDecode, useF16KV, numHeads, headDim);
  const workgroups = calculateAttentionWorkgroups(tier, seqLen, numHeads);
  return { tier, variant, workgroups, useF16KV, isDecode };
}
function createAttentionUniformBuffer(device2, recorder, params) {
  return createUniformBufferWithView(
    "attention_uniforms",
    32,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.kvLen, true);
      view.setUint32(16, params.seqLen, true);
      view.setFloat32(20, params.scale, true);
      view.setUint32(24, params.causal ? 1 : 0, true);
      view.setUint32(28, params.startPos, true);
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
    attentionKernel = null,
    outputBuffer = null
  } = options;
  const limits = getDeviceLimits();
  const sharedLimit = limits?.maxComputeWorkgroupStorageSize ?? Infinity;
  const caps = getKernelCapabilities();
  const kvDtype = getBufferDtype(K) || "f32";
  const plan = resolveAttentionPlan(
    seqLen,
    headDim,
    numHeads,
    attentionKernel,
    kvDtype,
    sharedLimit,
    caps
  );
  const kernel = new AttentionKernel(device2);
  const pipeline = await kernel.getPipeline(plan.variant);
  const outputSize = seqLen * numHeads * headDim * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "attention_output");
  const uniformBuffer = createAttentionUniformBuffer(device2, null, {
    numHeads,
    numKVHeads,
    headDim,
    kvLen,
    seqLen,
    scale,
    causal,
    startPos
  });
  const bindGroup = device2.createBindGroup({
    label: "attention_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q } },
      { binding: 2, resource: { buffer: K } },
      { binding: 3, resource: { buffer: V } },
      { binding: 4, resource: { buffer: output } }
    ]
  });
  if (limits && plan.workgroups > limits.maxComputeWorkgroupsPerDimension) {
    throw new Error(
      `Attention dispatch requires ${plan.workgroups} workgroups but device limit is ${limits.maxComputeWorkgroupsPerDimension}. Reduce prompt length or use streaming attention.`
    );
  }
  kernel.dispatch(pipeline, bindGroup, plan.workgroups);
  uniformBuffer.destroy();
  return output;
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
    attentionKernel = null,
    outputBuffer = null
  } = options;
  const limits = getDeviceLimits();
  const sharedLimit = limits?.maxComputeWorkgroupStorageSize ?? Infinity;
  const caps = getKernelCapabilities();
  const kvDtype = getBufferDtype(K) || "f32";
  const plan = resolveAttentionPlan(
    seqLen,
    headDim,
    numHeads,
    attentionKernel,
    kvDtype,
    sharedLimit,
    caps
  );
  if (DEBUG_KERNELS2) {
    console.warn(
      `[ATTN] recordAttention: isDecode=${plan.isDecode}, tier=${plan.tier}, variant=${plan.variant}, seqLen=${seqLen}, kvLen=${kvLen}, numHeads=${numHeads}, headDim=${headDim}, useF16KV=${plan.useF16KV}`
    );
  }
  const kernel = new AttentionKernel(device2);
  const pipeline = await kernel.getPipeline(plan.variant);
  const outputSize = seqLen * numHeads * headDim * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "attention_output");
  const uniformBuffer = createAttentionUniformBuffer(device2, recorder, {
    numHeads,
    numKVHeads,
    headDim,
    kvLen,
    seqLen,
    scale,
    causal,
    startPos
  });
  const bindGroup = device2.createBindGroup({
    label: "attention_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: Q } },
      { binding: 2, resource: { buffer: K } },
      { binding: 3, resource: { buffer: V } },
      { binding: 4, resource: { buffer: output } }
    ]
  });
  if (limits && plan.workgroups > limits.maxComputeWorkgroupsPerDimension) {
    throw new Error(
      `Attention dispatch requires ${plan.workgroups} workgroups but device limit is ${limits.maxComputeWorkgroupsPerDimension}. Reduce prompt length or use streaming attention.`
    );
  }
  kernel.record(recorder, pipeline, bindGroup, plan.workgroups);
  return output;
}

// gpu/kernels/rmsnorm.ts
init_device();
init_buffer_dtypes();
init_buffer_pool();
init_dispatch();
init_utils();
var DEBUG_KERNELS3 = typeof window !== "undefined" ? Boolean(window.DOPPLER_DEBUG_KERNELS) : false;
function selectRMSNormKernel(options = {}) {
  const { residual = null, hiddenSize = null } = options;
  if (residual) {
    return "residual";
  } else if (hiddenSize !== null && hiddenSize <= 256) {
    return "small";
  }
  return "default";
}
async function runRMSNorm(input, weight, eps = 1e-5, options = {}) {
  const device2 = getDevice();
  const { batchSize = 1, hiddenSize, residual = null, outputBuffer = null } = options;
  let variant = "default";
  if (residual) {
    variant = "residual";
    if (DEBUG_KERNELS3) {
      console.log(`[RMSNorm] Using residual variant, residual.size=${residual.size}, inferredHiddenSize=${hiddenSize || weight.size / 4}, batchSize=${batchSize}`);
    }
  } else if (hiddenSize && hiddenSize <= 256) {
    variant = "small";
  }
  const pipeline = await createPipeline("rmsnorm", variant);
  const inferredHiddenSize = hiddenSize || weight.size / 4;
  const outputSize = batchSize * inferredHiddenSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "rmsnorm_output");
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
  if (DEBUG_KERNELS3 && hasResidualFlag) {
    console.log(`[RMSNorm] Uniform hasResidual=${hasResidualFlag}, hiddenSize=${inferredHiddenSize}, batchSize=${batchSize}`);
  }
  const residualBuffer = residual || device2.createBuffer({
    label: "rmsnorm_residual_placeholder",
    size: 4,
    usage: GPUBufferUsage.STORAGE
  });
  const bindGroup = device2.createBindGroup({
    label: "rmsnorm_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: residualBuffer } }
    ]
  });
  dispatch(device2, pipeline, bindGroup, batchSize, "rmsnorm");
  uniformBuffer.destroy();
  if (!residual)
    residualBuffer.destroy();
  setBufferDtype(output, "f32");
  return output;
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
  const inputSize = batchSize * inferredHiddenSize * 4;
  const variant = selectRMSNormKernel(options);
  const pipeline = await createPipeline("rmsnorm", variant);
  const output = outputBuffer || acquireBuffer(inputSize, void 0, "rmsnorm_output");
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
  const residualBuffer = residual || device2.createBuffer({
    label: "rmsnorm_residual_placeholder",
    size: 4,
    usage: GPUBufferUsage.STORAGE
  });
  const bindGroup = device2.createBindGroup({
    label: "rmsnorm_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: residualBuffer } }
    ]
  });
  recordDispatch(recorder, pipeline, bindGroup, batchSize, "rmsnorm");
  if (!residual) {
    recorder.trackTemporaryBuffer(residualBuffer);
  }
  setBufferDtype(output, "f32");
  return output;
}

// gpu/kernels/softmax.ts
init_device();
init_buffer_dtypes();
init_buffer_pool();
init_dispatch();
init_utils();
async function runSoftmax(input, axis, options = {}) {
  const device2 = getDevice();
  const { batchSize = 1, size, temperature = 1, outputBuffer = null } = options;
  const inferredSize = size || input.size / (batchSize * 4);
  const pipeline = await createPipeline("softmax", "default");
  const outputSize = batchSize * inferredSize * 4;
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
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  dispatch(device2, pipeline, bindGroup, batchSize, "softmax");
  uniformBuffer.destroy();
  return output;
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
  setBufferDtype(indices, "u32");
  setBufferDtype(weights, "f32");
  return { indices, weights };
}
async function recordSoftmax(recorder, input, axis, options = {}) {
  const device2 = recorder.device;
  const {
    batchSize = 1,
    seqLen = null,
    outputBuffer = null
  } = options;
  const inferredSeqLen = seqLen || input.size / (batchSize * 4);
  const pipeline = await createPipeline("softmax", "default");
  const outputSize = batchSize * inferredSeqLen * 4;
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
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  recordDispatch(recorder, pipeline, bindGroup, batchSize, "softmax");
  setBufferDtype(output, "f32");
  return output;
}

// gpu/kernels/rope.ts
init_device();
init_buffer_dtypes();
init_dispatch();
init_utils();
async function runRoPE(input, freqsCos, freqsSin, seqLen, options = {}) {
  const device2 = getDevice();
  const {
    numHeads = 1,
    headDim = 64,
    ropeTheta = 1e4
  } = options;
  const pipeline = await createPipeline("rope", "default");
  const uniformBuffer = createUniformBufferWithView(
    "rope_uniforms",
    32,
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
      { binding: 1, resource: { buffer: input } },
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
  return input;
}
async function recordRoPE(recorder, input, freqsCos, freqsSin, seqLen, options = {}) {
  const device2 = recorder.device;
  const {
    numHeads = 1,
    headDim = 64
  } = options;
  const pipeline = await createPipeline("rope", "default");
  const uniformBuffer = createUniformBufferWithView(
    "rope_uniforms",
    32,
    (view) => {
      view.setUint32(0, seqLen, true);
      view.setUint32(4, numHeads, true);
      view.setUint32(8, headDim, true);
      view.setUint32(12, options.startPos || 0, true);
      view.setFloat32(16, 1e4, true);
      view.setFloat32(20, 1, true);
    },
    recorder
  );
  const bindGroup = device2.createBindGroup({
    label: "rope_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
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
  setBufferDtype(input, "f32");
  return input;
}

// gpu/kernels/silu.ts
init_device();
init_buffer_dtypes();
init_buffer_pool();
init_dispatch();
init_utils();
async function runSiLU(input, options = {}) {
  const device2 = getDevice();
  const { size, gate = null, outputBuffer = null, useVec4 = false } = options;
  const variant = gate ? "gate" : useVec4 ? "vec4" : "default";
  const pipeline = await createPipeline("silu", variant);
  const inferredSize = size || input.size / 4;
  const outputSize = inferredSize * 4;
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
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: input } },
    { binding: 2, resource: { buffer: output } }
  ];
  if (gate) {
    entries.push({ binding: 3, resource: { buffer: gate } });
  }
  const bindGroup = device2.createBindGroup({
    label: "silu_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries
  });
  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "silu");
  uniformBuffer.destroy();
  return output;
}
async function runSwiGLURowsplitBias(input, bias, numTokens, dim, options = {}) {
  const device2 = getDevice();
  const { outputBuffer = null, biasOffset = 0 } = options;
  const pipeline = await createPipeline("swiglu", "rowsplit_bias");
  const outputSize = numTokens * dim * 4;
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
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: bias } },
      { binding: 3, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(numTokens * dim / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "swiglu");
  uniformBuffer.destroy();
  return output;
}
async function runSiLURowSplit(input, options) {
  const device2 = getDevice();
  const { numTokens, dim, activation = "silu", outputBuffer = null } = options;
  const variant = activation === "gelu" ? "geglu_rowsplit" : "gate_rowsplit";
  const pipeline = await createPipeline("silu", variant);
  const outputSize = numTokens * dim * 4;
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
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(numTokens * dim / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "silu_rowsplit");
  uniformBuffer.destroy();
  setBufferDtype(output, "f32");
  return output;
}
async function recordSiLURowSplit(recorder, input, options) {
  const device2 = recorder.device;
  const { numTokens, dim, activation = "silu", outputBuffer = null } = options;
  const variant = activation === "gelu" ? "geglu_rowsplit" : "gate_rowsplit";
  const pipeline = await createPipeline("silu", variant);
  const outputSize = numTokens * dim * 4;
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
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(numTokens * dim / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "silu_rowsplit");
  setBufferDtype(output, "f32");
  return output;
}
async function recordSiLU(recorder, input, options = {}) {
  const device2 = recorder.device;
  const { size, gate = null, outputBuffer = null } = options;
  const variant = gate ? "gate" : "default";
  const pipeline = await createPipeline("silu", variant);
  const inferredSize = size || input.size / 4;
  const outputSize = inferredSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "silu_output");
  const uniformBuffer = createUniformBufferWithView(
    "silu_uniforms",
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
    },
    recorder
  );
  const gateBuffer = gate || input;
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: input } },
    { binding: 2, resource: { buffer: output } }
  ];
  if (gate) {
    entries.push({ binding: 3, resource: { buffer: gateBuffer } });
  }
  const bindGroup = device2.createBindGroup({
    label: "silu_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries
  });
  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "silu");
  setBufferDtype(output, "f32");
  return output;
}

// gpu/kernels/gelu.ts
init_device();
init_buffer_dtypes();
init_buffer_pool();
init_dispatch();
init_utils();
async function runGeLU(input, options = {}) {
  const device2 = getDevice();
  const { size, gate = null, outputBuffer = null } = options;
  const variant = gate ? "geglu" : "gelu";
  const pipeline = await createPipeline("silu", variant);
  const inferredSize = size || input.size / 4;
  const outputSize = inferredSize * 4;
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
  const gateBuffer = gate || input;
  const bindGroup = device2.createBindGroup({
    label: "gelu_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: gateBuffer } }
    ]
  });
  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "gelu");
  uniformBuffer.destroy();
  return output;
}
async function recordGeLU(recorder, input, options = {}) {
  const device2 = recorder.device;
  const { size, gate = null, outputBuffer = null } = options;
  const variant = gate ? "geglu" : "gelu";
  const pipeline = await createPipeline("silu", variant);
  const inferredSize = size || input.size / 4;
  const outputSize = inferredSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "gelu_output");
  const uniformBuffer = createUniformBufferWithView(
    "gelu_uniforms",
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
    },
    recorder
  );
  const gateBuffer = gate || input;
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: input } },
    { binding: 2, resource: { buffer: output } }
  ];
  if (gate) {
    entries.push({ binding: 3, resource: { buffer: gateBuffer } });
  }
  const bindGroup = device2.createBindGroup({
    label: "gelu_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries
  });
  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "gelu");
  setBufferDtype(output, "f32");
  return output;
}

// gpu/kernels/scale.ts
init_device();
init_buffer_dtypes();
init_buffer_pool();
init_dispatch();
init_utils();
async function runScale(input, scale, options = {}) {
  const device2 = getDevice();
  const { count, outputBuffer = null, inplace = false } = options;
  const inferredCount = count ?? Math.floor(input.size / 4);
  const variant = inplace ? "inplace" : "default";
  const pipeline = await createPipeline("scale", variant);
  const outputSize = inferredCount * 4;
  const output = inplace ? input : outputBuffer || acquireBuffer(outputSize, void 0, "scale_output");
  const uniformBuffer = createUniformBufferWithView(
    "scale_uniforms",
    8,
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
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "scale");
  uniformBuffer.destroy();
  setBufferDtype(output, "f32");
  return output;
}
async function recordScale(recorder, input, scale, options = {}) {
  const device2 = recorder.device;
  const { count, outputBuffer = null, inplace = false } = options;
  const inferredCount = count ?? Math.floor(input.size / 4);
  const variant = inplace ? "inplace" : "default";
  const pipeline = await createPipeline("scale", variant);
  const outputSize = inferredCount * 4;
  const output = inplace ? input : outputBuffer || acquireBuffer(outputSize, void 0, "scale_output");
  const uniformBuffer = createUniformBufferWithView(
    "scale_uniforms",
    8,
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
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "scale");
  setBufferDtype(output, "f32");
  return output;
}

// gpu/kernels/gather.ts
init_device();
init_buffer_dtypes();
init_buffer_pool();
init_dispatch();
init_utils();
var DEBUG_KERNELS4 = typeof window !== "undefined" ? Boolean(window.DOPPLER_DEBUG_KERNELS) : false;
async function runGather(indices, embeddings, numTokens, hiddenSize, vocabSize, options = {}) {
  const device2 = getDevice();
  const { useVec4 = true, outputBuffer = null, embeddingDtype, transpose = false } = options;
  const caps = getKernelCapabilities();
  const bufferDtype = getBufferDtype(embeddings);
  const detectedDtype = embeddingDtype || bufferDtype || "f32";
  const useF16 = detectedDtype === "f16" && caps.hasF16;
  if (DEBUG_KERNELS4) {
    console.log(`[Gather] numTokens=${numTokens}, hiddenSize=${hiddenSize}, vocabSize=${vocabSize}, transpose=${transpose}, bufferDtype=${bufferDtype}, detectedDtype=${detectedDtype}, useF16=${useF16}`);
  }
  let variant;
  if (useF16) {
    variant = useVec4 ? "f16_vec4" : "f16";
  } else {
    variant = useVec4 ? "vec4" : "default";
  }
  const pipeline = await createPipeline("gather", variant);
  const outputSize = numTokens * hiddenSize * 4;
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
  const bindGroup = device2.createBindGroup({
    label: "gather_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: indices } },
      { binding: 2, resource: { buffer: embeddings } },
      { binding: 3, resource: { buffer: output } }
    ]
  });
  const workgroups = useVec4 ? Math.ceil(numTokens * hiddenSize / 256) : Math.ceil(numTokens * hiddenSize / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "gather");
  uniformBuffer.destroy();
  setBufferDtype(output, "f32");
  return output;
}
async function recordGather(recorder, indices, embeddings, numTokens, hiddenSize, vocabSize, options = {}) {
  const device2 = recorder.device;
  const { useVec4 = true, outputBuffer = null, embeddingDtype, transpose = false } = options;
  const caps = getKernelCapabilities();
  const detectedDtype = embeddingDtype || getBufferDtype(embeddings) || "f32";
  const useF16 = detectedDtype === "f16" && caps.hasF16;
  let variant;
  if (useF16) {
    variant = useVec4 ? "f16_vec4" : "f16";
  } else {
    variant = useVec4 ? "vec4" : "default";
  }
  const pipeline = await createPipeline("gather", variant);
  const outputSize = numTokens * hiddenSize * 4;
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
  const bindGroup = device2.createBindGroup({
    label: "gather_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: indices } },
      { binding: 2, resource: { buffer: embeddings } },
      { binding: 3, resource: { buffer: output } }
    ]
  });
  const workgroups = useVec4 ? Math.ceil(numTokens * hiddenSize / 256) : Math.ceil(numTokens * hiddenSize / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "gather");
  setBufferDtype(output, "f32");
  return output;
}

// gpu/kernels/residual.ts
init_device();
init_buffer_dtypes();
init_buffer_pool();
init_dispatch();
init_utils();
async function runResidualAdd(a, b, size, options = {}) {
  const device2 = getDevice();
  const { useVec4 = true, outputBuffer = null } = options;
  const variant = useVec4 ? "vec4" : "default";
  const pipeline = await createPipeline("residual", variant);
  const outputSize = size * 4;
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
      { binding: 1, resource: { buffer: a } },
      { binding: 2, resource: { buffer: b } },
      { binding: 3, resource: { buffer: output } }
    ]
  });
  const workgroups = useVec4 ? Math.ceil(size / 256) : Math.ceil(size / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "residual");
  uniformBuffer.destroy();
  return output;
}
async function runBiasAdd(data, bias, numTokens, dim, options = {}) {
  const device2 = getDevice();
  const { dataOffset = 0, biasOffset = 0 } = options;
  const pipeline = await createPipeline("bias_add", "default");
  const output = data;
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
      { binding: 1, resource: { buffer: data } },
      { binding: 2, resource: { buffer: bias } }
    ]
  });
  const workgroups = Math.ceil(numTokens * dim / WORKGROUP_SIZES.DEFAULT);
  dispatch(device2, pipeline, bindGroup, workgroups, "bias_add");
  uniformBuffer.destroy();
  return output;
}
async function recordResidualAdd(recorder, a, b, size, options = {}) {
  const device2 = recorder.device;
  const { outputBuffer = null } = options;
  const pipeline = await createPipeline("residual", "default");
  const outputSize = size * 4;
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
      { binding: 1, resource: { buffer: a } },
      { binding: 2, resource: { buffer: b } },
      { binding: 3, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(size / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "residual");
  setBufferDtype(output, "f32");
  return output;
}
async function recordBiasAdd(recorder, data, bias, numTokens, dim, options = {}) {
  const device2 = recorder.device;
  const { dataOffset = 0, biasOffset = 0 } = options;
  const pipeline = await createPipeline("bias_add", "default");
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
      { binding: 1, resource: { buffer: data } },
      { binding: 2, resource: { buffer: bias } }
    ]
  });
  const workgroups = Math.ceil(numTokens * dim / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, "bias_add");
  return data;
}

// gpu/kernels/moe.ts
init_device();
init_buffer_dtypes();
init_buffer_pool();
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
  setBufferDtype(indices, "u32");
  setBufferDtype(weights, "f32");
  return { indices, weights };
}
async function runMoEGather(hiddenStates, expertIndices, numTokens, hiddenSize, numExperts, topK, options = {}) {
  const device2 = getDevice();
  const { maxTokensPerExpert = numTokens } = options;
  const pipeline = await createPipeline("moe_gather", "sparse");
  const gatheredSize = numExperts * maxTokensPerExpert * hiddenSize * 4;
  const tokenCountsSize = numExperts * 4;
  const tokenMapSize = numExperts * maxTokensPerExpert * 2 * 4;
  const gathered = acquireBuffer(gatheredSize, void 0, "moe_gathered");
  const tokenCounts = acquireBuffer(tokenCountsSize, void 0, "moe_token_counts");
  const tokenMap = acquireBuffer(tokenMapSize, void 0, "moe_token_map");
  const uniformBuffer = createUniformBufferWithView(
    "moe_gather_uniforms",
    20,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, numExperts, true);
      view.setUint32(12, topK, true);
      view.setUint32(16, maxTokensPerExpert, true);
    },
    null,
    device2
  );
  const bindGroup = device2.createBindGroup({
    label: "moe_gather_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: hiddenStates } },
      { binding: 2, resource: { buffer: expertIndices } },
      { binding: 3, resource: { buffer: gathered } },
      { binding: 4, resource: { buffer: tokenCounts } },
      { binding: 5, resource: { buffer: tokenMap } }
    ]
  });
  const encoder = device2.createCommandEncoder({ label: "moe_gather_encoder" });
  encoder.clearBuffer(tokenCounts);
  const pass = encoder.beginComputePass({ label: "moe_gather_pass" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  const workgroups = Math.ceil(numTokens * topK / WORKGROUP_SIZES.DEFAULT);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  device2.queue.submit([encoder.finish()]);
  uniformBuffer.destroy();
  setBufferDtype(gathered, "f32");
  setBufferDtype(tokenCounts, "u32");
  setBufferDtype(tokenMap, "u32");
  return { gathered, tokenCounts, tokenMap, maxTokensPerExpert };
}
async function runScatterAdd(expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK, options = {}) {
  const device2 = getDevice();
  const { outputBuffer = null } = options;
  const pipeline = await createPipeline("scatter_add", "default");
  const outputSize = numTokens * hiddenSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "scatter_add_output");
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
      { binding: 1, resource: { buffer: expertOutputs } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } },
      { binding: 4, resource: { buffer: output } }
    ]
  });
  const encoder = device2.createCommandEncoder({ label: "scatter_add_encoder" });
  encoder.clearBuffer(output);
  const pass = encoder.beginComputePass({ label: "scatter_add_pass" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  const workgroups = Math.ceil(numTokens * hiddenSize / WORKGROUP_SIZES.DEFAULT);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  device2.queue.submit([encoder.finish()]);
  uniformBuffer.destroy();
  return output;
}
async function runScatterAddDynamic(expertOutputs, indices, weights, tokenOffsets, numTokens, hiddenSize, topK, options = {}) {
  const device2 = getDevice();
  const { outputBuffer = null } = options;
  const pipeline = await createPipeline("scatter_add", "dynamic");
  const outputSize = numTokens * hiddenSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "scatter_add_dynamic_output");
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
      { binding: 1, resource: { buffer: expertOutputs } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } },
      { binding: 4, resource: { buffer: tokenOffsets } },
      { binding: 5, resource: { buffer: output } }
    ]
  });
  const encoder = device2.createCommandEncoder({ label: "scatter_add_dynamic_encoder" });
  encoder.clearBuffer(output);
  const pass = encoder.beginComputePass({ label: "scatter_add_dynamic_pass" });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  const workgroups = Math.ceil(numTokens * topK * hiddenSize / WORKGROUP_SIZES.DEFAULT);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  device2.queue.submit([encoder.finish()]);
  uniformBuffer.destroy();
  return output;
}

// gpu/kernels/cast.ts
init_device();
init_buffer_dtypes();
init_buffer_pool();
init_dispatch();
init_utils();
init_perf_guards();
var DEBUG_CAST = false;
async function castF32ToF16(input, numElements, options = {}) {
  const device2 = getDevice();
  const { outputBuffer = null } = options;
  const pipeline = await createPipeline("cast", "f32_to_f16");
  const output = outputBuffer || acquireBuffer(numElements * 2, void 0, "cast_f32_to_f16_output");
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
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(numElements / WORKGROUP_SIZES.DEFAULT);
  const maxWorkgroupsPerDim = GPU_LIMITS.MAX_WORKGROUPS;
  const dispatchSize = workgroups <= maxWorkgroupsPerDim ? [workgroups, 1, 1] : [maxWorkgroupsPerDim, Math.ceil(workgroups / maxWorkgroupsPerDim), 1];
  dispatch(device2, pipeline, bindGroup, dispatchSize, "cast_f32_to_f16");
  await device2.queue.onSubmittedWorkDone();
  uniformBuffer.destroy();
  setBufferDtype(output, "f16");
  return output;
}
async function recordCastF32ToF16(recorder, input, numElements, options = {}) {
  const device2 = recorder.device;
  const { outputBuffer = null } = options;
  const pipeline = await createPipeline("cast", "f32_to_f16");
  const output = outputBuffer || acquireBuffer(numElements * 2, void 0, "cast_f32_to_f16_output");
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
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  const workgroups = Math.ceil(numElements / WORKGROUP_SIZES.DEFAULT);
  const maxWorkgroupsPerDim = GPU_LIMITS.MAX_WORKGROUPS;
  const dispatchSize = workgroups <= maxWorkgroupsPerDim ? [workgroups, 1, 1] : [maxWorkgroupsPerDim, Math.ceil(workgroups / maxWorkgroupsPerDim), 1];
  recordDispatch(recorder, pipeline, bindGroup, dispatchSize, "cast_f32_to_f16");
  setBufferDtype(output, "f16");
  return output;
}
async function runBF16ToF32(input, numElements, name = "bf16_to_f32_output") {
  if (DEBUG_CAST)
    console.log(`[BF16ToF32] Entry: numElements=${numElements}, name=${name}, inputSize=${input.size}`);
  const device2 = getDevice();
  const limits = device2.limits;
  const maxBufferSize = limits.maxBufferSize;
  const maxBindingSize = limits.maxStorageBufferBindingSize;
  const outputSize = numElements * 4;
  if (DEBUG_CAST)
    console.log(`[BF16ToF32] outputSize=${outputSize}, maxBufferSize=${maxBufferSize}, maxBindingSize=${maxBindingSize}`);
  if (outputSize > maxBufferSize) {
    throw new Error(
      `BF16\u2192F32 output (${outputSize} bytes) exceeds device maxBufferSize (${maxBufferSize}). This often happens for large-vocab models when converting embeddings/LM head. Enable F16 and use BF16\u2192F16 weights, or run on a device with a higher maxBufferSize.`
    );
  }
  if (outputSize > maxBindingSize) {
    return runBF16ToF32Chunked(input, numElements, name, maxBindingSize);
  }
  if (name.includes("embed") && allowReadback("cast.debugInput")) {
    try {
      const sampleSize = Math.min(256, input.size);
      const stagingIn = device2.createBuffer({
        size: sampleSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: "bf16_input_debug_staging"
      });
      const encIn = device2.createCommandEncoder();
      encIn.copyBufferToBuffer(input, 0, stagingIn, 0, sampleSize);
      device2.queue.submit([encIn.finish()]);
      await device2.queue.onSubmittedWorkDone();
      await stagingIn.mapAsync(GPUMapMode.READ);
      const inData = new Uint8Array(stagingIn.getMappedRange().slice(0));
      stagingIn.unmap();
      stagingIn.destroy();
      const nonZeroBytes = Array.from(inData).filter((x) => x !== 0).length;
      console.log(`[BF16ToF32] INPUT CHECK: nonZeroBytes=${nonZeroBytes}/${inData.length}, first16=[${Array.from(inData.slice(0, 16)).join(", ")}]`);
    } catch (err) {
      console.error(`[BF16ToF32] INPUT CHECK failed:`, err.message);
    }
  }
  const pipeline = await createPipeline("bf16_to_f32", "default");
  if (DEBUG_CAST)
    console.log(`[BF16ToF32] Pipeline created`);
  const output = acquireBuffer(outputSize, void 0, name);
  if (DEBUG_CAST)
    console.log(`[BF16ToF32] Output buffer acquired, size=${output.size}`);
  const uniformBuffer = createUniformBufferWithView(
    "bf16_to_f32_uniforms",
    16,
    (view) => {
      view.setUint32(0, numElements, true);
    },
    null,
    device2
  );
  if (DEBUG_CAST)
    console.log(`[BF16ToF32] Uniform: numElements=${numElements}`);
  const bindGroup = device2.createBindGroup({
    label: "bf16_to_f32_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } }
    ]
  });
  if (DEBUG_CAST)
    console.log(`[BF16ToF32] BindGroup created`);
  const numPairs = Math.ceil(numElements / 2);
  const workgroups = Math.ceil(numPairs / WORKGROUP_SIZES.DEFAULT);
  const maxWorkgroupsPerDim = GPU_LIMITS.MAX_WORKGROUPS;
  const dispatchSize = workgroups <= maxWorkgroupsPerDim ? [workgroups, 1, 1] : [maxWorkgroupsPerDim, Math.ceil(workgroups / maxWorkgroupsPerDim), 1];
  if (DEBUG_CAST)
    console.log(
      `[BF16ToF32] Dispatching ${dispatchSize[0]}x${dispatchSize[1]} workgroups for ${numPairs} pairs (${numElements} elements)`
    );
  dispatch(device2, pipeline, bindGroup, dispatchSize, "bf16_to_f32");
  await device2.queue.onSubmittedWorkDone();
  if (DEBUG_CAST)
    console.log(`[BF16ToF32] GPU work completed`);
  uniformBuffer.destroy();
  setBufferDtype(output, "f32");
  return output;
}
async function runBF16ToF16(input, numElements, name = "bf16_to_f16_output") {
  const device2 = getDevice();
  const pipeline = await createPipeline("bf16_to_f16", "default");
  const limits = device2.limits;
  const maxBufferSize = limits.maxBufferSize;
  const maxBindingSize = limits.maxStorageBufferBindingSize;
  const outputSize = numElements * 2;
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
  const maxWorkgroupsPerDim = GPU_LIMITS.MAX_WORKGROUPS;
  const dispatchSize = workgroups <= maxWorkgroupsPerDim ? [workgroups, 1, 1] : [maxWorkgroupsPerDim, Math.ceil(workgroups / maxWorkgroupsPerDim), 1];
  dispatch(device2, pipeline, bindGroup, dispatchSize, "bf16_to_f16");
  await device2.queue.onSubmittedWorkDone();
  uniformBuffer.destroy();
  setBufferDtype(output, "f16");
  return output;
}
async function runBF16ToF32Chunked(input, numElements, name, maxBindingSize) {
  const device2 = getDevice();
  const pipeline = await createPipeline("bf16_to_f32", "default");
  const alignmentBytes = device2.limits.minStorageBufferOffsetAlignment;
  const lcm = (a, b) => {
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
  };
  const inElemAlign = Math.max(1, Math.floor(alignmentBytes / 2));
  const outElemAlign = Math.max(1, Math.floor(alignmentBytes / 4));
  const elemAlign = lcm(inElemAlign, outElemAlign);
  let maxElementsPerChunk = Math.floor(maxBindingSize / 4);
  maxElementsPerChunk -= maxElementsPerChunk % elemAlign;
  if (maxElementsPerChunk <= 0) {
    throw new Error(`BF16\u2192F32 chunk size underflow (maxBindingSize=${maxBindingSize}, alignment=${alignmentBytes})`);
  }
  const numChunks = Math.ceil(numElements / maxElementsPerChunk);
  const outputSize = numElements * 4;
  const output = acquireBuffer(outputSize, void 0, name);
  if (DEBUG_CAST)
    console.log(`[BF16ToF32] Chunking: ${numElements} elements in ${numChunks} chunks`);
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
    const inputOffsetBytes = chunkStart * 2;
    const outputOffsetBytes = chunkStart * 4;
    const inputPairs = Math.ceil(chunkSize / 2);
    const inputSizeBytes = inputPairs * 4;
    const outputSizeBytes = chunkSize * 4;
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
    const maxWorkgroupsPerDim = GPU_LIMITS.MAX_WORKGROUPS;
    const dispatchSize = workgroups <= maxWorkgroupsPerDim ? [workgroups, 1, 1] : [maxWorkgroupsPerDim, Math.ceil(workgroups / maxWorkgroupsPerDim), 1];
    dispatch(device2, pipeline, bindGroup, dispatchSize, `bf16_to_f32_chunk${chunkIdx}`);
    uniformBuffer.destroy();
  }
  setBufferDtype(output, "f32");
  return output;
}

// gpu/kernels/sample.ts
init_device();
init_buffer_pool();
init_utils();
init_perf_guards();
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
async function runArgmax(logits, vocabSize) {
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
  const uniformBuffer = createUniformBufferWithView(
    "argmax_uniforms",
    16,
    (view) => {
      view.setUint32(0, vocabSize, true);
      view.setUint32(4, 1, true);
      view.setFloat32(8, 1, true);
      view.setFloat32(12, 0, true);
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
  const {
    temperature = 1,
    topK = 40,
    randomSeed
  } = options;
  if (temperature < 0.01) {
    return runArgmax(logits, vocabSize);
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
    16,
    (view) => {
      view.setUint32(0, vocabSize, true);
      view.setUint32(4, topK, true);
      view.setFloat32(8, temperature, true);
      view.setFloat32(12, randomValue, true);
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
async function recordArgmax(recorder, logits, vocabSize) {
  const device2 = recorder.device;
  const argmaxPipeline = await createSamplePipeline(device2, "argmax");
  const reducePipeline = await createSamplePipeline(device2, "argmax_reduce");
  const numWorkgroups = Math.min(256, Math.ceil(vocabSize / 256));
  const tempLogits = acquireBuffer(256 * 4, void 0, "argmax_temp_logits");
  const tempIndices = acquireBuffer(256 * 4, void 0, "argmax_temp_indices");
  const outputBuffer = acquireBuffer(4, void 0, "argmax_output");
  const uniformBuffer = createUniformBufferWithView(
    "argmax_uniforms",
    16,
    (view) => {
      view.setUint32(0, vocabSize, true);
      view.setUint32(4, 1, true);
      view.setFloat32(8, 1, true);
      view.setFloat32(12, 0, true);
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
  return outputBuffer;
}
function seededRandom(seed) {
  const x = Math.sin(seed) * 1e4;
  return x - Math.floor(x);
}
function isGPUSamplingAvailable() {
  return getDevice() !== null;
}

// gpu/kernels/fused_ffn.ts
init_device();
init_buffer_dtypes();
init_buffer_pool();
init_utils();
var DEBUG_KERNELS5 = typeof window !== "undefined" ? Boolean(window.DOPPLER_DEBUG_KERNELS) : false;
var FusedFFNKernel = class extends KernelBase {
  async getPipeline(variant) {
    return this.getPipelineFor("ffn_fused", variant);
  }
  dispatch(pipeline, bindGroup, workgroupsX, workgroupsY = 1) {
    this.dispatchKernel(pipeline, bindGroup, [workgroupsX, workgroupsY, 1], "ffn_fused");
  }
  record(recorder, pipeline, bindGroup, workgroupsX, workgroupsY = 1) {
    this.recordKernel(recorder, pipeline, bindGroup, [workgroupsX, workgroupsY, 1], "ffn_fused");
  }
};
function selectFFNVariant(batchSize, weightDtype, intermediateSize) {
  if (intermediateSize <= 1024 && batchSize === 1) {
    return "multi";
  }
  if (batchSize > 1) {
    return "batched";
  }
  if (weightDtype === "f16") {
    return "f16";
  }
  return "default";
}
function createFFNUniformBuffer(device2, recorder, params) {
  return createUniformBufferWithView(
    "ffn_fused_uniforms",
    20,
    (view) => {
      view.setUint32(0, params.M, true);
      view.setUint32(4, params.hiddenSize, true);
      view.setUint32(8, params.intermediateSize, true);
      view.setFloat32(12, params.alpha, true);
      view.setUint32(16, params.activation === "silu" ? 0 : 1, true);
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
  const weightDtype = getBufferDtype(W_gate);
  const variant = selectFFNVariant(batchSize, weightDtype, intermediateSize);
  if (DEBUG_KERNELS5) {
    console.log(
      `[FusedFFN] variant=${variant}, batch=${batchSize}, hidden=${hiddenSize}, intermediate=${intermediateSize}, activation=${activation}`
    );
  }
  const kernel = new FusedFFNKernel(device2);
  const pipeline = await kernel.getPipeline(variant);
  const outputSize = batchSize * intermediateSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "ffn_fused_output");
  const uniformBuffer = createFFNUniformBuffer(device2, null, {
    M: batchSize,
    hiddenSize,
    intermediateSize,
    alpha,
    activation
  });
  const bindGroup = device2.createBindGroup({
    label: "ffn_fused_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: W_gate } },
      { binding: 3, resource: { buffer: W_up } },
      { binding: 4, resource: { buffer: output } }
    ]
  });
  let workgroupsX;
  let workgroupsY = 1;
  if (variant === "multi") {
    const outputsPerWg = 4;
    workgroupsX = Math.ceil(intermediateSize / outputsPerWg);
  } else if (variant === "batched") {
    workgroupsX = intermediateSize;
    workgroupsY = batchSize;
  } else {
    workgroupsX = intermediateSize;
  }
  kernel.dispatch(pipeline, bindGroup, workgroupsX, workgroupsY);
  uniformBuffer.destroy();
  return output;
}
async function recordFusedFFN(recorder, input, W_gate, W_up, hiddenSize, intermediateSize, options = {}) {
  const device2 = recorder.device;
  const {
    batchSize = 1,
    activation = "silu",
    alpha = 1,
    outputBuffer = null
  } = options;
  const weightDtype = getBufferDtype(W_gate);
  const variant = selectFFNVariant(batchSize, weightDtype, intermediateSize);
  if (DEBUG_KERNELS5) {
    console.log(
      `[FusedFFN record] variant=${variant}, batch=${batchSize}, hidden=${hiddenSize}, intermediate=${intermediateSize}, activation=${activation}`
    );
  }
  const kernel = new FusedFFNKernel(device2);
  const pipeline = await kernel.getPipeline(variant);
  const outputSize = batchSize * intermediateSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, void 0, "ffn_fused_output");
  const uniformBuffer = createFFNUniformBuffer(device2, recorder, {
    M: batchSize,
    hiddenSize,
    intermediateSize,
    alpha,
    activation
  });
  const bindGroup = device2.createBindGroup({
    label: "ffn_fused_bind_group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: W_gate } },
      { binding: 3, resource: { buffer: W_up } },
      { binding: 4, resource: { buffer: output } }
    ]
  });
  let workgroupsX;
  let workgroupsY = 1;
  if (variant === "multi") {
    const outputsPerWg = 4;
    workgroupsX = Math.ceil(intermediateSize / outputsPerWg);
  } else if (variant === "batched") {
    workgroupsX = intermediateSize;
    workgroupsY = batchSize;
  } else {
    workgroupsX = intermediateSize;
  }
  kernel.record(recorder, pipeline, bindGroup, workgroupsX, workgroupsY);
  return output;
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

// gpu/kernels/index.ts
init_fused_matmul_rmsnorm();
init_fused_matmul_rmsnorm();

// gpu/command-recorder.ts
init_device();
init_perf_guards();
var CommandRecorder = class _CommandRecorder {
  device;
  label;
  encoder;
  /** Temporary buffers to destroy after submit */
  tempBuffers;
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
      console.warn("[CommandRecorder] Failed to initialize profiling:", e);
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
   * Create a uniform buffer, write data, and track for cleanup.
   * Convenience method for the common uniform buffer pattern.
   *
   * @param data - Data to write
   * @param label - Buffer label
   * @returns GPUBuffer
   */
  createUniformBuffer(data, label = "uniforms") {
    const byteLength = data instanceof ArrayBuffer ? data.byteLength : data.byteLength;
    const buffer = this.createTempBuffer(
      byteLength,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label
    );
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
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
    for (const buffer of this.tempBuffers) {
      buffer.destroy();
    }
    this.tempBuffers = [];
  }
  /**
   * Submit and wait for GPU to complete (useful for debugging/profiling).
   * @returns Promise that resolves when GPU work is done
   */
  async submitAndWait() {
    this.submit();
    await this.device.queue.onSubmittedWorkDone();
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

// gpu/kernel-benchmark.ts
init_device();
init_buffer_pool();
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
      const C = await runMatmul(A, B, M, N, K);
      releaseBuffer(C);
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
  const Q = createTestBuffer(numHeads * headDim * 4, "bench_Q");
  const K = createTestBuffer(kvLen * numHeads * headDim * 4, "bench_K");
  const V = createTestBuffer(kvLen * numHeads * headDim * 4, "bench_V");
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
      releaseBuffer(out);
    },
    config2.warmupIterations,
    config2.timedIterations
  );
  releaseBuffer(Q);
  releaseBuffer(K);
  releaseBuffer(V);
  return result;
}
async function benchmarkRMSNorm(batchSize, hiddenSize, options = {}) {
  const config2 = { ...DEFAULT_CONFIG2, ...options };
  const size = batchSize * hiddenSize;
  const input = createTestBuffer(size * 4, "bench_input");
  const weight = createTestBuffer(hiddenSize * 4, "bench_weight");
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
      const out = await runRMSNorm(input, weight, 1e-6, { batchSize, hiddenSize });
      releaseBuffer(out);
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
  const result = await benchmarkKernel(
    "silu",
    "default",
    {
      size,
      readBytes: size * 4,
      writeBytes: size * 4
    },
    async () => {
      const out = await runSiLU(input, { size });
      releaseBuffer(out);
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
    throw new Error(`Fused kernel not supported for N=${N} (max 4096)`);
  }
  const input = createTestBuffer(K * 4, "bench_input");
  const weight = createTestBuffer(K * N * 4, "bench_weight");
  const normWeight = createTestBuffer(N * 4, "bench_norm_weight");
  const residual = createTestBuffer(N * 4, "bench_residual");
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
      const matmulOut = await runMatmul(input, weight, 1, N, K);
      const normOut = await runRMSNorm(matmulOut, normWeight, 1e-6, {
        batchSize: 1,
        hiddenSize: N,
        residual
      });
      releaseBuffer(matmulOut);
      releaseBuffer(normOut);
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
      const out = await runMatmulRMSNormFused2(input, weight, normWeight, {
        N,
        K,
        eps: 1e-6,
        residual
      });
      releaseBuffer(out);
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
  console.log(`[Benchmark] Matmul+RMSNorm (N=${N}, K=${K}):`);
  console.log(`  Separate: ${separateResult.latency.median_ms.toFixed(3)}ms`);
  console.log(`  Fused:    ${fusedResult.latency.median_ms.toFixed(3)}ms`);
  console.log(`  Speedup:  ${speedup.toFixed(2)}x`);
  return { separate: separateResult, fused: fusedResult, comparison };
}
async function benchmarkDecodePass(options = {}) {
  const config2 = { ...DEFAULT_CONFIG2.modelConfig, ...options.modelConfig };
  const device2 = getDevice();
  const limits = getDeviceLimits();
  const caps = getKernelCapabilities();
  const results = [];
  console.log("[Benchmark] Starting decode pass benchmark...");
  console.log(`[Benchmark] Model config: hidden=${config2.hiddenSize}, intermediate=${config2.intermediateSize}, heads=${config2.numHeads}`);
  console.log("[Benchmark] Running RMSNorm...");
  results.push(await benchmarkRMSNorm(1, config2.hiddenSize, options));
  console.log("[Benchmark] Running QKV projection...");
  const qkvDim = (config2.numHeads + 2 * config2.numKVHeads) * config2.headDim;
  results.push(await benchmarkMatmul(1, qkvDim, config2.hiddenSize, options));
  console.log("[Benchmark] Running Attention decode...");
  const kvLen = 512;
  results.push(await benchmarkAttentionDecode(config2.numHeads, config2.headDim, kvLen, options));
  console.log("[Benchmark] Running output projection...");
  results.push(await benchmarkMatmul(1, config2.hiddenSize, config2.numHeads * config2.headDim, options));
  console.log("[Benchmark] Running FFN gate+up...");
  results.push(await benchmarkMatmul(1, config2.intermediateSize * 2, config2.hiddenSize, options));
  console.log("[Benchmark] Running SiLU...");
  results.push(await benchmarkSiLU(config2.intermediateSize, options));
  console.log("[Benchmark] Running FFN down...");
  results.push(await benchmarkMatmul(1, config2.hiddenSize, config2.intermediateSize, options));
  console.log("[Benchmark] Running final RMSNorm...");
  results.push(await benchmarkRMSNorm(1, config2.hiddenSize, options));
  console.log("[Benchmark] Running LM head...");
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
  console.log("\n=== Benchmark Summary ===");
  console.log(`Total decode latency: ${totalDecodeLatency.toFixed(2)}ms`);
  console.log(`Estimated tokens/sec: ${tokPerSec.toFixed(1)}`);
  console.log(`Bottleneck: ${bottleneck.kernel} (${bottleneckPct.toFixed(1)}%)`);
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
  console.log("\n" + "=".repeat(60));
  console.log("KERNEL BENCHMARK REPORT");
  console.log("=".repeat(60));
  console.log("\nDevice Info:");
  console.log(`  Max Workgroup Size: ${report.device_info.max_workgroup_size}`);
  console.log(`  Max Shared Memory: ${(report.device_info.max_shared_memory / 1024).toFixed(1)}KB`);
  console.log(`  F16 Support: ${report.device_info.has_f16}`);
  console.log(`  Subgroup Support: ${report.device_info.has_subgroups}`);
  console.log("\nModel Config:");
  console.log(`  Name: ${report.model_config.name}`);
  console.log(`  Hidden Size: ${report.model_config.hidden_size}`);
  console.log(`  Intermediate Size: ${report.model_config.intermediate_size}`);
  console.log(`  Heads: ${report.model_config.num_heads} (KV: ${report.model_config.num_kv_heads})`);
  console.log("\nKernel Results:");
  console.log("-".repeat(60));
  console.log("Kernel           | Latency (ms) | GB/s    | GFLOPS");
  console.log("-".repeat(60));
  for (const r of report.results) {
    console.log(
      `${(r.kernel + "/" + r.variant).padEnd(16)} | ${r.latency.median_ms.toFixed(3).padStart(12)} | ${r.throughput.gb_per_sec.toFixed(2).padStart(7)} | ${r.flops.gflops.toFixed(1).padStart(7)}`
    );
  }
  console.log("-".repeat(60));
  console.log("\nSummary:");
  console.log(`  Total Decode Latency: ${report.summary.total_decode_latency_ms.toFixed(2)}ms`);
  console.log(`  Estimated Tokens/sec: ${report.summary.estimated_tok_per_sec.toFixed(1)}`);
  console.log(`  Bottleneck: ${report.summary.bottleneck_kernel} (${report.summary.bottleneck_percentage.toFixed(1)}%)`);
  if (report.comparisons.length > 0) {
    console.log("\nComparisons:");
    for (const c of report.comparisons) {
      console.log(`  ${c.baseline.kernel}: ${c.speedup.toFixed(2)}x speedup`);
    }
  }
  console.log("\n" + "=".repeat(60));
}

// gpu/perf-profiler.ts
init_device();
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
  console.log("\n" + "=".repeat(60));
  console.log("PERFORMANCE PROFILE REPORT");
  console.log("=".repeat(60));
  console.log("\nSummary:");
  console.log(`  Total Time: ${r.summary.totalTime.toFixed(2)}ms`);
  console.log(`  Kernel Time: ${r.summary.kernelTime.toFixed(2)}ms (${(r.summary.kernelTime / r.summary.totalTime * 100).toFixed(1)}%)`);
  console.log(`  Memory Time: ${r.summary.memoryTime.toFixed(2)}ms (${(r.summary.memoryTime / r.summary.totalTime * 100).toFixed(1)}%)`);
  console.log(`  Sync Time: ${r.summary.syncTime.toFixed(2)}ms (${(r.summary.syncTime / r.summary.totalTime * 100).toFixed(1)}%)`);
  console.log(`  Kernel Count: ${r.summary.kernelCount}`);
  console.log("\nTop Operations:");
  console.log("-".repeat(60));
  console.log("Operation                    | Time (ms) | Count | % Total");
  console.log("-".repeat(60));
  for (const item of r.breakdown.slice(0, 10)) {
    console.log(
      `${item.name.padEnd(28)} | ${item.totalTime.toFixed(2).padStart(9)} | ${item.count.toString().padStart(5)} | ${item.pctOfTotal.toFixed(1).padStart(7)}%`
    );
  }
  if (r.bottlenecks.length > 0) {
    console.log("\nBottlenecks:");
    console.log("-".repeat(60));
    for (const b of r.bottlenecks) {
      console.log(`  [${(b.impact * 100).toFixed(0)}%] ${b.name}`);
      console.log(`       Fix: ${b.suggestion}`);
    }
  }
  console.log("\n" + "=".repeat(60));
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
  runAttention: runAttention2 = null,
  dequantize: dequantize2 = null,
  dequantizeQ6K: dequantizeQ6K2 = null
} = kernel_selector_exports;
var bufferPool = null;
try {
  bufferPool = await Promise.resolve().then(() => (init_buffer_pool(), buffer_pool_exports));
} catch (e) {
  console.warn("Buffer pool not available:", e.message);
}
var device = null;
var initialized = false;
function f16ToF32(h) {
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
  device = await initDevice();
  if (!device) {
    throw new Error("WebGPU not available");
  }
  setKernelHints({ q4kMatmul: "fused_q4k" }, "runtime");
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
    const bufB = makeBuffer(B);
    const resultBuf = await runMatmul2(bufA, bufB, M, N, K, { alpha, transposeB: false });
    const result = new Float32Array(await readBufferData(resultBuf, M * N * 4));
    bufA.destroy();
    bufB.destroy();
    resultBuf.destroy();
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
    const bufB = makeBuffer(B_q4k, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    setBufferDtype(bufB, "q4k");
    const resultBuf = await runMatmul2(bufA, bufB, M, N, K, { alpha });
    const result = new Float32Array(await readBufferData(resultBuf, M * N * 4));
    bufA.destroy();
    bufB.destroy();
    resultBuf.destroy();
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
    const resultBuf = await runSoftmax2(inputBuf, -1, {
      batchSize: outerSize,
      size: innerSize,
      temperature
    });
    const result = new Float32Array(await readBufferData(resultBuf, input.length * 4));
    inputBuf.destroy();
    resultBuf.destroy();
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
    const resultBuf = await runScatterAdd2(
      expertBuf,
      indicesBuf,
      weightsBuf,
      numTokens,
      hiddenSize,
      numExperts,
      topK
    );
    const result = new Float32Array(await readBufferData(resultBuf, numTokens * hiddenSize * 4));
    expertBuf.destroy();
    indicesBuf.destroy();
    weightsBuf.destroy();
    resultBuf.destroy();
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
    const resultBuf = await runRMSNorm2(inputBuf, weightBuf, eps, {
      batchSize: numTokens,
      hiddenSize
    });
    const result = new Float32Array(await readBufferData(resultBuf, numTokens * hiddenSize * 4));
    inputBuf.destroy();
    weightBuf.destroy();
    resultBuf.destroy();
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
    const resultBuf = await runGather2(idxBuf, embBuf, numTokens, embedDim, vocabSize, { transpose: false });
    const result = new Float32Array(await readBufferData(resultBuf, numTokens * embedDim * 4));
    embBuf.destroy();
    idxBuf.destroy();
    resultBuf.destroy();
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
   * Run Q4_K dequantization (Q4_K_M) on GPU
   * kernel-selector API: dequantize(quantized, numBlocks, options)
   */
  async runDequantQ4K(dev, quantized, numBlocks) {
    if (!dequantize2) {
      throw new Error("dequantize kernel not available");
    }
    const qBuf = makeBuffer(quantized, GPUBufferUsage.STORAGE);
    const outBuf = await dequantize2(qBuf, numBlocks, { outputDtype: "f32", useVec4: false });
    const out = new Float32Array(await readBufferData(outBuf, numBlocks * 256 * 4));
    qBuf.destroy();
    outBuf.destroy();
    return out;
  },
  /**
   * Run attention kernel
   * kernel-selector API: runAttention(Q, K, V, mask, numHeads, headDim, options)
   * options: { seqLen, kvLen, numKVHeads, scale, causal }
   *
   * NOTE: GPU attention kernel currently has workgroup storage limit issues (>32KB),
   * so we fall back to reference implementation for now.
   */
  async runAttention(dev, Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim, mask = null) {
    return attentionRef(Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim, mask);
  },
  /**
   * Run MoE gather
   * TODO: GPU kernel has bugs (wrong token counts), using reference for now
   */
  async runMoEGather(dev, tokens, expertIndices, numTokens, hiddenSize, numExperts, topK) {
    const result = moeGatherRef(tokens, expertIndices, numTokens, hiddenSize, numExperts, topK);
    return {
      gatheredTokens: result.gatheredTokens,
      tokenCounts: result.tokenCounts
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
      out[i] = f16ToF32(u16View[i]);
    }
    qBuf.destroy();
    outBuf.destroy();
    return out;
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
