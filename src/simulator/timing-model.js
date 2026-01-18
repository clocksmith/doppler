/**
 * Timing Model for NVIDIA Superchip Simulation
 *
 * Provides delay injection to simulate realistic compute and memory
 * timing for GH200/GB200 superchips based on theoretical peak performance.
 *
 * @module simulator/timing-model
 */

import { log } from '../debug/index.js';

// =============================================================================
// Constants
// =============================================================================

const MODULE = 'TimingModel';

/** Default efficiency for compute operations (70% of theoretical peak) */
const DEFAULT_COMPUTE_EFFICIENCY = 0.7;

/** Default memory efficiency (80% of theoretical bandwidth) */
const DEFAULT_MEMORY_EFFICIENCY = 0.8;

/** FLOPS per operation for different dtypes */
const DTYPE_FLOPS_MULTIPLIER = {
  'f16': 1.0,
  'f32': 0.5,     // F32 is typically half the F16 FLOPS
  'fp8': 2.0,     // FP8 is typically 2x F16 FLOPS on B200
  'bf16': 1.0,
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Calculate theoretical FLOPS for a GPU spec
 * @param {import('../config/schema/emulation.schema.js').EmulatedGPUSpec} spec
 * @param {string} [dtype='f16']
 * @returns {number} FLOPS
 */
export function calculateTheoreticalFlops(spec, dtype = 'f16') {
  const baseFlops = spec.fp16Tflops * 1e12; // Convert TFLOPS to FLOPS

  if (dtype === 'fp8' && spec.fp8Tflops) {
    return spec.fp8Tflops * 1e12;
  }

  const multiplier = DTYPE_FLOPS_MULTIPLIER[dtype] || 1.0;
  return baseFlops * multiplier;
}

/**
 * Convert FLOPS to time
 * @param {number} flops - Operation count
 * @param {number} theoreticalFlops - Peak FLOPS
 * @param {number} [efficiency=0.7] - Utilization efficiency
 * @returns {number} Time in ms
 */
export function flopsToTimeMs(flops, theoreticalFlops, efficiency = DEFAULT_COMPUTE_EFFICIENCY) {
  const effectiveFlops = theoreticalFlops * efficiency;
  return (flops / effectiveFlops) * 1000; // Convert to ms
}

/**
 * Sleep for a given duration
 * @param {number} ms - Duration in milliseconds
 * @returns {Promise<void>}
 */
function sleep(ms) {
  if (ms <= 0) return Promise.resolve();
  return new Promise(resolve => setTimeout(resolve, ms));
}

// =============================================================================
// Timing Model
// =============================================================================

/**
 * Timing model for emulated NVIDIA hardware
 */
export class TimingModel {
  /**
   * @param {import('../config/schema/emulation.schema.js').EmulationConfigSchema} config
   */
  constructor(config) {
    /** @type {import('../config/schema/emulation.schema.js').EmulatedGPUSpec} */
    this.gpuSpec = config.gpuSpec;

    /** @type {'functional'|'timed'|'hybrid'} */
    this.mode = config.timingMode;

    /** @type {{computeScale: number, memoryScale: number, nvlinkScale: number}} */
    this.scaling = { ...config.timingScaling };

    /** @type {number} */
    this._nvlinkBandwidth = config.nvlink.bandwidthBytesPerSec;

    /** @type {number} */
    this._nvlinkLatencyUs = config.nvlink.latencyUs;

    /** @type {number} */
    this._totalInjectedDelayMs = 0;

    log.verbose(MODULE, `Created timing model: mode=${this.mode}, GPU=${this.gpuSpec.name}`);
  }

  /**
   * Calculate time for matrix multiplication
   * @param {number} M - Output rows
   * @param {number} N - Output columns
   * @param {number} K - Inner dimension
   * @param {string} [dtype='f16'] - Data type
   * @param {number} [gpuCount=1] - Number of GPUs
   * @returns {import('./timing-model.js').TimingResult}
   */
  computeMatmulTimeMs(M, N, K, dtype = 'f16', gpuCount = 1) {
    // FLOPS = 2 * M * N * K (multiply-add)
    const totalFlops = 2 * M * N * K;

    // Divide by GPU count for tensor parallelism
    const flopsPerGpu = totalFlops / gpuCount;

    const theoreticalFlops = calculateTheoreticalFlops(this.gpuSpec, dtype);
    const computeMs = flopsToTimeMs(flopsPerGpu, theoreticalFlops) * this.scaling.computeScale;

    // Memory access: read A (M*K) + B (K*N), write C (M*N)
    const bytesPerElement = dtype === 'f32' ? 4 : 2;
    const memoryBytes = (M * K + K * N + M * N) * bytesPerElement;
    const memoryMs = this._computeMemoryTime(memoryBytes);

    // Total time is max of compute and memory (overlapped)
    const timeMs = Math.max(computeMs, memoryMs);

    return {
      timeMs,
      breakdown: {
        computeMs,
        memoryMs,
        communicationMs: 0,
      },
    };
  }

  /**
   * Calculate time for attention operation
   * @param {number} seqLen - Sequence length
   * @param {number} numHeads - Number of attention heads
   * @param {number} headDim - Dimension per head
   * @param {number} [batchSize=1] - Batch size
   * @returns {import('./timing-model.js').TimingResult}
   */
  computeAttentionTimeMs(seqLen, numHeads, headDim, batchSize = 1) {
    // Attention FLOPS:
    // QK^T: 2 * batch * heads * seq * seq * headDim
    // Softmax: ~5 * batch * heads * seq * seq
    // AV: 2 * batch * heads * seq * seq * headDim
    const qkFlops = 2 * batchSize * numHeads * seqLen * seqLen * headDim;
    const softmaxFlops = 5 * batchSize * numHeads * seqLen * seqLen;
    const avFlops = 2 * batchSize * numHeads * seqLen * seqLen * headDim;
    const totalFlops = qkFlops + softmaxFlops + avFlops;

    const theoreticalFlops = calculateTheoreticalFlops(this.gpuSpec, 'f16');
    const computeMs = flopsToTimeMs(totalFlops, theoreticalFlops) * this.scaling.computeScale;

    // Memory: Q, K, V input + output
    const memoryBytes = 4 * batchSize * numHeads * seqLen * headDim * 2; // F16
    const memoryMs = this._computeMemoryTime(memoryBytes);

    const timeMs = Math.max(computeMs, memoryMs);

    return {
      timeMs,
      breakdown: {
        computeMs,
        memoryMs,
        communicationMs: 0,
      },
    };
  }

  /**
   * Calculate time for memory transfer
   * @param {number} sizeBytes - Transfer size
   * @param {number} [bandwidthBytesPerSec] - Bandwidth
   * @returns {import('./timing-model.js').TimingResult}
   */
  computeMemoryTimeMs(sizeBytes, bandwidthBytesPerSec) {
    const bandwidth = bandwidthBytesPerSec || this.gpuSpec.hbmBandwidthBytesPerSec;
    const timeMs = this._computeMemoryTime(sizeBytes, bandwidth);

    return {
      timeMs,
      breakdown: {
        computeMs: 0,
        memoryMs: timeMs,
        communicationMs: 0,
      },
    };
  }

  /**
   * Calculate memory time (internal helper)
   * @param {number} sizeBytes
   * @param {number} [bandwidth]
   * @returns {number}
   */
  _computeMemoryTime(sizeBytes, bandwidth) {
    const bw = bandwidth || this.gpuSpec.hbmBandwidthBytesPerSec;
    const effectiveBw = bw * DEFAULT_MEMORY_EFFICIENCY;
    return (sizeBytes / effectiveBw) * 1000 * this.scaling.memoryScale;
  }

  /**
   * Calculate time for NVLink transfer
   * @param {number} sizeBytes - Transfer size
   * @param {number} srcGpu - Source GPU
   * @param {number} dstGpu - Destination GPU
   * @param {number} [nvlinkBandwidth] - NVLink bandwidth
   * @returns {import('./timing-model.js').TimingResult}
   */
  computeNvlinkTimeMs(sizeBytes, srcGpu, dstGpu, nvlinkBandwidth) {
    const bandwidth = nvlinkBandwidth || this._nvlinkBandwidth;
    const effectiveBw = bandwidth * DEFAULT_MEMORY_EFFICIENCY;

    // Transfer time + base latency
    const transferMs = (sizeBytes / effectiveBw) * 1000;
    const latencyMs = this._nvlinkLatencyUs / 1000;

    const timeMs = (transferMs + latencyMs) * this.scaling.nvlinkScale;

    return {
      timeMs,
      breakdown: {
        computeMs: 0,
        memoryMs: 0,
        communicationMs: timeMs,
      },
    };
  }

  /**
   * Calculate time for all-reduce operation
   * Ring all-reduce: 2 * (n-1) / n * data_size / bandwidth
   * @param {number} sizeBytes - Buffer size
   * @param {number} gpuCount - Number of GPUs
   * @returns {import('./timing-model.js').TimingResult}
   */
  computeAllReduceTimeMs(sizeBytes, gpuCount) {
    if (gpuCount <= 1) {
      return { timeMs: 0 };
    }

    // Ring all-reduce formula
    const factor = 2 * (gpuCount - 1) / gpuCount;
    const effectiveBw = this._nvlinkBandwidth * DEFAULT_MEMORY_EFFICIENCY;
    const transferMs = (factor * sizeBytes / effectiveBw) * 1000;

    // Add latency per step (2 * (n-1) steps)
    const steps = 2 * (gpuCount - 1);
    const latencyMs = steps * (this._nvlinkLatencyUs / 1000);

    const timeMs = (transferMs + latencyMs) * this.scaling.nvlinkScale;

    return {
      timeMs,
      breakdown: {
        computeMs: 0,
        memoryMs: 0,
        communicationMs: timeMs,
      },
    };
  }

  /**
   * Calculate time for all-gather operation
   * @param {number} sizeBytes - Per-GPU buffer size
   * @param {number} gpuCount - Number of GPUs
   * @returns {import('./timing-model.js').TimingResult}
   */
  computeAllGatherTimeMs(sizeBytes, gpuCount) {
    if (gpuCount <= 1) {
      return { timeMs: 0 };
    }

    // Ring all-gather: (n-1) / n * total_data / bandwidth
    const totalSize = sizeBytes * gpuCount;
    const factor = (gpuCount - 1) / gpuCount;
    const effectiveBw = this._nvlinkBandwidth * DEFAULT_MEMORY_EFFICIENCY;
    const transferMs = (factor * totalSize / effectiveBw) * 1000;

    const steps = gpuCount - 1;
    const latencyMs = steps * (this._nvlinkLatencyUs / 1000);

    const timeMs = (transferMs + latencyMs) * this.scaling.nvlinkScale;

    return {
      timeMs,
      breakdown: {
        computeMs: 0,
        memoryMs: 0,
        communicationMs: timeMs,
      },
    };
  }

  /**
   * Calculate time for reduce-scatter operation
   * @param {number} sizeBytes - Total buffer size
   * @param {number} gpuCount - Number of GPUs
   * @returns {import('./timing-model.js').TimingResult}
   */
  computeReduceScatterTimeMs(sizeBytes, gpuCount) {
    // Same complexity as all-gather
    return this.computeAllGatherTimeMs(sizeBytes / gpuCount, gpuCount);
  }

  /**
   * Calculate time for embedding lookup
   * @param {number} vocabSize - Vocabulary size
   * @param {number} embeddingDim - Embedding dimension
   * @param {number} tokenCount - Number of tokens
   * @returns {import('./timing-model.js').TimingResult}
   */
  computeEmbeddingTimeMs(vocabSize, embeddingDim, tokenCount) {
    // Memory-bound operation: read tokenCount embeddings
    const bytesRead = tokenCount * embeddingDim * 2; // F16
    return this.computeMemoryTimeMs(bytesRead);
  }

  /**
   * Calculate time for RMSNorm
   * @param {number} seqLen - Sequence length
   * @param {number} hiddenDim - Hidden dimension
   * @returns {import('./timing-model.js').TimingResult}
   */
  computeRmsNormTimeMs(seqLen, hiddenDim) {
    // FLOPS: seqLen * hiddenDim * ~8 (square, sum, sqrt, multiply, add)
    const flops = seqLen * hiddenDim * 8;
    const theoreticalFlops = calculateTheoreticalFlops(this.gpuSpec, 'f16');
    const computeMs = flopsToTimeMs(flops, theoreticalFlops) * this.scaling.computeScale;

    // Memory: read input + weight, write output
    const memoryBytes = seqLen * hiddenDim * 2 * 3; // F16, 3 tensors
    const memoryMs = this._computeMemoryTime(memoryBytes);

    const timeMs = Math.max(computeMs, memoryMs);

    return {
      timeMs,
      breakdown: {
        computeMs,
        memoryMs,
        communicationMs: 0,
      },
    };
  }

  /**
   * Inject delay based on timing calculation
   * @param {number} timeMs - Time to delay
   */
  async injectDelay(timeMs) {
    if (this.mode === 'functional') {
      return; // No delay in functional mode
    }

    if (this.mode === 'hybrid') {
      // In hybrid mode, only delay for communication
      return;
    }

    if (timeMs > 0) {
      this._totalInjectedDelayMs += timeMs;
      await sleep(timeMs);
    }
  }

  /**
   * Get total injected delay
   * @returns {number}
   */
  getTotalInjectedDelayMs() {
    return this._totalInjectedDelayMs;
  }

  /**
   * Reset timing statistics
   */
  reset() {
    this._totalInjectedDelayMs = 0;
  }
}

// =============================================================================
// Factory Function
// =============================================================================

/**
 * Create a timing model from emulation config
 * @param {import('../config/schema/emulation.schema.js').EmulationConfigSchema} config
 * @returns {TimingModel}
 */
export function createTimingModel(config) {
  return new TimingModel(config);
}
