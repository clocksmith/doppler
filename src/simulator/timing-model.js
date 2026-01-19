
import { log } from '../debug/index.js';

// =============================================================================
// Constants
// =============================================================================

const MODULE = 'TimingModel';

const DEFAULT_COMPUTE_EFFICIENCY = 0.7;

const DEFAULT_MEMORY_EFFICIENCY = 0.8;

const DTYPE_FLOPS_MULTIPLIER = {
  'f16': 1.0,
  'f32': 0.5,     // F32 is typically half the F16 FLOPS
  'fp8': 2.0,     // FP8 is typically 2x F16 FLOPS on B200
  'bf16': 1.0,
};

// =============================================================================
// Utility Functions
// =============================================================================

export function calculateTheoreticalFlops(spec, dtype = 'f16') {
  const baseFlops = spec.fp16Tflops * 1e12; // Convert TFLOPS to FLOPS

  if (dtype === 'fp8' && spec.fp8Tflops) {
    return spec.fp8Tflops * 1e12;
  }

  const multiplier = DTYPE_FLOPS_MULTIPLIER[dtype] || 1.0;
  return baseFlops * multiplier;
}

export function flopsToTimeMs(flops, theoreticalFlops, efficiency = DEFAULT_COMPUTE_EFFICIENCY) {
  const effectiveFlops = theoreticalFlops * efficiency;
  return (flops / effectiveFlops) * 1000; // Convert to ms
}

function sleep(ms) {
  if (ms <= 0) return Promise.resolve();
  return new Promise(resolve => setTimeout(resolve, ms));
}

// =============================================================================
// Timing Model
// =============================================================================

export class TimingModel {
  constructor(config) {
    this.gpuSpec = config.gpuSpec;

    this.mode = config.timingMode;

    this.scaling = { ...config.timingScaling };

    this._nvlinkBandwidth = config.nvlink.bandwidthBytesPerSec;

    this._nvlinkLatencyUs = config.nvlink.latencyUs;

    this._totalInjectedDelayMs = 0;

    log.verbose(MODULE, `Created timing model: mode=${this.mode}, GPU=${this.gpuSpec.name}`);
  }

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

  _computeMemoryTime(sizeBytes, bandwidth) {
    const bw = bandwidth || this.gpuSpec.hbmBandwidthBytesPerSec;
    const effectiveBw = bw * DEFAULT_MEMORY_EFFICIENCY;
    return (sizeBytes / effectiveBw) * 1000 * this.scaling.memoryScale;
  }

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

  computeReduceScatterTimeMs(sizeBytes, gpuCount) {
    // Same complexity as all-gather
    return this.computeAllGatherTimeMs(sizeBytes / gpuCount, gpuCount);
  }

  computeEmbeddingTimeMs(vocabSize, embeddingDim, tokenCount) {
    // Memory-bound operation: read tokenCount embeddings
    const bytesRead = tokenCount * embeddingDim * 2; // F16
    return this.computeMemoryTimeMs(bytesRead);
  }

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

  getTotalInjectedDelayMs() {
    return this._totalInjectedDelayMs;
  }

  reset() {
    this._totalInjectedDelayMs = 0;
  }
}

// =============================================================================
// Factory Function
// =============================================================================

export function createTimingModel(config) {
  return new TimingModel(config);
}
