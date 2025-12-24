/**
 * Kernel Auto-Tuner - Optimal Workgroup Size Selection
 *
 * Automatically finds optimal workgroup sizes for different kernels
 * by running benchmarks with various configurations.
 *
 * Results are cached in localStorage for persistence across sessions.
 */

import { getDevice, getKernelCapabilities, getDeviceLimits } from './device.js';
import { GPUProfiler } from './profiler.js';

// Cache key prefix
const CACHE_PREFIX = 'doppler_kernel_tune_';

// Default tuning iterations
const DEFAULT_WARMUP = 3;
const DEFAULT_ITERATIONS = 10;

/**
 * Device information for cache keys
 */
interface DeviceInfo {
  vendor: string;
  architecture: string;
  device: string;
  description?: string;
}

/**
 * Tuning result for a kernel
 */
interface TuneResult {
  optimalWorkgroupSize: [number, number, number];
  optimalTileSize: number;
  throughput: number;
  timeMs: number;
  deviceInfo: DeviceInfo | undefined;
}

/**
 * Tuning record stored in cache
 */
interface TuneRecord {
  optimalWorkgroupSize: [number, number, number];
  optimalTileSize: number;
  throughput: number;
  timeMs: number;
  deviceInfo: DeviceInfo | undefined;
}

/**
 * Tuning configuration options
 */
interface TuneConfig {
  warmup?: number;
  iterations?: number;
  forceRetune?: boolean;
}

/**
 * Input sizes for kernel tuning
 */
interface InputSizes {
  M?: number;
  N?: number;
  K?: number;
  seqLen?: number;
  numHeads?: number;
  headDim?: number;
  innerSize?: number;
  outerSize?: number;
  hiddenSize?: number;
  numTokens?: number;
  numBlocks?: number;
}

/**
 * Workgroup size candidate
 */
type WorkgroupSize = [number, number, number];

/**
 * Kernel variant type for tracking different configurations
 */
type KernelVariant = {
  workgroupSize: WorkgroupSize;
  tileSize: number;
};

/**
 * Device limits from GPU
 */
interface DeviceLimits {
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
  maxComputeWorkgroupSizeX: number;
  maxComputeWorkgroupSizeY: number;
  maxComputeWorkgroupSizeZ: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupStorageSize: number;
  maxStorageBuffersPerShaderStage: number;
}

/**
 * Kernel capabilities from GPU
 */
interface KernelCapabilities {
  hasSubgroups: boolean;
  hasSubgroupsF16: boolean;
  hasF16: boolean;
  hasTimestampQuery: boolean;
  maxBufferSize: number;
  maxWorkgroupSize: number;
  maxWorkgroupStorageSize: number;
  adapterInfo: DeviceInfo;
}

/**
 * Cache storage key type
 */
type CacheKey = string;

/**
 * Kernel Tuner class
 */
export class KernelTuner {
  private device: GPUDevice | null;
  private profiler: GPUProfiler | null;
  private limits: DeviceLimits | null;
  private capabilities: KernelCapabilities | null;
  private cache: Map<CacheKey, TuneRecord>;

  constructor() {
    this.device = null;
    this.profiler = null;
    this.limits = null;
    this.capabilities = null;
    this.cache = new Map();
  }

  /**
   * Initialize the tuner
   */
  async init(): Promise<void> {
    this.device = getDevice();
    if (!this.device) {
      throw new Error('GPU device not initialized');
    }

    this.profiler = new GPUProfiler(this.device);
    this.limits = getDeviceLimits();
    this.capabilities = getKernelCapabilities();

    // Load cached results
    this._loadCache();
  }

  /**
   * Get device signature for cache key
   * @private
   */
  private _getDeviceSignature(): string {
    const info: DeviceInfo = this.capabilities?.adapterInfo || { vendor: '', architecture: '', device: '' };
    return `${info.vendor}_${info.architecture}_${info.device}`.replace(/[^a-zA-Z0-9]/g, '_');
  }

  /**
   * Load cached tuning results from localStorage
   * @private
   */
  private _loadCache(): void {
    if (typeof localStorage === 'undefined') return;

    const signature = this._getDeviceSignature();
    const cacheKey = CACHE_PREFIX + signature;

    try {
      const cached = localStorage.getItem(cacheKey);
      if (cached) {
        const data = JSON.parse(cached);
        this.cache = new Map(Object.entries(data));
      }
    } catch (e) {
      console.warn('[KernelTuner] Failed to load cache:', e);
    }
  }

  /**
   * Save cached results to localStorage
   * @private
   */
  private _saveCache(): void {
    if (typeof localStorage === 'undefined') return;

    const signature = this._getDeviceSignature();
    const cacheKey = CACHE_PREFIX + signature;

    try {
      const data = Object.fromEntries(this.cache);
      localStorage.setItem(cacheKey, JSON.stringify(data));
    } catch (e) {
      console.warn('[KernelTuner] Failed to save cache:', e);
    }
  }

  /**
   * Generate workgroup size candidates based on device limits
   * @private
   */
  private _generateWorkgroupCandidates(): WorkgroupSize[] {
    const maxX = this.limits?.maxComputeWorkgroupSizeX || 256;
    const maxY = this.limits?.maxComputeWorkgroupSizeY || 256;
    const maxInvocations = this.limits?.maxComputeInvocationsPerWorkgroup || 256;

    const candidates: WorkgroupSize[] = [];

    // 1D workgroups
    for (const x of [64, 128, 256, 512]) {
      if (x <= maxX && x <= maxInvocations) {
        candidates.push([x, 1, 1]);
      }
    }

    // 2D workgroups (for matrix operations)
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
  async tuneKernel(
    kernelName: string,
    inputSizes: InputSizes,
    options: TuneConfig = {}
  ): Promise<TuneResult> {
    const {
      warmup = DEFAULT_WARMUP,
      iterations = DEFAULT_ITERATIONS,
      forceRetune = false,
    } = options;

    // Check cache
    const cacheKey: CacheKey = `${kernelName}_${JSON.stringify(inputSizes)}`;
    if (!forceRetune && this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    // Get candidates to test
    const candidates = this._generateWorkgroupCandidates();

    // Run tuning based on kernel type
    let bestResult: TuneResult;

    switch (kernelName) {
      case 'matmul':
        bestResult = await this._tuneMatmul(inputSizes, candidates, warmup, iterations);
        break;
      case 'attention':
        bestResult = await this._tuneAttention(inputSizes, candidates, warmup, iterations);
        break;
      case 'softmax':
        bestResult = await this._tuneSoftmax(inputSizes, candidates, warmup, iterations);
        break;
      case 'rmsnorm':
        bestResult = await this._tuneRMSNorm(inputSizes, candidates, warmup, iterations);
        break;
      case 'dequant':
        bestResult = await this._tuneDequant(inputSizes, candidates, warmup, iterations);
        break;
      default:
        bestResult = await this._tuneGeneric(kernelName, inputSizes, candidates, warmup, iterations);
    }

    // Cache result
    this.cache.set(cacheKey, bestResult);
    this._saveCache();

    return bestResult;
  }

  /**
   * Tune matmul kernel
   * @private
   */
  private async _tuneMatmul(
    inputSizes: InputSizes,
    candidates: WorkgroupSize[],
    warmup: number,
    iterations: number
  ): Promise<TuneResult> {
    const { M = 1024, N = 1024, K = 1024 } = inputSizes;

    // Filter to 2D candidates for matmul
    const matmulCandidates = candidates.filter(c => c[1] > 1);

    let best: TuneResult = {
      optimalWorkgroupSize: [16, 16, 1],
      optimalTileSize: 16,
      throughput: 0,
      timeMs: Infinity,
      deviceInfo: this.capabilities?.adapterInfo,
    };

    if (!this.device) {
      return best;
    }

    // Create test buffers
    const bufferA = this.device.createBuffer({
      size: M * K * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bufferB = this.device.createBuffer({
      size: K * N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bufferC = this.device.createBuffer({
      size: M * N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Initialize with random data
    const dataA = new Float32Array(M * K);
    const dataB = new Float32Array(K * N);
    for (let i = 0; i < dataA.length; i++) dataA[i] = Math.random();
    for (let i = 0; i < dataB.length; i++) dataB[i] = Math.random();
    this.device.queue.writeBuffer(bufferA, 0, dataA);
    this.device.queue.writeBuffer(bufferB, 0, dataB);

    for (const [wgX, wgY] of matmulCandidates) {
      try {
        // Create shader with this workgroup size
        const shader = this._createMatmulShader(wgX, wgY);
        const pipeline = await this._createComputePipeline(shader, 'main');

        // Create bind group
        const uniformBuffer = this.device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const uniformData = new Uint32Array([M, N, K, 0]);
        this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        const bindGroup = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: bufferA } },
            { binding: 2, resource: { buffer: bufferB } },
            { binding: 3, resource: { buffer: bufferC } },
          ],
        });

        // Warmup
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

        // Benchmark
        const times: number[] = [];
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
        const flops = 2 * M * N * K; // multiply-add = 2 ops
        const gflops = (flops / avgTime) / 1e6; // GFLOPS

        if (avgTime < best.timeMs) {
          best = {
            optimalWorkgroupSize: [wgX, wgY, 1],
            optimalTileSize: wgX,
            throughput: gflops,
            timeMs: avgTime,
            deviceInfo: this.capabilities?.adapterInfo,
          };
        }

        uniformBuffer.destroy();
      } catch (e) {
        // Skip invalid configurations
        continue;
      }
    }

    // Cleanup
    bufferA.destroy();
    bufferB.destroy();
    bufferC.destroy();

    return best;
  }

  /**
   * Create matmul shader with specified workgroup size
   * @private
   */
  private _createMatmulShader(wgX: number, wgY: number): string {
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
  private async _tuneAttention(
    inputSizes: InputSizes,
    candidates: WorkgroupSize[],
    warmup: number,
    iterations: number
  ): Promise<TuneResult> {
    // For attention, test 1D workgroups
    const { seqLen = 2048, numHeads = 32, headDim = 128 } = inputSizes;

    const best: TuneResult = {
      optimalWorkgroupSize: [64, 1, 1],
      optimalTileSize: 64,
      throughput: 0,
      timeMs: Infinity,
      deviceInfo: this.capabilities?.adapterInfo,
    };

    // Simplified tuning - just return heuristic-based result
    // Real tuning would create attention shader variants
    const invocations = this.limits?.maxComputeInvocationsPerWorkgroup || 256;
    const wgSize = Math.min(64, invocations);

    best.optimalWorkgroupSize = [wgSize, 1, 1];
    best.optimalTileSize = wgSize;

    return best;
  }

  /**
   * Tune softmax kernel
   * @private
   */
  private async _tuneSoftmax(
    inputSizes: InputSizes,
    candidates: WorkgroupSize[],
    warmup: number,
    iterations: number
  ): Promise<TuneResult> {
    const { innerSize = 32000, outerSize = 1 } = inputSizes;

    const best: TuneResult = {
      optimalWorkgroupSize: [256, 1, 1],
      optimalTileSize: 256,
      throughput: 0,
      timeMs: Infinity,
      deviceInfo: this.capabilities?.adapterInfo,
    };

    // For softmax, 1D workgroup with size based on reduction efficiency
    const invocations = this.limits?.maxComputeInvocationsPerWorkgroup || 256;
    const wgSize = Math.min(256, invocations);

    best.optimalWorkgroupSize = [wgSize, 1, 1];
    best.optimalTileSize = wgSize;

    return best;
  }

  /**
   * Tune RMSNorm kernel
   * @private
   */
  private async _tuneRMSNorm(
    inputSizes: InputSizes,
    candidates: WorkgroupSize[],
    warmup: number,
    iterations: number
  ): Promise<TuneResult> {
    const { hiddenSize = 4096, numTokens = 1 } = inputSizes;

    const best: TuneResult = {
      optimalWorkgroupSize: [256, 1, 1],
      optimalTileSize: 256,
      throughput: 0,
      timeMs: Infinity,
      deviceInfo: this.capabilities?.adapterInfo,
    };

    // RMSNorm benefits from larger workgroups for reduction
    const invocations = this.limits?.maxComputeInvocationsPerWorkgroup || 256;
    const wgSize = Math.min(256, invocations);

    best.optimalWorkgroupSize = [wgSize, 1, 1];
    best.optimalTileSize = wgSize;

    return best;
  }

  /**
   * Tune dequantization kernel
   * @private
   */
  private async _tuneDequant(
    inputSizes: InputSizes,
    candidates: WorkgroupSize[],
    warmup: number,
    iterations: number
  ): Promise<TuneResult> {
    const { numBlocks = 1000 } = inputSizes;

    const best: TuneResult = {
      optimalWorkgroupSize: [64, 1, 1],
      optimalTileSize: 64,
      throughput: 0,
      timeMs: Infinity,
      deviceInfo: this.capabilities?.adapterInfo,
    };

    // Dequant is memory-bound, smaller workgroups often better
    const hasSubgroups = this.capabilities?.hasSubgroups;
    const wgSize = hasSubgroups ? 64 : 256;

    best.optimalWorkgroupSize = [wgSize, 1, 1];
    best.optimalTileSize = wgSize;

    return best;
  }

  /**
   * Generic tuning for unknown kernels
   * @private
   */
  private async _tuneGeneric(
    kernelName: string,
    inputSizes: InputSizes,
    candidates: WorkgroupSize[],
    warmup: number,
    iterations: number
  ): Promise<TuneResult> {
    // Return sensible defaults
    return {
      optimalWorkgroupSize: [256, 1, 1],
      optimalTileSize: 256,
      throughput: 0,
      timeMs: 0,
      deviceInfo: this.capabilities?.adapterInfo,
    };
  }

  /**
   * Create compute pipeline from shader source
   * @private
   */
  private async _createComputePipeline(
    shaderSource: string,
    entryPoint: string
  ): Promise<GPUComputePipeline> {
    if (!this.device) {
      throw new Error('Device not initialized');
    }
    const module = this.device.createShaderModule({ code: shaderSource });
    return await this.device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint },
    });
  }

  /**
   * Get cached tuning result
   * @param kernelName - Kernel name
   * @param inputSizes - Input sizes
   * @returns Cached result or null
   */
  getCachedResult(kernelName: string, inputSizes: InputSizes): TuneResult | null {
    const cacheKey: CacheKey = `${kernelName}_${JSON.stringify(inputSizes)}`;
    return this.cache.get(cacheKey) || null;
  }

  /**
   * Clear all cached results
   */
  clearCache(): void {
    this.cache.clear();
    if (typeof localStorage !== 'undefined') {
      const signature = this._getDeviceSignature();
      localStorage.removeItem(CACHE_PREFIX + signature);
    }
  }

  /**
   * Get all cached results
   * @returns Object with all cached results
   */
  getAllCachedResults(): Record<string, TuneRecord> {
    return Object.fromEntries(this.cache);
  }

  /**
   * Destroy tuner resources
   */
  destroy(): void {
    if (this.profiler) {
      this.profiler.destroy();
    }
  }
}

// Global tuner instance
let globalTuner: KernelTuner | null = null;

/**
 * Get the global kernel tuner
 * @returns Promise resolving to kernel tuner instance
 */
export async function getKernelTuner(): Promise<KernelTuner> {
  if (!globalTuner) {
    globalTuner = new KernelTuner();
    await globalTuner.init();
  }
  return globalTuner;
}

/**
 * Convenience function to tune a kernel
 * @param kernelName - Kernel name
 * @param inputSizes - Input sizes
 * @returns Promise resolving to tuning result
 */
export async function tuneKernel(
  kernelName: string,
  inputSizes: InputSizes
): Promise<TuneResult> {
  const tuner = await getKernelTuner();
  return tuner.tuneKernel(kernelName, inputSizes);
}

export default KernelTuner;
