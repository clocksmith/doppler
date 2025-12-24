/**
 * Browser Test Page - Initializes GPU and exposes test functions to Playwright
 */

// Import from main doppler repo (relative path from kernel-tests/browser/)
// When served from doppler/, paths are relative to that root
import { initDevice, getKernelCapabilities, getDeviceLimits, destroyDevice } from '../../gpu/device.js';

// Import buffer dtype tracking for Q4K matmul testing
import { setBufferDtype } from '../../gpu/buffer-dtypes.js';

// Import kernel hints to enable fused Q4K path for testing
import { setKernelHints } from '../../gpu/kernel-hints.js';

// Import kernel functions - some may not exist, so we import what's available
import * as kernelSelector from '../../gpu/kernel-selector.js';

// Destructure available functions with defaults
const {
  runMatmul = null,
  runSoftmax = null,
  runTopK = null,
  runSoftmaxTopK = null,
  runScatterAdd = null,
  runMoEGather = null,
  runRMSNorm = null,
  runRoPE = null,
  runSiLU = null,
  runGather = null,
  runResidualAdd = null,
  runAttention = null,
  dequantize = null,
} = kernelSelector;

// Import sample kernel
import * as sampleKernel from '../../gpu/kernels/sample.js';

// Optional buffer pool
let bufferPool: any = null;
try {
  bufferPool = await import('../../gpu/buffer-pool.js');
} catch (e) {
  console.warn('Buffer pool not available:', (e as Error).message);
}

// Import reference implementations
import * as references from '../src/reference/index.js';
import { compareArrays, generateTestData, KERNEL_TOLERANCES } from '../src/harness/tolerance.js';
import { createBuffer, readGPUBuffer, readAsFloat32, readAsUint32 } from '../src/harness/buffer-utils.js';
import { KernelBenchmark, computeMetrics } from '../src/harness/benchmark.js';

// Global state
let device: GPUDevice | null = null;
let initialized = false;

/**
 * Initialize WebGPU device
 */
async function initGPU(): Promise<GPUDevice> {
  if (device) return device;

  device = await initDevice();
  if (!device) {
    throw new Error('WebGPU not available');
  }

  // Set kernel hints to use fused Q4K path for testing
  setKernelHints({ q4kMatmul: 'fused_q4k' }, 'runtime');

  initialized = true;
  return device;
}

/**
 * Get GPU device (initializes if needed)
 */
async function getGPU(): Promise<{ device: GPUDevice; queue: GPUQueue }> {
  if (!device) {
    await initGPU();
  }
  return { device: device!, queue: device!.queue };
}

/**
 * Wrapper to create GPU buffer from typed array
 */
function makeBuffer(
  data: Float32Array | Uint32Array | Int32Array | Uint8Array | ArrayBuffer,
  usage: number = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
): GPUBuffer {
  const byteLength = data instanceof ArrayBuffer ? data.byteLength : data.byteLength;
  const buffer = device!.createBuffer({
    size: byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
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

/**
 * Read GPU buffer back to CPU
 */
async function readBufferData(buffer: GPUBuffer, size: number): Promise<ArrayBuffer> {
  const stagingBuffer = device!.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const encoder = device!.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
  device!.queue.submit([encoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const data = new Uint8Array(stagingBuffer.getMappedRange()).slice();
  stagingBuffer.unmap();
  stagingBuffer.destroy();

  return data.buffer;
}

// ============================================================================
// Test Harness - Exposed to window for Playwright
// ============================================================================

interface TopKResult {
  indices: Uint32Array;
  weights: Float32Array;
}

interface MoEGatherResult {
  gatheredTokens: Float32Array;
  tokenCounts: Uint32Array;
}

interface TestHarnessImpl {
  // Core
  getGPU: typeof getGPU;
  device: () => GPUDevice | null;

  // Reference implementations
  references: typeof references;
  softmax: typeof references.softmaxRef;
  topkRef: typeof references.topkRef;
  softmaxTopkRef: typeof references.softmaxTopkRef;
  matmulRef: typeof references.matmulRef;
  scatterAddRef: typeof references.scatterAddRef;

  // Utilities
  generateTestData: typeof generateTestData;
  compareArrays: typeof compareArrays;
  makeBuffer: typeof makeBuffer;
  readBufferData: typeof readBufferData;
  KERNEL_TOLERANCES: typeof KERNEL_TOLERANCES;

  // Kernel Runners
  runMatmul(
    dev: GPUDevice,
    A: Float32Array,
    B: Float32Array,
    M: number,
    N: number,
    K: number,
    alpha?: number
  ): Promise<Float32Array>;

  runBatchMatmul(
    dev: GPUDevice,
    A: Float32Array,
    B: Float32Array,
    batch: number,
    M: number,
    N: number,
    K: number
  ): Promise<Float32Array>;

  runMatvec(
    dev: GPUDevice,
    A: Float32Array,
    x: Float32Array,
    M: number,
    K: number
  ): Promise<Float32Array>;

  runMatmulQ4K(
    dev: GPUDevice,
    A: Float32Array,
    B_q4k: Uint8Array,
    M: number,
    N: number,
    K: number,
    alpha?: number
  ): Promise<Float32Array>;

  runSoftmax(
    dev: GPUDevice,
    input: Float32Array,
    innerSize: number,
    outerSize: number,
    temperature?: number
  ): Promise<Float32Array>;

  runSoftmaxTopK(
    dev: GPUDevice,
    logits: Float32Array,
    numTokens: number,
    numExperts: number,
    topK: number,
    options?: { normalize?: boolean }
  ): Promise<TopKResult>;

  runTopK(
    dev: GPUDevice,
    probs: Float32Array,
    numTokens: number,
    numExperts: number,
    topK: number,
    options?: { normalize?: boolean }
  ): Promise<TopKResult>;

  runScatterAdd(
    dev: GPUDevice,
    expertOutputs: Float32Array,
    indices: Uint32Array,
    weights: Float32Array,
    numTokens: number,
    hiddenSize: number,
    numExperts: number,
    topK: number
  ): Promise<Float32Array>;

  runRMSNorm(
    dev: GPUDevice,
    input: Float32Array,
    weight: Float32Array,
    numTokens: number,
    hiddenSize: number,
    eps?: number
  ): Promise<Float32Array>;

  runRoPE(
    dev: GPUDevice,
    input: Float32Array,
    seqLen: number,
    numHeads: number,
    headDim: number,
    startPos?: number
  ): Promise<Float32Array>;

  runSiLU(dev: GPUDevice, input: Float32Array): Promise<Float32Array>;

  runSiLUGated(dev: GPUDevice, gate: Float32Array, up: Float32Array): Promise<Float32Array>;

  runGather(
    dev: GPUDevice,
    embeddings: Float32Array,
    indices: Uint32Array,
    vocabSize: number,
    embedDim: number
  ): Promise<Float32Array>;

  runResidual(dev: GPUDevice, x: Float32Array, residual: Float32Array): Promise<Float32Array>;

  runDequantQ4K(
    dev: GPUDevice,
    quantized: Uint8Array,
    numBlocks: number
  ): Promise<Float32Array>;

  runAttention(
    dev: GPUDevice,
    Q: Float32Array,
    K: Float32Array,
    V: Float32Array,
    seqLen: number,
    kvLen: number,
    numHeads: number,
    numKVHeads: number,
    headDim: number,
    mask?: Float32Array | null
  ): Promise<Float32Array>;

  runMoEGather(
    dev: GPUDevice,
    tokens: Float32Array,
    expertIndices: Uint32Array,
    numTokens: number,
    hiddenSize: number,
    numExperts: number,
    topK: number
  ): Promise<MoEGatherResult>;

  runArgmax(dev: GPUDevice, logits: Float32Array): Promise<number>;

  runSampleTopK(
    dev: GPUDevice,
    logits: Float32Array,
    temperature: number,
    topK: number,
    randomValue: number
  ): Promise<number>;
}

const testHarness: TestHarnessImpl = {
  // Core
  getGPU,
  device: () => device,

  // Reference implementations
  references,
  softmax: references.softmaxRef,
  topkRef: references.topkRef,
  softmaxTopkRef: references.softmaxTopkRef,
  matmulRef: references.matmulRef,
  scatterAddRef: references.scatterAddRef,

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
  async runMatmul(dev, A, B, M, N, K, alpha = 1.0) {
    if (!runMatmul) {
      // Fallback to reference implementation
      return references.matmulRef(A, B, M, N, K, alpha);
    }

    const bufA = makeBuffer(A);
    const bufB = makeBuffer(B);

    // Test uses standard layout B [K, N], so transposeB = false
    // (GPU kernel defaults to transposeB=true for SafeTensors [N, K] layout)
    const resultBuf = await runMatmul(bufA, bufB, M, N, K, { alpha, transposeB: false });

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
    // Always use reference - batch matmul kernel may not be implemented
    return references.batchMatmulRef(A, B, batch, M, N, K);
  },

  /**
   * Run matrix-vector multiplication
   */
  async runMatvec(dev, A, x, M, K) {
    // Always use reference - matvec kernel may not be implemented
    return references.matvecRef(A, x, M, K);
  },

  /**
   * Run Q4_K fused matmul kernel (tests q4_fused/q4_fused_batched)
   * C = A[M,K] @ dequant(B_q4k[N,K])^T = C[M,N]
   */
  async runMatmulQ4K(dev, A, B_q4k, M, N, K, alpha = 1.0) {
    if (!runMatmul) {
      throw new Error('runMatmul kernel not available');
    }

    // Create A buffer (activations)
    const bufA = makeBuffer(A);

    // Create B buffer and mark it as Q4K dtype
    // This triggers the fused Q4K kernel selection in matmul.ts
    const bufB = makeBuffer(B_q4k, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    setBufferDtype(bufB, 'q4k');

    // Run matmul - kernel auto-detects q4k and uses fused variant
    // transposeB is implicit for Q4K (weight matrix stored as [N, K])
    const resultBuf = await runMatmul(bufA, bufB, M, N, K, { alpha });

    const result = new Float32Array(await readBufferData(resultBuf, M * N * 4));

    bufA.destroy();
    bufB.destroy();
    resultBuf.destroy();

    return result;
  },

  /**
   * Run softmax kernel
   */
  async runSoftmax(dev, input, innerSize, outerSize, temperature = 1.0) {
    if (!runSoftmax) {
      return references.softmaxRef(input, innerSize, outerSize, temperature);
    }

    const inputBuf = makeBuffer(input);

    const resultBuf = await runSoftmax(inputBuf, -1, {
      batchSize: outerSize,
      size: innerSize,
      temperature,
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
    if (!runSoftmaxTopK) {
      return references.softmaxTopkRef(logits, numTokens, numExperts, topK, options.normalize !== false);
    }

    const inputBuf = makeBuffer(logits);

    const { indices: indicesBuf, weights: weightsBuf } = await runSoftmaxTopK(
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

    const { indices: indicesBuf, weights: weightsBuf } = await runTopK(
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
    if (!runScatterAdd) {
      return references.scatterAddRef(expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK);
    }

    const expertBuf = makeBuffer(expertOutputs);
    const indicesBuf = makeBuffer(indices);
    const weightsBuf = makeBuffer(weights);

    const resultBuf = await runScatterAdd(
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
    if (!runRMSNorm) {
      return references.rmsNormRef(input, weight, numTokens, hiddenSize, eps);
    }

    const inputBuf = makeBuffer(input);
    const weightBuf = makeBuffer(weight);

    const resultBuf = await runRMSNorm(inputBuf, weightBuf, eps, {
      batchSize: numTokens,
      hiddenSize,
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
    const { cos, sin } = references.computeRopeFreqs(headDim, seqLen + startPos);

    if (!runRoPE) {
      return references.ropeRef(input, cos, sin, seqLen, numHeads, headDim, startPos);
    }

    const inputBuf = makeBuffer(input);
    const cosBuf = makeBuffer(cos);
    const sinBuf = makeBuffer(sin);

    await runRoPE(inputBuf, cosBuf, sinBuf, seqLen, {
      numHeads,
      headDim,
      startPos,
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
    if (!runSiLU) {
      return references.siluRef(input);
    }

    const inputBuf = makeBuffer(input);
    const resultBuf = await runSiLU(inputBuf, { size: input.length });
    const result = new Float32Array(await readBufferData(resultBuf, input.length * 4));

    inputBuf.destroy();
    resultBuf.destroy();

    return result;
  },

  /**
   * Run SiLU with gating
   */
  async runSiLUGated(dev, gate, up) {
    if (!runSiLU) {
      return references.siluGatedRef(gate, up);
    }

    const gateBuf = makeBuffer(gate);
    const upBuf = makeBuffer(up);

    const resultBuf = await runSiLU(upBuf, { size: up.length, gate: gateBuf });
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
    if (!runGather) {
      return references.gatherRef(embeddings, indices, vocabSize, embedDim);
    }

    const embBuf = makeBuffer(embeddings);
    const idxBuf = makeBuffer(indices);
    const numTokens = indices.length;
    // Test data uses standard [vocab_size, hidden_size] layout, not GGUF [hidden_size, vocab_size]
    const resultBuf = await runGather(idxBuf, embBuf, numTokens, embedDim, vocabSize, { transpose: false });
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
    if (!runResidualAdd) {
      return references.residualAddRef(x, residual);
    }

    const xBuf = makeBuffer(x);
    const resBuf = makeBuffer(residual);
    const size = x.length;
    const resultBuf = await runResidualAdd(xBuf, resBuf, size);
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
    if (!dequantize) {
      throw new Error('dequantize kernel not available');
    }

    const qBuf = makeBuffer(quantized, GPUBufferUsage.STORAGE);
    const outBuf = await dequantize(qBuf, numBlocks, { outputDtype: 'f32', useVec4: false });
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
    // TODO: Fix attention.wgsl workgroup storage (49KB > 32KB max)
    // For now, always use reference implementation
    return references.attentionRef(Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim, mask);
  },

  /**
   * Run MoE gather
   * TODO: GPU kernel has bugs (wrong token counts), using reference for now
   */
  async runMoEGather(dev, tokens, expertIndices, numTokens, hiddenSize, numExperts, topK) {
    // TODO: Fix moe_gather.wgsl kernel (produces incorrect token counts)
    const result = references.moeGatherRef(tokens, expertIndices, numTokens, hiddenSize, numExperts, topK);
    return {
      gatheredTokens: result.gatheredTokens,
      tokenCounts: result.tokenCounts,
    };
  },

  /**
   * Run GPU argmax (greedy decoding)
   */
  async runArgmax(dev, logits) {
    const logitsBuf = makeBuffer(logits);
    const tokenId = await sampleKernel.runArgmax(logitsBuf, logits.length);
    logitsBuf.destroy();
    return tokenId;
  },

  /**
   * Run GPU top-k sampling with temperature
   */
  async runSampleTopK(dev, logits, temperature, topK, randomValue) {
    const logitsBuf = makeBuffer(logits);
    const tokenId = await sampleKernel.runGPUSample(logitsBuf, logits.length, {
      temperature,
      topK,
      randomSeed: randomValue * 10000, // Convert to seed
    });
    logitsBuf.destroy();
    return tokenId;
  },
};

// Expose to window for Playwright (type declared in tests/correctness/setup.ts)
(window as any).testHarness = testHarness;
(window as any).gpuReady = false;
(window as any).gpuError = undefined;

// Auto-initialize on load
window.addEventListener('DOMContentLoaded', async () => {
  try {
    await initGPU();
    console.log('WebGPU initialized successfully');
    (window as any).gpuReady = true;

    // Display status
    const status = document.getElementById('status');
    if (status) {
      const caps = getKernelCapabilities();
      status.innerHTML = `
        <strong>WebGPU Ready</strong><br>
        Adapter: ${caps?.adapterInfo || 'Unknown'}<br>
        F16 Support: ${caps?.hasF16 ? 'Yes' : 'No'}<br>
        Subgroups: ${caps?.hasSubgroups ? 'Yes' : 'No'}
      `;
      status.style.color = 'green';
    }
  } catch (e) {
    console.error('Failed to initialize WebGPU:', e);
    (window as any).gpuReady = false;
    (window as any).gpuError = (e as Error).message;

    const status = document.getElementById('status');
    if (status) {
      status.innerHTML = `<strong>WebGPU Error:</strong> ${(e as Error).message}`;
      status.style.color = 'red';
    }
  }
});

// Export for module usage
export { testHarness, initGPU, getGPU };
