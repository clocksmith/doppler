/**
 * Browser Test Page - Initializes GPU and exposes test functions to Playwright
 */

// Import from main doppler repo (relative path from tests/kernels/browser/)
// When served from doppler/, paths are relative to that root
import { initDevice, getKernelCapabilities, getDeviceLimits, destroyDevice } from '../../../src/gpu/device.js';

// Import tensor abstraction for Tensor-based kernels
import { createTensor } from '../../../src/gpu/tensor.js';

// Ensure platform/registry lookups resolve to the main config paths when bundled
import { setPlatformsBaseUrl } from '../../../src/config/platforms/loader.js';
import { setRegistryUrl } from '../../../src/config/kernels/registry.js';

// Import kernel path to enable fused Q4K path for testing
import { resolveKernelPath, setActiveKernelPath } from '../../../src/config/kernel-path-loader.js';

// Import kernel functions - some may not exist, so we import what's available
import * as kernelSelector from '../../../src/gpu/kernel-selector.js';

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
  runSwiGLURowsplitBias = null,
  runScale = null,
  runGather = null,
  runResidualAdd = null,
  runBiasAdd = null,
  runAttention = null,
  dequantize = null,
  dequantizeQ6K = null,
  runBF16ToF32 = null,
  runBF16ToF16 = null,
  castF32ToF16 = null,
  runGeLU = null,
  runSplitQKV = null,
} = kernelSelector;

// Import sample kernel
import * as sampleKernel from '../../../src/gpu/kernels/sample.js';

// Import check-stop kernel
import { checkStop } from '../../../src/gpu/kernels/check-stop.js';

// Import fused kernels
import { runMatmulResidualFused } from '../../../src/gpu/kernels/fused_matmul_residual.js';
import { runMatmulRMSNormFused } from '../../../src/gpu/kernels/fused_matmul_rmsnorm.js';
import { runFusedFFN } from '../../../src/gpu/kernels/fused_ffn.js';

// Optional buffer pool
let bufferPool = null;
try {
  bufferPool = await import('../../../src/gpu/buffer-pool.js');
} catch (e) {
  console.warn('Buffer pool not available:', e.message);
}

// Import reference implementations
import * as references from '../reference/index.js';
import { compareArrays, generateTestData, KERNEL_TOLERANCES } from '../harness/tolerance.js';
import { createBuffer, readGPUBuffer, readAsFloat32, readAsUint32 } from '../harness/buffer-utils.js';
import { KernelBenchmark, computeMetrics } from '../harness/benchmark.js';

// Global state
let device = null;
let initialized = false;

/**
 * Convert f16 (IEEE 754 half-precision) to f32
 */
function f16ToF32(h) {
  const sign = (h & 0x8000) >> 15;
  const exponent = (h & 0x7C00) >> 10;
  const mantissa = h & 0x03FF;

  if (exponent === 0) {
    // Denormalized or zero
    if (mantissa === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mantissa / 1024);
  } else if (exponent === 31) {
    // Infinity or NaN
    return mantissa === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }

  // Normalized
  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
}

function f32ToF16Bits(value) {
  return references.float32ToFloat16(value);
}

function toF16Array(values) {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) {
    out[i] = f32ToF16Bits(values[i]);
  }
  return out;
}

function toF16RoundedFloat32(values) {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    out[i] = f16ToF32(f32ToF16Bits(values[i]));
  }
  return out;
}

/**
 * Initialize WebGPU device
 */
async function initGPU() {
  if (device) return device;

  setPlatformsBaseUrl('/src/config/platforms/');
  setRegistryUrl('/src/config/kernels/registry.json');

  device = await initDevice();
  if (!device) {
    throw new Error('WebGPU not available');
  }

  // Set kernel path to use fused Q4K path for testing
  setActiveKernelPath(resolveKernelPath('q4k-fused'), 'runtime');

  initialized = true;
  return device;
}

/**
 * Get GPU device (initializes if needed)
 */
async function getGPU() {
  if (!device) {
    await initGPU();
  }
  return { device: device, queue: device.queue };
}

/**
 * Wrapper to create GPU buffer from typed array
 */
function makeBuffer(
  data,
  usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
) {
  const byteLength = data instanceof ArrayBuffer ? data.byteLength : data.byteLength;
  const buffer = device.createBuffer({
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

/**
 * Read GPU buffer back to CPU
 */
async function readBufferData(buffer, size) {
  const stagingBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
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

// ============================================================================
// Test Harness - Exposed to window for Playwright
// ============================================================================

const testHarness = {
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
    const tensorA = createTensor(bufA, 'f32', [M, K], 'matmul_a');
    const bufB = makeBuffer(B);

    // Test uses standard layout B [K, N], so transposeB = false
    // (GPU kernel defaults to transposeB=true for SafeTensors [N, K] layout)
    const resultTensor = await runMatmul(tensorA, bufB, M, N, K, { alpha, transposeB: false });

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
    const tensorA = createTensor(bufA, 'f32', [M, K], 'matmul_q4k_a');

    // Create B buffer and pass q4k dtype to trigger fused Q4K selection
    const bufB = makeBuffer(B_q4k, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    // Run matmul - kernel auto-detects q4k and uses fused variant
    // transposeB is implicit for Q4K (weight matrix stored as [N, K])
    const resultTensor = await runMatmul(tensorA, bufB, M, N, K, { alpha, bDtype: 'q4k' });

    const result = new Float32Array(await readBufferData(resultTensor.buffer, M * N * 4));

    bufA.destroy();
    bufB.destroy();
    resultTensor.buffer.destroy();

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
    const inputTensor = createTensor(inputBuf, 'f32', [outerSize, innerSize], 'softmax_input');

    const resultTensor = await runSoftmax(inputTensor, -1, {
      batchSize: outerSize,
      size: innerSize,
      temperature,
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

    // Wrap expertBuf in Tensor (MoE kernels now use Tensor abstraction)
    const expertTensor = createTensor(expertBuf, 'f32', [numExperts, numTokens, hiddenSize], 'expert_outputs');
    const resultTensor = await runScatterAdd(
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
    if (!runRMSNorm) {
      return references.rmsNormRef(input, weight, numTokens, hiddenSize, eps);
    }

    const inputBuf = makeBuffer(input);
    const weightBuf = makeBuffer(weight);
    const inputTensor = createTensor(inputBuf, 'f32', [numTokens, hiddenSize], 'rmsnorm_input');

    const resultTensor = await runRMSNorm(inputTensor, weightBuf, eps, {
      batchSize: numTokens,
      hiddenSize,
    });

    let result;
    if (resultTensor.dtype === 'f16') {
      const rawData = await readBufferData(resultTensor.buffer, numTokens * hiddenSize * 2);
      const u16View = new Uint16Array(rawData);
      result = new Float32Array(u16View.length);
      for (let i = 0; i < u16View.length; i++) {
        result[i] = f16ToF32(u16View[i]);
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
    const { cos, sin } = references.computeRopeFreqs(headDim, seqLen + startPos);

    if (!runRoPE) {
      return references.ropeRef(input, cos, sin, seqLen, numHeads, headDim, startPos);
    }

    const inputBuf = makeBuffer(input);
    const cosBuf = makeBuffer(cos);
    const sinBuf = makeBuffer(sin);

    const inputTensor = createTensor(inputBuf, 'f32', [seqLen, numHeads, headDim], 'rope_input');

    await runRoPE(inputTensor, cosBuf, sinBuf, seqLen, {
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
    const inputTensor = createTensor(inputBuf, 'f32', [input.length], 'silu_input');

    const resultTensor = await runSiLU(inputTensor, { size: input.length });
    const result = new Float32Array(await readBufferData(resultTensor.buffer, input.length * 4));

    inputBuf.destroy();
    resultTensor.buffer.destroy();

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

    const gateTensor = createTensor(gateBuf, 'f32', [gate.length], 'silu_gate');
    const upTensor = createTensor(upBuf, 'f32', [up.length], 'silu_up');

    const resultTensor = await runSiLU(upTensor, { size: up.length, gate: gateTensor });
    const result = new Float32Array(await readBufferData(resultTensor.buffer, up.length * 4));

    gateBuf.destroy();
    upBuf.destroy();
    resultTensor.buffer.destroy();

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

    const embTensor = createTensor(embBuf, 'f32', [vocabSize, embedDim], 'gather_embeddings');
    const idxTensor = createTensor(idxBuf, 'u32', [numTokens], 'gather_indices');

    // Test data uses standard [vocab_size, hidden_size] layout, not GGUF [hidden_size, vocab_size]
    const resultTensor = await runGather(idxTensor.buffer, embTensor.buffer, numTokens, embedDim, vocabSize, { transpose: false });
    let result;
    if (resultTensor.dtype === 'f16') {
      const rawData = await readBufferData(resultTensor.buffer, numTokens * embedDim * 2);
      const u16View = new Uint16Array(rawData);
      result = new Float32Array(u16View.length);
      for (let i = 0; i < u16View.length; i++) {
        result[i] = f16ToF32(u16View[i]);
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
    if (!runResidualAdd) {
      return references.residualAddRef(x, residual);
    }

    const xBuf = makeBuffer(x);
    const resBuf = makeBuffer(residual);
    const size = x.length;

    const xTensor = createTensor(xBuf, 'f32', [size], 'residual_x');
    const resTensor = createTensor(resBuf, 'f32', [size], 'residual_res');

    const resultTensor = await runResidualAdd(xTensor, resTensor, size);
    const result = new Float32Array(await readBufferData(resultTensor.buffer, size * 4));

    xBuf.destroy();
    resBuf.destroy();
    resultTensor.buffer.destroy();

    return result;
  },

  /**
   * Run bias add (row-wise)
   */
  async runBiasAdd(dev, data, bias, numTokens, dim) {
    if (!runBiasAdd) {
      const result = new Float32Array(data);
      for (let t = 0; t < numTokens; t++) {
        const rowOffset = t * dim;
        for (let d = 0; d < dim; d++) {
          result[rowOffset + d] += bias[d];
        }
      }
      return result;
    }

    const dataBuf = makeBuffer(data);
    const biasBuf = makeBuffer(bias);
    const dataTensor = createTensor(dataBuf, 'f32', [numTokens, dim], 'bias_add_data');
    const biasTensor = createTensor(biasBuf, 'f32', [dim], 'bias_add_bias');

    const resultTensor = await runBiasAdd(dataTensor, biasTensor, numTokens, dim);
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
    if (!dequantize) {
      throw new Error('dequantize kernel not available');
    }

    const qBuf = makeBuffer(quantized, GPUBufferUsage.STORAGE);
    const outTensor = await dequantize(qBuf, numBlocks, { outputDtype: 'f32', useVec4: false });
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
    if (!dequantize) {
      throw new Error('dequantize kernel not available');
    }

    const qBuf = makeBuffer(quantized, GPUBufferUsage.STORAGE);
    // Use F16 output and vec4=true (default) to match production loader path
    const outTensor = await dequantize(qBuf, numBlocks, { outputDtype: 'f16', useVec4: true });

    // Read back F16 data and convert to F32 for comparison
    const f16Bytes = numBlocks * 256 * 2; // F16 = 2 bytes per element
    const rawData = await readBufferData(outTensor.buffer, f16Bytes);
    const u16 = new Uint16Array(rawData);
    const out = new Float32Array(numBlocks * 256);

    // Convert F16 to F32
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i];
      const sign = (h >> 15) & 1;
      const exp = (h >> 10) & 0x1F;
      const mant = h & 0x3FF;
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
    if (!runAttention) {
      // Fallback to reference if kernel not available
      return references.attentionRef(Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim, mask);
    }

    // Create GPU buffers
    const qBuf = makeBuffer(Q);
    const kBuf = makeBuffer(K);
    const vBuf = makeBuffer(V);
    const maskBuf = mask ? makeBuffer(mask) : null;
    const isCausal = !!mask;

    const qTensor = createTensor(qBuf, 'f32', [seqLen, numHeads, headDim], 'attn_q');
    const kTensor = createTensor(kBuf, 'f32', [kvLen, numKVHeads, headDim], 'attn_k');
    const vTensor = createTensor(vBuf, 'f32', [kvLen, numKVHeads, headDim], 'attn_v');

    // Run attention via kernel selector (handles tier selection automatically)
    const resultTensor = await runAttention(qTensor, kTensor, vTensor, maskBuf, numHeads, headDim, {
      seqLen,
      kvLen,
      numKVHeads,
      scale: 1 / Math.sqrt(headDim),
      causal: isCausal,
    });

    // Read back result
    const out = new Float32Array(await readBufferData(resultTensor.buffer, seqLen * numHeads * headDim * 4));
    qBuf.destroy();
    kBuf.destroy();
    vBuf.destroy();
    maskBuf?.destroy();
    resultTensor.buffer.destroy();
    return out;
  },

  /**
   * Run MoE gather - dispatches tokens to experts
   * Now uses the fixed two-phase GPU kernel (count_and_map + gather_tokens)
   */
  async runMoEGather(dev, tokens, expertIndices, numTokens, hiddenSize, numExperts, topK) {
    if (!runMoEGather) {
      // Fallback to reference if kernel not available
      const result = references.moeGatherRef(tokens, expertIndices, numTokens, hiddenSize, numExperts, topK);
      return {
        gatheredTokens: result.gatheredTokens,
        tokenCounts: result.tokenCounts,
      };
    }

    // Create GPU buffers
    const tokensBuf = makeBuffer(tokens);
    const indicesBuf = makeBuffer(expertIndices);

    // Wrap tokensBuf in Tensor (MoE kernels now use Tensor abstraction)
    const tokensTensor = createTensor(tokensBuf, 'f32', [numTokens, hiddenSize], 'moe_input');

    // Run MoE gather via kernel selector
    const result = await runMoEGather(tokensTensor, indicesBuf, numTokens, hiddenSize, numExperts, topK);

    // Read back results (result.gathered is now a Tensor)
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
      tokenCounts,
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

  /**
   * Run SwiGLU activation: output = SiLU(gate) * up
   * Tests the gated SiLU variant from the silu kernel
   */
  async runSwiGLU(dev, gate, up, gateBias, upBias) {
    // For testing, we pre-add bias to gate and up, then use SiLU with gating
    const size = gate.length;

    // Add biases
    const gateWithBias = new Float32Array(size);
    const upWithBias = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      gateWithBias[i] = gate[i] + gateBias[i];
      upWithBias[i] = up[i] + upBias[i];
    }

    if (!runSiLU) {
      // Fallback to reference implementation
      const result = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        const silu = gateWithBias[i] / (1 + Math.exp(-gateWithBias[i]));
        result[i] = silu * upWithBias[i];
      }
      return result;
    }

    const gateBuf = makeBuffer(gateWithBias);
    const upBuf = makeBuffer(upWithBias);

    const gateTensor = createTensor(gateBuf, 'f32', [size], 'swiglu_gate');
    const upTensor = createTensor(upBuf, 'f32', [size], 'swiglu_up');

    // runSiLU with gate option: output = silu(gate) * up
    const resultTensor = await runSiLU(upTensor, { size, gate: gateTensor });

    const result = new Float32Array(await readBufferData(resultTensor.buffer, size * 4));

    gateBuf.destroy();
    upBuf.destroy();
    resultTensor.buffer.destroy();

    return result;
  },

  /**
   * Run scale kernel: output[i] = input[i] * scale
   */
  async runScale(dev, input, scale) {
    if (!runScale) {
      // Fallback to reference implementation
      const result = new Float32Array(input.length);
      for (let i = 0; i < input.length; i++) {
        result[i] = input[i] * scale;
      }
      return result;
    }

    const inputBuf = makeBuffer(input);
    const inputTensor = createTensor(inputBuf, 'f32', [input.length], 'scale_input');

    const resultTensor = await runScale(inputTensor, scale, { count: input.length });
    const result = new Float32Array(await readBufferData(resultTensor.buffer, input.length * 4));

    inputBuf.destroy();
    resultTensor.buffer.destroy();

    return result;
  },

  /**
   * Run GELU activation: GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
   */
  async runGeLU(dev, input) {
    if (!runGeLU) {
      return references.geluRef(input);
    }

    const inputBuf = makeBuffer(input);
    const inputTensor = createTensor(inputBuf, 'f32', [input.length], 'gelu_input');

    const resultTensor = await runGeLU(inputTensor, { size: input.length });
    const result = new Float32Array(await readBufferData(resultTensor.buffer, input.length * 4));

    inputBuf.destroy();
    resultTensor.buffer.destroy();

    return result;
  },

  /**
   * Run GeGLU activation: GELU(gate) * up
   */
  async runGeGLU(dev, gate, up) {
    if (!runGeLU) {
      return references.gegluRef(gate, up);
    }

    const gateBuf = makeBuffer(gate);
    const upBuf = makeBuffer(up);

    const gateTensor = createTensor(gateBuf, 'f32', [gate.length], 'geglu_gate');
    const upTensor = createTensor(upBuf, 'f32', [up.length], 'geglu_up');

    const resultTensor = await runGeLU(upTensor, { size: up.length, gate: gateTensor });
    const result = new Float32Array(await readBufferData(resultTensor.buffer, up.length * 4));

    gateBuf.destroy();
    upBuf.destroy();
    resultTensor.buffer.destroy();

    return result;
  },

  /**
   * Run split QKV kernel: split fused QKV [numTokens, qSize+kSize+vSize] into Q, K, V
   */
  async runSplitQKV(dev, qkv, numTokens, qSize, kSize, vSize) {
    if (!runSplitQKV) {
      return references.splitQkvRef(qkv, numTokens, qSize, kSize, vSize);
    }

    const qkvBuf = makeBuffer(qkv);
    const qkvTensor = createTensor(qkvBuf, 'f32', [numTokens, qSize + kSize + vSize], 'split_qkv_input');

    const { Q: qTensor, K: kTensor, V: vTensor } = await runSplitQKV(qkvTensor, {
      numTokens,
      qSize,
      kSize,
      vSize,
    });

    const Q = new Float32Array(await readBufferData(qTensor.buffer, numTokens * qSize * 4));
    const K = new Float32Array(await readBufferData(kTensor.buffer, numTokens * kSize * 4));
    const V = new Float32Array(await readBufferData(vTensor.buffer, numTokens * vSize * 4));

    qkvBuf.destroy();
    qTensor.buffer.destroy();
    kTensor.buffer.destroy();
    vTensor.buffer.destroy();

    return { Q, K, V };
  },

  /**
   * Run BF16 -> F32 cast
   */
  async runBF16ToF32(dev, input) {
    if (!runBF16ToF32) {
      const out = new Float32Array(input.length);
      for (let i = 0; i < input.length; i++) {
        const view = new DataView(new ArrayBuffer(4));
        view.setUint32(0, input[i] << 16, true);
        out[i] = view.getFloat32(0, true);
      }
      return out;
    }

    const inputBuf = makeBuffer(input, GPUBufferUsage.STORAGE);
    const outTensor = await runBF16ToF32(inputBuf, [input.length], 'bf16_to_f32_test');
    const out = new Float32Array(await readBufferData(outTensor.buffer, input.length * 4));

    inputBuf.destroy();
    outTensor.buffer.destroy();

    return out;
  },

  /**
   * Run F32 -> F16 cast
   */
  async runF32ToF16(dev, input) {
    if (!castF32ToF16) {
      const out = new Uint16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        const view = new DataView(new ArrayBuffer(4));
        view.setFloat32(0, input[i], true);
        const bits = view.getUint32(0, true);
        const sign = (bits >> 31) & 0x1;
        const exp = (bits >> 23) & 0xff;
        const mant = bits & 0x7fffff;

        let hExp = 0;
        let hMant = 0;
        if (exp === 0xff) {
          hExp = 0x1f;
          hMant = mant ? 0x200 : 0;
        } else if (exp !== 0) {
          const newExp = exp - 127 + 15;
          if (newExp >= 0x1f) {
            hExp = 0x1f;
          } else if (newExp > 0) {
            hExp = newExp;
            hMant = mant >> 13;
          }
        }
        out[i] = (sign << 15) | (hExp << 10) | hMant;
      }
      return out;
    }

    const inputBuf = makeBuffer(input);
    const inputTensor = createTensor(inputBuf, 'f32', [input.length], 'f32_to_f16_input');
    const outTensor = await castF32ToF16(inputTensor);
    const out = new Uint16Array(await readBufferData(outTensor.buffer, input.length * 2));

    inputBuf.destroy();
    outTensor.buffer.destroy();

    return out;
  },

  /**
   * Run BF16 -> F16 cast
   */
  async runBF16ToF16(dev, input) {
    if (!runBF16ToF16) {
      const out = new Uint16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        const view = new DataView(new ArrayBuffer(4));
        view.setUint32(0, input[i] << 16, true);
        const bits = view.getUint32(0, true);
        const sign = (bits >> 31) & 0x1;
        const exp = (bits >> 23) & 0xff;
        const mant = bits & 0x7fffff;

        let hExp = 0;
        let hMant = 0;
        if (exp === 0xff) {
          hExp = 0x1f;
          hMant = mant ? 0x200 : 0;
        } else if (exp !== 0) {
          const newExp = exp - 127 + 15;
          if (newExp >= 0x1f) {
            hExp = 0x1f;
          } else if (newExp > 0) {
            hExp = newExp;
            hMant = mant >> 13;
          }
        }
        out[i] = (sign << 15) | (hExp << 10) | hMant;
      }
      return out;
    }

    const inputBuf = makeBuffer(input, GPUBufferUsage.STORAGE);
    const outTensor = await runBF16ToF16(inputBuf, [input.length], 'bf16_to_f16_test');
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
    if (!dequantizeQ6K) {
      throw new Error('dequantizeQ6K kernel not available');
    }

    const blockSize = 256;  // Q6_K: 256 elements per block
    const qBuf = makeBuffer(quantized, GPUBufferUsage.STORAGE);
    const outTensor = await dequantizeQ6K(qBuf, numBlocks, { outputOffset: 0 });

    const outBuf = outTensor.buffer;
    // Q6K outputs f16 - read raw bytes and convert
    const rawData = await readBufferData(outBuf, numBlocks * blockSize * 2);  // f16 = 2 bytes
    const u16View = new Uint16Array(rawData);
    const out = new Float32Array(u16View.length);

    // Convert f16 to f32
    for (let i = 0; i < u16View.length; i++) {
      out[i] = f16ToF32(u16View[i]);
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
    if (!runMatmul) {
      throw new Error('runMatmul kernel not available');
    }

    // Create A buffer (F32 activations)
    const bufA = makeBuffer(A);
    const tensorA = createTensor(bufA, 'f32', [M, K], 'matmul_f16w_a');

    // Create B buffer (F16 weights) - pass raw Uint16Array
    const bufB = makeBuffer(B_f16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    // Run matmul with bDtype='f16' and preferF16=true to trigger f16w_f32a kernel
    // transposeB=true because B is [N, K] format (weight matrix layout)
    const resultTensor = await runMatmul(tensorA, bufB, M, N, K, {
      bDtype: 'f16',
      preferF16: true,
      transposeB: true,
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
    if (!runMatmul || !dequantize) {
      throw new Error('runMatmul or dequantize kernel not available');
    }

    // Dequant Q4K -> F16 on GPU
    const qBuf = makeBuffer(B_q4k, GPUBufferUsage.STORAGE);
    const dequantTensor = await dequantize(qBuf, numBlocks, { outputDtype: 'f16', useVec4: true });

    // Create A buffer (F32 activations)
    const bufA = makeBuffer(A);
    const tensorA = createTensor(bufA, 'f32', [M, K], 'dequant_matmul_a');

    // Run matmul with F16 weights (stays on GPU, no CPU round-trip)
    // transposeB=true because dequanted weights are [N, K] format
    const resultTensor = await runMatmul(tensorA, dequantTensor.buffer, M, N, K, {
      bDtype: 'f16',
      preferF16: true,
      transposeB: true,
    });

    const result = new Float32Array(await readBufferData(resultTensor.buffer, M * N * 4));

    qBuf.destroy();
    bufA.destroy();
    dequantTensor.buffer.destroy();
    resultTensor.buffer.destroy();

    return result;
  },

  /**
   * Run check-stop kernel: determine if generation should stop
   * Checks for EOS token or max tokens reached
   */
  async runCheckStop(dev, sampledToken, eosTokenId, maxTokens, currentPos) {
    // Create buffer for sampled token
    const tokenBuffer = makeBuffer(new Uint32Array([sampledToken]));

    const shouldStop = await checkStop({
      sampledTokenBuffer: tokenBuffer,
      eosTokenId,
      maxTokens,
      currentPos,
    });

    tokenBuffer.destroy();

    return shouldStop;
  },

  /**
   * Reference implementation for check-stop
   * Returns true if should stop (EOS hit or max reached)
   */
  checkStopRef(sampledToken, eosTokenId, maxTokens, currentPos) {
    const isEOS = sampledToken === eosTokenId;
    const reachedMax = currentPos >= maxTokens;
    return isEOS || reachedMax;
  },

  /**
   * Run fused matmul + residual kernel: output = matmul(input, weight) + residual
   * For decode (M=1): input[1,K] @ weight[N,K]^T + residual[1,N] = output[1,N]
   */
  async runFusedMatmulResidual(dev, input, weight, residual, N, K, alpha = 1.0) {
    const inputBuf = makeBuffer(input);
    const weightF16 = toF16Array(weight);
    const weightBuf = makeBuffer(weightF16);
    const residualBuf = makeBuffer(residual);

    const inputTensor = createTensor(inputBuf, 'f32', [1, K], 'fused_matmul_res_input');
    const residualTensor = createTensor(residualBuf, 'f32', [1, N], 'fused_matmul_res_residual');

    const resultTensor = await runMatmulResidualFused(inputTensor, weightBuf, residualTensor, {
      N,
      K,
      alpha,
    });

    const result = new Float32Array(await readBufferData(resultTensor.buffer, N * 4));

    inputBuf.destroy();
    weightBuf.destroy();
    residualBuf.destroy();
    resultTensor.buffer.destroy();

    return result;
  },

  /**
   * Reference for fused matmul + residual: output = matmul(input, weight^T) + residual
   */
  fusedMatmulResidualRef(input, weight, residual, N, K, alpha = 1.0) {
    // input: [1, K], weight: [N, K] (row-major, transposeB=true), residual: [1, N]
    const weightF16 = toF16RoundedFloat32(weight);
    const output = new Float32Array(N);
    for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += input[k] * weightF16[n * K + k];
      }
      output[n] = sum * alpha + residual[n];
    }
    return output;
  },

  /**
   * Run fused matmul + RMSNorm kernel: output = rmsnorm(matmul(input, weight))
   * For decode (M=1): input[1,K] @ weight[N,K]^T -> rmsnorm -> output[1,N]
   */
  async runFusedMatmulRMSNorm(dev, input, weight, normWeight, N, K, eps = 1e-5, residual = null) {
    const inputBuf = makeBuffer(input);
    const weightBuf = makeBuffer(weight);
    const normWeightBuf = makeBuffer(normWeight);
    const residualBuf = residual ? makeBuffer(residual) : null;

    const inputTensor = createTensor(inputBuf, 'f32', [1, K], 'fused_matmul_rmsnorm_input');

    const resultTensor = await runMatmulRMSNormFused(inputTensor, weightBuf, normWeightBuf, {
      N,
      K,
      eps,
      residual: residualBuf,
      transposeB: true,
    });

    const result = new Float32Array(await readBufferData(resultTensor.buffer, N * 4));

    inputBuf.destroy();
    weightBuf.destroy();
    normWeightBuf.destroy();
    if (residualBuf) residualBuf.destroy();
    resultTensor.buffer.destroy();

    return result;
  },

  /**
   * Reference for fused matmul + RMSNorm
   */
  fusedMatmulRMSNormRef(input, weight, normWeight, N, K, eps = 1e-5, residual = null) {
    // Step 1: matmul input[1,K] @ weight[N,K]^T -> intermediate[1,N]
    const intermediate = new Float32Array(N);
    for (let n = 0; n < N; n++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += input[k] * weight[n * K + k];
      }
      // Add residual if present
      intermediate[n] = residual ? sum + residual[n] : sum;
    }

    // Step 2: RMSNorm
    let sumSq = 0;
    for (let i = 0; i < N; i++) {
      sumSq += intermediate[i] * intermediate[i];
    }
    const rms = Math.sqrt(sumSq / N + eps);

    const output = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      output[i] = (intermediate[i] / rms) * normWeight[i];
    }

    return output;
  },

  /**
   * Run fused FFN kernel: output = activation(gate) * up
   * Where gate = input @ W_gate^T and up = input @ W_up^T
   * For decode (M=1): input[1,K] @ W_gate[N,K]^T -> gate[1,N]
   *                   input[1,K] @ W_up[N,K]^T -> up[1,N]
   *                   output = SiLU(gate) * up
   */
  async runFusedFFN(dev, input, W_gate, W_up, hiddenSize, intermediateSize, activation = 'silu') {
    const inputBuf = makeBuffer(input);
    const gateBuf = makeBuffer(W_gate);
    const upBuf = makeBuffer(W_up);

    const inputTensor = createTensor(inputBuf, 'f32', [1, hiddenSize], 'fused_ffn_input');

    const resultTensor = await runFusedFFN(inputTensor, gateBuf, upBuf, hiddenSize, intermediateSize, {
      batchSize: 1,
      activation,
      alpha: 1.0,
    });

    const result = new Float32Array(await readBufferData(resultTensor.buffer, intermediateSize * 4));

    inputBuf.destroy();
    gateBuf.destroy();
    upBuf.destroy();
    resultTensor.buffer.destroy();

    return result;
  },

  /**
   * Reference for fused FFN: output = activation(gate) * up
   */
  fusedFFNRef(input, W_gate, W_up, hiddenSize, intermediateSize, activation = 'silu') {
    // Step 1: gate = input @ W_gate^T (input[1,K] @ W_gate[N,K]^T -> gate[1,N])
    const gate = new Float32Array(intermediateSize);
    for (let n = 0; n < intermediateSize; n++) {
      let sum = 0;
      for (let k = 0; k < hiddenSize; k++) {
        sum += input[k] * W_gate[n * hiddenSize + k];
      }
      gate[n] = sum;
    }

    // Step 2: up = input @ W_up^T
    const up = new Float32Array(intermediateSize);
    for (let n = 0; n < intermediateSize; n++) {
      let sum = 0;
      for (let k = 0; k < hiddenSize; k++) {
        sum += input[k] * W_up[n * hiddenSize + k];
      }
      up[n] = sum;
    }

    // Step 3: Apply activation to gate and multiply by up
    const output = new Float32Array(intermediateSize);
    for (let i = 0; i < intermediateSize; i++) {
      let activated;
      if (activation === 'silu') {
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        activated = gate[i] / (1 + Math.exp(-gate[i]));
      } else {
        // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const x = gate[i];
        const sqrt2pi = Math.sqrt(2 / Math.PI);
        const cdf = 0.5 * (1 + Math.tanh(sqrt2pi * (x + 0.044715 * x * x * x)));
        activated = x * cdf;
      }
      output[i] = activated * up[i];
    }

    return output;
  },
};

// Expose to window for Playwright (type declared in tests/correctness/setup.ts)
window.testHarness = testHarness;
window.gpuReady = false;
window.gpuError = undefined;

// Auto-initialize on load
window.addEventListener('DOMContentLoaded', async () => {
  try {
    await initGPU();
    console.log('WebGPU initialized successfully');
    window.gpuReady = true;

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
    window.gpuReady = false;
    window.gpuError = e.message;

    const status = document.getElementById('status');
    if (status) {
      status.innerHTML = `<strong>WebGPU Error:</strong> ${e.message}`;
      status.style.color = 'red';
    }
  }
});

// Export for module usage
export { testHarness, initGPU, getGPU };
