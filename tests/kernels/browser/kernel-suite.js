import { testHarness as defaultHarness } from './test-page.js';

function generateUint32(size, max, seed = 123) {
  const out = new Uint32Array(size);
  let state = seed >>> 0;
  for (let i = 0; i < size; i++) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[i] = max === 0 ? 0 : state % max;
  }
  return out;
}

function generateNormalizedWeights(numTokens, topK, seed = 321) {
  const weights = new Float32Array(numTokens * topK);
  let state = seed >>> 0;
  for (let t = 0; t < numTokens; t++) {
    let sum = 0;
    for (let k = 0; k < topK; k++) {
      state = (state * 1103515245 + 12345) >>> 0;
      const val = (state & 0xffff) / 0xffff;
      weights[t * topK + k] = val;
      sum += val;
    }
    if (sum > 0) {
      for (let k = 0; k < topK; k++) {
        weights[t * topK + k] /= sum;
      }
    }
  }
  return weights;
}

function compareExact(expected, actual) {
  if (expected.length !== actual.length) return false;
  for (let i = 0; i < expected.length; i++) {
    if (expected[i] !== actual[i]) return false;
  }
  return true;
}

function f16ToF32(bits) {
  const sign = (bits & 0x8000) >> 15;
  const exp = (bits & 0x7c00) >> 10;
  const mant = bits & 0x03ff;
  if (exp === 0) {
    if (mant === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024);
  }
  if (exp === 31) {
    return mant === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
}

function f32ToF16Bits(value) {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, value, true);
  const bits = view.getUint32(0, true);
  const sign = (bits >> 31) & 1;
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
  return (sign << 15) | (hExp << 10) | hMant;
}

function f32ArrayToF16(arr) {
  const out = new Uint16Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    out[i] = f32ToF16Bits(arr[i]);
  }
  return out;
}

function bf16ToF32Array(bf16) {
  const out = new Float32Array(bf16.length);
  for (let i = 0; i < bf16.length; i++) {
    const view = new DataView(new ArrayBuffer(4));
    view.setUint32(0, bf16[i] << 16, true);
    out[i] = view.getFloat32(0, true);
  }
  return out;
}

function bf16ToF16Array(bf16) {
  const asF32 = bf16ToF32Array(bf16);
  return f32ArrayToF16(asF32);
}

function isFiniteArray(arr) {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) return false;
  }
  return true;
}

function transposeRef(input, rows, cols) {
  const output = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      output[c * rows + r] = input[r * cols + c];
    }
  }
  return output;
}

function matmulRefRowMajorB(input, weights, M, N, K) {
  const output = new Float32Array(M * N);
  for (let m = 0; m < M; m++) {
    const aBase = m * K;
    for (let n = 0; n < N; n++) {
      const bBase = n * K;
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += input[aBase + k] * weights[bBase + k];
      }
      output[m * N + n] = sum;
    }
  }
  return output;
}

function biasAddRef(data, bias, numTokens, dim) {
  const output = new Float32Array(data.length);
  for (let t = 0; t < numTokens; t++) {
    const base = t * dim;
    for (let d = 0; d < dim; d++) {
      output[base + d] = data[base + d] + bias[d];
    }
  }
  return output;
}

function layerNormRef(data, weight, bias, batchSize, hiddenSize, eps) {
  const output = new Float32Array(batchSize * hiddenSize);
  for (let b = 0; b < batchSize; b++) {
    const base = b * hiddenSize;
    let mean = 0;
    for (let i = 0; i < hiddenSize; i++) {
      mean += data[base + i];
    }
    mean /= hiddenSize;
    let varSum = 0;
    for (let i = 0; i < hiddenSize; i++) {
      const diff = data[base + i] - mean;
      varSum += diff * diff;
    }
    const invStd = 1 / Math.sqrt(varSum / hiddenSize + eps);
    for (let i = 0; i < hiddenSize; i++) {
      const norm = (data[base + i] - mean) * invStd;
      output[base + i] = norm * weight[i] + bias[i];
    }
  }
  return output;
}

function groupNormRef(data, weight, bias, channels, height, width, numGroups, eps) {
  const output = new Float32Array(channels * height * width);
  const channelsPerGroup = Math.floor(channels / numGroups);
  const spatial = height * width;
  for (let g = 0; g < numGroups; g++) {
    const cStart = g * channelsPerGroup;
    const cEnd = cStart + channelsPerGroup;
    let mean = 0;
    let count = 0;
    for (let c = cStart; c < cEnd; c++) {
      const base = c * spatial;
      for (let i = 0; i < spatial; i++) {
        mean += data[base + i];
        count++;
      }
    }
    mean /= count;
    let varSum = 0;
    for (let c = cStart; c < cEnd; c++) {
      const base = c * spatial;
      for (let i = 0; i < spatial; i++) {
        const diff = data[base + i] - mean;
        varSum += diff * diff;
      }
    }
    const invStd = 1 / Math.sqrt(varSum / count + eps);
    for (let c = cStart; c < cEnd; c++) {
      const base = c * spatial;
      for (let i = 0; i < spatial; i++) {
        const norm = (data[base + i] - mean) * invStd;
        output[base + i] = norm * weight[c] + bias[c];
      }
    }
  }
  return output;
}

function modulateRef(input, mod, numTokens, hiddenSize, options) {
  const {
    scaleOffset = 0,
    shiftOffset = hiddenSize,
    gateOffset = hiddenSize * 2,
    hasGate = false,
    addOne = true,
  } = options;
  const output = new Float32Array(numTokens * hiddenSize);
  for (let t = 0; t < numTokens; t++) {
    const base = t * hiddenSize;
    for (let d = 0; d < hiddenSize; d++) {
      const rawScale = mod[scaleOffset + d];
      const shift = mod[shiftOffset + d];
      const scale = addOne ? 1 + rawScale : rawScale;
      let value = input[base + d] * scale + shift;
      if (hasGate) {
        value *= mod[gateOffset + d];
      }
      output[base + d] = value;
    }
  }
  return output;
}

function conv2dRef(input, weight, bias, options) {
  const { inChannels, outChannels, height, width, kernelH, kernelW, stride = 1, pad = 0 } = options;
  const outHeight = Math.floor((height + pad * 2 - kernelH) / stride) + 1;
  const outWidth = Math.floor((width + pad * 2 - kernelW) / stride) + 1;
  const output = new Float32Array(outChannels * outHeight * outWidth);
  for (let oc = 0; oc < outChannels; oc++) {
    for (let oy = 0; oy < outHeight; oy++) {
      for (let ox = 0; ox < outWidth; ox++) {
        let sum = bias ? bias[oc] : 0;
        for (let ic = 0; ic < inChannels; ic++) {
          for (let ky = 0; ky < kernelH; ky++) {
            const inY = oy * stride + ky - pad;
            if (inY < 0 || inY >= height) continue;
            for (let kx = 0; kx < kernelW; kx++) {
              const inX = ox * stride + kx - pad;
              if (inX < 0 || inX >= width) continue;
              const inputIdx = (ic * height + inY) * width + inX;
              const weightIdx = (((oc * inChannels + ic) * kernelH + ky) * kernelW + kx);
              sum += input[inputIdx] * weight[weightIdx];
            }
          }
        }
        output[(oc * outHeight + oy) * outWidth + ox] = sum;
      }
    }
  }
  return output;
}

function pixelShuffleRef(input, options) {
  const { outChannels, outHeight, outWidth, gridWidth, patchSize, patchChannels } = options;
  const output = new Float32Array(outChannels * outHeight * outWidth);
  const spatial = outHeight * outWidth;
  for (let idx = 0; idx < output.length; idx++) {
    const c = Math.floor(idx / spatial);
    const rem = idx - c * spatial;
    const y = Math.floor(rem / outWidth);
    const x = rem - y * outWidth;
    const gridY = Math.floor(y / patchSize);
    const gridX = Math.floor(x / patchSize);
    const subY = y - gridY * patchSize;
    const subX = x - gridX * patchSize;
    const tokenIdx = gridY * gridWidth + gridX;
    const patchIdx = (subY * patchSize + subX) * outChannels + c;
    const inputIdx = tokenIdx * patchChannels + patchIdx;
    output[idx] = input[inputIdx];
  }
  return output;
}

function upsample2dRef(input, options) {
  const { channels, height, width, inHeight, inWidth, outHeight, outWidth, scale } = options;
  const srcHeight = Number.isFinite(height) ? height : inHeight;
  const srcWidth = Number.isFinite(width) ? width : inWidth;
  const output = new Float32Array(channels * outHeight * outWidth);
  for (let c = 0; c < channels; c++) {
    for (let oy = 0; oy < outHeight; oy++) {
      for (let ox = 0; ox < outWidth; ox++) {
        const inY = Math.floor(oy / scale);
        const inX = Math.floor(ox / scale);
        const inIdx = (c * srcHeight + inY) * srcWidth + inX;
        const outIdx = (c * outHeight + oy) * outWidth + ox;
        output[outIdx] = input[inIdx];
      }
    }
  }
  return output;
}

function softmaxBackwardRef(softmax, gradOutput, rows, cols) {
  const output = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      sum += softmax[base + c] * gradOutput[base + c];
    }
    for (let c = 0; c < cols; c++) {
      const idx = base + c;
      output[idx] = softmax[idx] * (gradOutput[idx] - sum);
    }
  }
  return output;
}

function siluBackwardRef(input, gradOutput) {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const x = input[i];
    const sigmoid = 1 / (1 + Math.exp(-x));
    const deriv = sigmoid * (1 + x * (1 - sigmoid));
    output[i] = gradOutput[i] * deriv;
  }
  return output;
}

function geluBackwardRef(input, gradOutput) {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const x = input[i];
    const sqrt2pi = 0.7978845608;
    const c = 0.044715;
    const x3 = x * x * x;
    const inner = sqrt2pi * (x + c * x3);
    const innerClamped = Math.max(-15, Math.min(15, inner));
    const tanhInner = Math.tanh(innerClamped);
    const sech2 = 1 - tanhInner * tanhInner;
    const innerDeriv = sqrt2pi * (1 + 3 * c * x * x);
    const deriv = 0.5 * (1 + tanhInner) + 0.5 * x * sech2 * innerDeriv;
    output[i] = gradOutput[i] * deriv;
  }
  return output;
}

function scaleBackwardRef(gradOutput, scale) {
  const output = new Float32Array(gradOutput.length);
  for (let i = 0; i < gradOutput.length; i++) {
    output[i] = gradOutput[i] * scale;
  }
  return output;
}

function ropeBackwardRef(gradOutput, cos, sin, seqLen, numHeads, headDim, startPos) {
  const output = new Float32Array(gradOutput.length);
  const halfDim = headDim / 2;
  for (let pos = 0; pos < seqLen; pos++) {
    for (let h = 0; h < numHeads; h++) {
      const base = (pos * numHeads + h) * headDim;
      const freqBase = (startPos + pos) * halfDim;
      for (let i = 0; i < halfDim; i++) {
        const dy0 = gradOutput[base + i];
        const dy1 = gradOutput[base + i + halfDim];
        const c = cos[freqBase + i];
        const s = sin[freqBase + i];
        output[base + i] = dy0 * c + dy1 * s;
        output[base + i + halfDim] = -dy0 * s + dy1 * c;
      }
    }
  }
  return output;
}

function rmsNormBackwardRef(input, weight, gradOutput, numTokens, hiddenSize, eps) {
  const output = new Float32Array(numTokens * hiddenSize);
  for (let t = 0; t < numTokens; t++) {
    const base = t * hiddenSize;
    let sumSq = 0;
    let sumGX = 0;
    for (let i = 0; i < hiddenSize; i++) {
      const x = input[base + i];
      const g = gradOutput[base + i] * weight[i];
      sumSq += x * x;
      sumGX += g * x;
    }
    const invRms = 1 / Math.sqrt(sumSq / hiddenSize + eps);
    const invRms3 = invRms * invRms * invRms;
    const coeff = (sumGX / hiddenSize) * invRms3;
    for (let i = 0; i < hiddenSize; i++) {
      const x = input[base + i];
      const g = gradOutput[base + i] * weight[i];
      output[base + i] = g * invRms - x * coeff;
    }
  }
  return output;
}

function crossEntropyLossRef(softmax, targets, numTokens, vocabSize) {
  const output = new Float32Array(numTokens);
  for (let t = 0; t < numTokens; t++) {
    const target = targets[t];
    if (target >= vocabSize) {
      output[t] = 0;
      continue;
    }
    const p = Math.max(softmax[t * vocabSize + target], 1e-9);
    output[t] = -Math.log(p);
  }
  return output;
}

function crossEntropyBackwardRef(softmax, targets, gradOutput, numTokens, vocabSize) {
  const output = new Float32Array(numTokens * vocabSize);
  for (let t = 0; t < numTokens; t++) {
    const target = targets[t];
    for (let c = 0; c < vocabSize; c++) {
      let grad = softmax[t * vocabSize + c];
      if (c === target) grad -= 1;
      output[t * vocabSize + c] = grad * gradOutput[t];
    }
  }
  return output;
}

function embedBackwardRef(gradOutput) {
  return new Float32Array(gradOutput);
}

function adamRef(params, grads, moment1, moment2, options) {
  const { count, step, lr, beta1, beta2, eps } = options;
  const outParams = new Float32Array(params);
  const outM1 = new Float32Array(moment1);
  const outM2 = new Float32Array(moment2);
  for (let i = 0; i < count; i++) {
    const g = grads[i];
    outM1[i] = beta1 * outM1[i] + (1 - beta1) * g;
    outM2[i] = beta2 * outM2[i] + (1 - beta2) * g * g;
    const mHat = outM1[i] / (1 - Math.pow(beta1, step));
    const vHat = outM2[i] / (1 - Math.pow(beta2, step));
    outParams[i] = outParams[i] - lr * mHat / (Math.sqrt(vHat) + eps);
  }
  return { params: outParams, moment1: outM1, moment2: outM2 };
}

export async function runKernelSuite(harness) {
  const h = harness || (typeof window !== 'undefined' ? window.testHarness : null) || defaultHarness;
  if (!h) throw new Error('Kernel harness not ready');

  const tests = [];

  tests.push([
    'matmul_f32',
    async () => {
      const M = 32, N = 48, K = 16;
      const A = h.generateTestData(M * K, 42);
      const B = h.generateTestData(K * N, 1337);
      const expected = h.matmulRef(A, B, M, N, K, 1.0);
      const actual = await h.runMatmul(null, A, B, M, N, K, 1.0);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.matmul_f32);
      return result.passed;
    },
  ]);

  tests.push([
    'matmul_q4k',
    async () => {
      const M = 2, N = 4, K = 256;
      const A = h.generateTestData(M * K, 2101, { min: -0.5, max: 0.5 });
      const B = h.generateTestData(N * K, 2102, { min: -0.5, max: 0.5 });
      const numBlocks = N;
      const B_q4k = h.references.quantizeQ4_KRef(B, numBlocks);
      const B_dequant = h.references.dequantQ4_KRef(B_q4k, numBlocks);
      const expected = matmulRefRowMajorB(A, B_dequant, M, N, K);
      const actual = await h.runMatmulQ4K(null, A, B_q4k, M, N, K, 1.0);
      const result = h.compareArrays(expected, actual, { rtol: 1e-2, atol: 1e-2 });
      return result.passed;
    },
  ]);

  tests.push([
    'matmul_f16w',
    async () => {
      const M = 2, N = 6, K = 32;
      const A = h.generateTestData(M * K, 2201);
      const B = h.generateTestData(N * K, 2202);
      const B_f16 = f32ArrayToF16(B);
      const B_rounded = new Float32Array(B_f16.length);
      for (let i = 0; i < B_f16.length; i++) {
        B_rounded[i] = f16ToF32(B_f16[i]);
      }
      const expected = matmulRefRowMajorB(A, B_rounded, M, N, K);
      const actual = await h.runMatmulF16W(null, A, B_f16, M, N, K);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.matmul_f16);
      return result.passed;
    },
  ]);

  tests.push([
    'dequant_matmul_f16w',
    async () => {
      const M = 2, N = 4, K = 256;
      const A = h.generateTestData(M * K, 2301, { min: -0.5, max: 0.5 });
      const B = h.generateTestData(N * K, 2302, { min: -0.5, max: 0.5 });
      const numBlocks = N;
      const B_q4k = h.references.quantizeQ4_KRef(B, numBlocks);
      const B_dequant = h.references.dequantQ4_KRef(B_q4k, numBlocks);
      const B_f16 = f32ArrayToF16(B_dequant);
      const B_rounded = new Float32Array(B_f16.length);
      for (let i = 0; i < B_f16.length; i++) {
        B_rounded[i] = f16ToF32(B_f16[i]);
      }
      const expected = matmulRefRowMajorB(A, B_rounded, M, N, K);
      const actual = await h.runDequantAndMatmulF16W(null, A, B_q4k, M, N, K, numBlocks);
      const result = h.compareArrays(expected, actual, { rtol: 1e-2, atol: 1e-2 });
      return result.passed;
    },
  ]);

  tests.push([
    'softmax',
    async () => {
      const outerSize = 4;
      const innerSize = 32;
      const input = h.generateTestData(outerSize * innerSize, 7);
      const expected = h.softmax(input, innerSize, outerSize, 1.0);
      const actual = await h.runSoftmax(null, input, innerSize, outerSize, 1.0);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.softmax);
      return result.passed;
    },
  ]);

  tests.push([
    'rmsnorm',
    async () => {
      const numTokens = 4;
      const hiddenSize = 32;
      const input = h.generateTestData(numTokens * hiddenSize, 91);
      const weight = h.generateTestData(hiddenSize, 92);
      const expected = h.references.rmsNormRef(input, weight, numTokens, hiddenSize, 1e-6);
      const actual = await h.runRMSNorm(null, input, weight, numTokens, hiddenSize, 1e-6);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.rmsnorm);
      return result.passed;
    },
  ]);

  tests.push([
    'rope',
    async () => {
      const seqLen = 4;
      const numHeads = 2;
      const headDim = 8;
      const input = h.generateTestData(seqLen * numHeads * headDim, 314);
      const { cos, sin } = h.references.computeRopeFreqs(headDim, seqLen);
      const expected = h.references.ropeRef(input, cos, sin, seqLen, numHeads, headDim, 0);
      const actual = await h.runRoPE(null, input, seqLen, numHeads, headDim, 0);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.rope);
      return result.passed;
    },
  ]);

  tests.push([
    'silu',
    async () => {
      const input = h.generateTestData(1024, 123);
      const expected = h.references.siluRef(input);
      const actual = await h.runSiLU(null, input);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.silu);
      return result.passed;
    },
  ]);

  tests.push([
    'silu_gated',
    async () => {
      const gate = h.generateTestData(1024, 124);
      const up = h.generateTestData(1024, 125);
      const expected = h.references.siluGatedRef(gate, up);
      const actual = await h.runSiLUGated(null, gate, up);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.silu);
      return result.passed;
    },
  ]);

  tests.push([
    'gelu',
    async () => {
      const input = h.generateTestData(1024, 201);
      const expected = h.references.geluRef(input);
      const actual = await h.runGeLU(null, input);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.silu);
      return result.passed;
    },
  ]);

  tests.push([
    'geglu',
    async () => {
      const gate = h.generateTestData(1024, 202);
      const up = h.generateTestData(1024, 203);
      const expected = h.references.gegluRef(gate, up);
      const actual = await h.runGeGLU(null, gate, up);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.silu);
      return result.passed;
    },
  ]);

  tests.push([
    'gather',
    async () => {
      const vocabSize = 64;
      const embedDim = 16;
      const numTokens = 8;
      const embeddings = h.generateTestData(vocabSize * embedDim, 401);
      const indices = generateUint32(numTokens, vocabSize, 402);
      const expected = h.references.gatherRef(embeddings, indices, vocabSize, embedDim);
      const actual = await h.runGather(null, embeddings, indices, vocabSize, embedDim);
      const result = h.compareArrays(expected, actual, { rtol: 0, atol: 0 });
      return result.passed;
    },
  ]);

  tests.push([
    'residual',
    async () => {
      const x = h.generateTestData(512, 501);
      const residual = h.generateTestData(512, 502);
      const expected = h.references.residualAddRef(x, residual);
      const actual = await h.runResidual(null, x, residual);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.residual);
      return result.passed;
    },
  ]);

  tests.push([
    'topk',
    async () => {
      const numTokens = 2;
      const numExperts = 16;
      const topK = 4;
      const probs = h.generateTestData(numTokens * numExperts, 601, { min: 0, max: 1 });
      const expected = h.references.topkRef(probs, numTokens, numExperts, topK, true);
      const actual = await h.runTopK(null, probs, numTokens, numExperts, topK, { normalize: true });
      const indicesMatch = compareExact(expected.indices, actual.indices);
      const weightsMatch = h.compareArrays(
        expected.weights,
        actual.weights,
        h.KERNEL_TOLERANCES.topk.weights
      ).passed;
      return indicesMatch && weightsMatch;
    },
  ]);

  tests.push([
    'softmax_topk',
    async () => {
      const numTokens = 2;
      const numExperts = 16;
      const topK = 4;
      const logits = h.generateTestData(numTokens * numExperts, 602);
      const expected = h.references.softmaxTopkRef(logits, numTokens, numExperts, topK, true);
      const actual = await h.runSoftmaxTopK(null, logits, numTokens, numExperts, topK, { normalize: true });
      const indicesMatch = compareExact(expected.indices, actual.indices);
      const weightsMatch = h.compareArrays(
        expected.weights,
        actual.weights,
        h.KERNEL_TOLERANCES.topk.weights
      ).passed;
      return indicesMatch && weightsMatch;
    },
  ]);

  tests.push([
    'scatter_add',
    async () => {
      const numTokens = 4;
      const hiddenSize = 8;
      const numExperts = 3;
      const topK = 2;
      const expertOutputs = h.generateTestData(numExperts * numTokens * hiddenSize, 701);
      const indices = generateUint32(numTokens * topK, numExperts, 702);
      const weights = generateNormalizedWeights(numTokens, topK, 703);
      const expected = h.references.scatterAddRef(
        expertOutputs,
        indices,
        weights,
        numTokens,
        hiddenSize,
        numExperts,
        topK
      );
      const actual = await h.runScatterAdd(
        null,
        expertOutputs,
        indices,
        weights,
        numTokens,
        hiddenSize,
        numExperts,
        topK
      );
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.scatter_add);
      return result.passed;
    },
  ]);

  tests.push([
    'moe_gather',
    async () => {
      const numTokens = 4;
      const hiddenSize = 8;
      const numExperts = 3;
      const topK = 2;
      const tokens = h.generateTestData(numTokens * hiddenSize, 801);
      const indices = generateUint32(numTokens * topK, numExperts, 802);
      const expected = h.references.moeGatherRef(tokens, indices, numTokens, hiddenSize, numExperts, topK);
      const actual = await h.runMoEGather(null, tokens, indices, numTokens, hiddenSize, numExperts, topK);
      const tokensMatch = h.compareArrays(
        expected.gatheredTokens,
        actual.gatheredTokens,
        h.KERNEL_TOLERANCES.moe_gather
      ).passed;
      const countsMatch = compareExact(expected.tokenCounts, actual.tokenCounts);
      return tokensMatch && countsMatch;
    },
  ]);

  tests.push([
    'attention',
    async () => {
      const seqLen = 2;
      const kvLen = 4;
      const numHeads = 2;
      const numKVHeads = 1;
      const headDim = 8;
      const Q = h.generateTestData(seqLen * numHeads * headDim, 901);
      const K = h.generateTestData(kvLen * numKVHeads * headDim, 902);
      const V = h.generateTestData(kvLen * numKVHeads * headDim, 903);
      const expected = h.references.attentionRef(Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim, null);
      const actual = await h.runAttention(null, Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim, null);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.attention);
      return result.passed;
    },
  ]);

  tests.push([
    'dequant_q4k_f32',
    async () => {
      const numBlocks = 2;
      const values = h.generateTestData(numBlocks * 256, 1001);
      const quantized = h.references.quantizeQ4_KRef(values, numBlocks);
      const expected = h.references.dequantQ4_KRef(quantized, numBlocks);
      const actual = await h.runDequantQ4K(null, quantized, numBlocks);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.dequant);
      return result.passed;
    },
  ]);

  tests.push([
    'dequant_q4k_f16',
    async () => {
      const numBlocks = 2;
      const values = h.generateTestData(numBlocks * 256, 1002);
      const quantized = h.references.quantizeQ4_KRef(values, numBlocks);
      const expected = h.references.dequantQ4_KRef(quantized, numBlocks);
      const expectedF16 = f32ArrayToF16(expected);
      const expectedRounded = new Float32Array(expectedF16.length);
      for (let i = 0; i < expectedF16.length; i++) {
        expectedRounded[i] = f16ToF32(expectedF16[i]);
      }
      const actual = await h.runDequantQ4K_F16(null, quantized, numBlocks);
      const result = h.compareArrays(expectedRounded, actual, { rtol: 1e-3, atol: 1e-3 });
      return result.passed;
    },
  ]);

  tests.push([
    'dequant_q6k_smoke',
    async () => {
      const numBlocks = 1;
      const quantized = new Uint8Array(numBlocks * 210);
      const actual = await h.runDequantQ6K(null, quantized, numBlocks);
      return actual.length === numBlocks * 256 && isFiniteArray(actual);
    },
  ]);

  tests.push([
    'cast_f32_to_f16',
    async () => {
      const input = h.generateTestData(128, 1101);
      const expectedBits = f32ArrayToF16(input);
      const actualBits = await h.runF32ToF16(null, input);
      const expected = new Float32Array(expectedBits.length);
      const actual = new Float32Array(actualBits.length);
      for (let i = 0; i < expectedBits.length; i++) {
        expected[i] = f16ToF32(expectedBits[i]);
        actual[i] = f16ToF32(actualBits[i]);
      }
      const result = h.compareArrays(expected, actual, { rtol: 1e-3, atol: 1e-3 });
      return result.passed;
    },
  ]);

  tests.push([
    'cast_f16_to_f32',
    async () => {
      const input = f32ArrayToF16(h.generateTestData(128, 1102));
      const expected = new Float32Array(input.length);
      for (let i = 0; i < input.length; i++) {
        expected[i] = f16ToF32(input[i]);
      }
      const actual = await h.runF16ToF32(null, input);
      const result = h.compareArrays(expected, actual, { rtol: 1e-5, atol: 1e-6 });
      return result.passed;
    },
  ]);

  tests.push([
    'bf16_to_f32',
    async () => {
      const floats = h.generateTestData(128, 1103);
      const bf16 = new Uint16Array(floats.length);
      const view = new DataView(new ArrayBuffer(4));
      for (let i = 0; i < floats.length; i++) {
        view.setFloat32(0, floats[i], true);
        bf16[i] = view.getUint32(0, true) >>> 16;
      }
      const expected = bf16ToF32Array(bf16);
      const actual = await h.runBF16ToF32(null, bf16);
      const result = h.compareArrays(expected, actual, { rtol: 1e-5, atol: 1e-6 });
      return result.passed;
    },
  ]);

  tests.push([
    'bf16_to_f16',
    async () => {
      const floats = h.generateTestData(128, 1104);
      const bf16 = new Uint16Array(floats.length);
      const view = new DataView(new ArrayBuffer(4));
      for (let i = 0; i < floats.length; i++) {
        view.setFloat32(0, floats[i], true);
        bf16[i] = view.getUint32(0, true) >>> 16;
      }
      const expected = bf16ToF16Array(bf16);
      const actual = await h.runBF16ToF16(null, bf16);
      return compareExact(expected, actual);
    },
  ]);

  tests.push([
    'bias_add',
    async () => {
      const numTokens = 4;
      const dim = 16;
      const data = h.generateTestData(numTokens * dim, 1201);
      const bias = h.generateTestData(dim, 1202);
      const expected = biasAddRef(data, bias, numTokens, dim);
      const actual = await h.runBiasAdd(null, data, bias, numTokens, dim);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.residual);
      return result.passed;
    },
  ]);

  tests.push([
    'scale',
    async () => {
      const input = h.generateTestData(256, 1203);
      const scale = 0.5;
      const expected = input.map((v) => v * scale);
      const actual = await h.runScale(null, input, scale);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.residual);
      return result.passed;
    },
  ]);

  tests.push([
    'scale_backward',
    async () => {
      const input = h.generateTestData(256, 1204);
      const gradOutput = h.generateTestData(256, 1205);
      const scale = 0.75;
      const expected = scaleBackwardRef(gradOutput, scale);
      const actual = await h.runScaleBackward(null, input, gradOutput, scale);
      const result = h.compareArrays(expected, actual, { rtol: 1e-5, atol: 1e-6 });
      if (!result.passed) {
        console.error('[KernelTests] scale_backward mismatch', result);
      }
      return result.passed;
    },
  ]);

  tests.push([
    'softmax_backward',
    async () => {
      const rows = 4;
      const cols = 16;
      const logits = h.generateTestData(rows * cols, 1206);
      const softmax = h.softmax(logits, cols, rows, 1.0);
      const gradOutput = h.generateTestData(rows * cols, 1207);
      const expected = softmaxBackwardRef(softmax, gradOutput, rows, cols);
      const actual = await h.runSoftmaxBackward(null, softmax, gradOutput, rows, cols);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.softmax);
      return result.passed;
    },
  ]);

  tests.push([
    'silu_backward',
    async () => {
      const input = h.generateTestData(512, 1208);
      const gradOutput = h.generateTestData(512, 1209);
      const expected = siluBackwardRef(input, gradOutput);
      const actual = await h.runSiluBackward(null, input, gradOutput);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.silu);
      return result.passed;
    },
  ]);

  tests.push([
    'gelu_backward',
    async () => {
      const input = h.generateTestData(512, 1210);
      const gradOutput = h.generateTestData(512, 1211);
      const expected = geluBackwardRef(input, gradOutput);
      const actual = await h.runGeluBackward(null, input, gradOutput);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.silu);
      return result.passed;
    },
  ]);

  tests.push([
    'rmsnorm_backward',
    async () => {
      const numTokens = 4;
      const hiddenSize = 16;
      const input = h.generateTestData(numTokens * hiddenSize, 1212);
      const weight = h.generateTestData(hiddenSize, 1213);
      const gradOutput = h.generateTestData(numTokens * hiddenSize, 1214);
      const expected = rmsNormBackwardRef(input, weight, gradOutput, numTokens, hiddenSize, 1e-6);
      const actual = await h.runRmsNormBackward(null, input, weight, gradOutput, numTokens, hiddenSize, 1e-6);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.rmsnorm);
      return result.passed;
    },
  ]);

  tests.push([
    'rope_backward',
    async () => {
      const seqLen = 4;
      const numHeads = 2;
      const headDim = 8;
      const gradOutput = h.generateTestData(seqLen * numHeads * headDim, 1215);
      const { cos, sin } = h.references.computeRopeFreqs(headDim, seqLen);
      const expected = ropeBackwardRef(gradOutput, cos, sin, seqLen, numHeads, headDim, 0);
      const actual = await h.runRoPEBackward(null, gradOutput, cos, sin, seqLen, numHeads, headDim, 0);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.rope);
      return result.passed;
    },
  ]);

  tests.push([
    'layernorm',
    async () => {
      const batchSize = 2;
      const hiddenSize = 16;
      const input = h.generateTestData(batchSize * hiddenSize, 1216);
      const weight = h.generateTestData(hiddenSize, 1217);
      const bias = h.generateTestData(hiddenSize, 1218);
      const expected = layerNormRef(input, weight, bias, batchSize, hiddenSize, 1e-5);
      const actual = await h.runLayerNorm(null, input, weight, bias, batchSize, hiddenSize, 1e-5);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.rmsnorm);
      return result.passed;
    },
  ]);

  tests.push([
    'groupnorm',
    async () => {
      const channels = 4;
      const height = 2;
      const width = 2;
      const numGroups = 2;
      const input = h.generateTestData(channels * height * width, 1219);
      const weight = h.generateTestData(channels, 1220);
      const bias = h.generateTestData(channels, 1221);
      const expected = groupNormRef(input, weight, bias, channels, height, width, numGroups, 1e-5);
      const actual = await h.runGroupNorm(null, input, weight, bias, channels, height, width, numGroups, 1e-5);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.rmsnorm);
      return result.passed;
    },
  ]);

  tests.push([
    'groupnorm_stats',
    async () => {
      const channels = 4;
      const height = 2;
      const width = 2;
      const numGroups = 2;
      const input = h.generateTestData(channels * height * width, 1222);
      const weight = h.generateTestData(channels, 1223);
      const bias = h.generateTestData(channels, 1224);
      const actual = await h.runGroupNorm(null, input, weight, bias, channels, height, width, numGroups, 1e-5);
      return actual.length === channels * height * width && isFiniteArray(actual);
    },
  ]);

  tests.push([
    'modulate',
    async () => {
      const numTokens = 2;
      const hiddenSize = 8;
      const input = h.generateTestData(numTokens * hiddenSize, 1222);
      const mod = h.generateTestData(hiddenSize * 3, 1223);
      const options = { scaleOffset: 0, shiftOffset: hiddenSize, gateOffset: hiddenSize * 2, hasGate: true, addOne: true };
      const expected = modulateRef(input, mod, numTokens, hiddenSize, options);
      const actual = await h.runModulate(null, input, mod, numTokens, hiddenSize, options);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.residual);
      return result.passed;
    },
  ]);

  tests.push([
    'conv2d',
    async () => {
      const options = {
        inChannels: 2,
        outChannels: 2,
        height: 3,
        width: 3,
        kernelH: 3,
        kernelW: 3,
        stride: 1,
        pad: 1,
      };
      const input = h.generateTestData(options.inChannels * options.height * options.width, 1224);
      const weight = h.generateTestData(options.outChannels * options.inChannels * options.kernelH * options.kernelW, 1225);
      const bias = h.generateTestData(options.outChannels, 1226);
      const expected = conv2dRef(input, weight, bias, options);
      const actual = await h.runConv2D(null, input, weight, bias, options);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.residual);
      return result.passed;
    },
  ]);

  tests.push([
    'pixel_shuffle',
    async () => {
      const options = {
        outChannels: 1,
        gridWidth: 2,
        gridHeight: 2,
        patchSize: 2,
      };
      options.outHeight = options.gridHeight * options.patchSize;
      options.outWidth = options.gridWidth * options.patchSize;
      options.patchChannels = options.outChannels * options.patchSize * options.patchSize;
      const input = h.generateTestData(options.gridWidth * options.gridHeight * options.patchChannels, 1227);
      const expected = pixelShuffleRef(input, options);
      const actual = await h.runPixelShuffle(null, input, options);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.residual);
      return result.passed;
    },
  ]);

  tests.push([
    'upsample2d',
    async () => {
      const options = {
        channels: 2,
        height: 2,
        width: 2,
        scale: 2,
      };
      options.outHeight = options.height * options.scale;
      options.outWidth = options.width * options.scale;
      const input = h.generateTestData(options.channels * options.height * options.width, 1228);
      const expected = upsample2dRef(input, options);
      const actual = await h.runUpsample2D(null, input, options);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.residual);
      return result.passed;
    },
  ]);

  tests.push([
    'transpose',
    async () => {
      const rows = 4;
      const cols = 6;
      const input = h.generateTestData(rows * cols, 1229);
      const expected = transposeRef(input, rows, cols);
      const actual = await h.runTranspose(null, input, rows, cols);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.residual);
      return result.passed;
    },
  ]);

  tests.push([
    'split_qkv',
    async () => {
      const numTokens = 2;
      const qSize = 4;
      const kSize = 4;
      const vSize = 4;
      const qkv = h.generateTestData(numTokens * (qSize + kSize + vSize), 1230);
      const expected = h.references.splitQkvRef(qkv, numTokens, qSize, kSize, vSize);
      const actual = await h.runSplitQKV(null, qkv, numTokens, qSize, kSize, vSize);
      const qPass = h.compareArrays(expected.Q, actual.Q, h.KERNEL_TOLERANCES.residual).passed;
      const kPass = h.compareArrays(expected.K, actual.K, h.KERNEL_TOLERANCES.residual).passed;
      const vPass = h.compareArrays(expected.V, actual.V, h.KERNEL_TOLERANCES.residual).passed;
      return qPass && kPass && vPass;
    },
  ]);

  tests.push([
    'swiglu',
    async () => {
      const size = 128;
      const gate = h.generateTestData(size, 1231);
      const up = h.generateTestData(size, 1232);
      const gateBias = h.generateTestData(size, 1233);
      const upBias = h.generateTestData(size, 1234);
      const expected = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        const g = gate[i] + gateBias[i];
        const u = up[i] + upBias[i];
        const silu = g / (1 + Math.exp(-g));
        expected[i] = silu * u;
      }
      const actual = await h.runSwiGLU(null, gate, up, gateBias, upBias);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.silu);
      return result.passed;
    },
  ]);

  tests.push([
    'fused_matmul_residual',
    async () => {
      const N = 16;
      const K = 8;
      const input = h.generateTestData(K, 1301);
      const weight = h.generateTestData(N * K, 1302);
      const residual = h.generateTestData(N, 1303);
      const expected = h.fusedMatmulResidualRef(input, weight, residual, N, K, 1.0);
      const actual = await h.runFusedMatmulResidual(null, input, weight, residual, N, K, 1.0);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.residual);
      return result.passed;
    },
  ]);

  tests.push([
    'fused_matmul_rmsnorm',
    async () => {
      const N = 16;
      const K = 8;
      const input = h.generateTestData(K, 1304);
      const weight = h.generateTestData(N * K, 1305);
      const normWeight = h.generateTestData(N, 1306);
      const expected = h.fusedMatmulRMSNormRef(input, weight, normWeight, N, K, 1e-5, null);
      const actual = await h.runFusedMatmulRMSNorm(null, input, weight, normWeight, N, K, 1e-5, null);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.rmsnorm);
      return result.passed;
    },
  ]);

  tests.push([
    'fused_ffn',
    async () => {
      const hiddenSize = 8;
      const intermediateSize = 16;
      const input = h.generateTestData(hiddenSize, 1307);
      const gateW = h.generateTestData(intermediateSize * hiddenSize, 1308);
      const upW = h.generateTestData(intermediateSize * hiddenSize, 1309);
      const expected = h.fusedFFNRef(input, gateW, upW, hiddenSize, intermediateSize, 'silu');
      const actual = await h.runFusedFFN(null, input, gateW, upW, hiddenSize, intermediateSize, 'silu');
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.silu);
      return result.passed;
    },
  ]);

  tests.push([
    'attention_decode_optimized',
    async () => {
      const numHeads = 2;
      const numKVHeads = 1;
      const headDim = 8;
      const kvLen = 4;
      const Q = h.generateTestData(numHeads * headDim, 1401);
      const K = h.generateTestData(kvLen * numKVHeads * headDim, 1402);
      const V = h.generateTestData(kvLen * numKVHeads * headDim, 1403);
      const expected = h.references.attentionRef(Q, K, V, 1, kvLen, numHeads, numKVHeads, headDim, null);
      const actual = await h.runAttentionDecodeOptimized(null, Q, K, V, numHeads, numKVHeads, headDim, kvLen);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.attention);
      return result.passed;
    },
  ]);

  tests.push([
    'attention_tiered',
    async () => {
      const seqLen = 1;
      const kvLen = 4;
      const numHeads = 2;
      const numKVHeads = 1;
      const headDim = 8;
      const Q = h.generateTestData(seqLen * numHeads * headDim, 1404);
      const K = h.generateTestData(kvLen * numKVHeads * headDim, 1405);
      const V = h.generateTestData(kvLen * numKVHeads * headDim, 1406);
      const qF16 = f32ArrayToF16(Q);
      const kF16 = f32ArrayToF16(K);
      const vF16 = f32ArrayToF16(V);
      const qRounded = new Float32Array(qF16.length);
      const kRounded = new Float32Array(kF16.length);
      const vRounded = new Float32Array(vF16.length);
      for (let i = 0; i < qF16.length; i++) qRounded[i] = f16ToF32(qF16[i]);
      for (let i = 0; i < kF16.length; i++) kRounded[i] = f16ToF32(kF16[i]);
      for (let i = 0; i < vF16.length; i++) vRounded[i] = f16ToF32(vF16[i]);
      const expected = h.toF16RoundedFloat32(
        h.references.attentionRef(qRounded, kRounded, vRounded, seqLen, kvLen, numHeads, numKVHeads, headDim, null)
      );
      const actual = await h.runAttentionTiered(null, Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim);
      const tieredTolerance = { rtol: 1e-3, atol: 1e-4 };
      const result = h.compareArrays(expected, actual, tieredTolerance);
      if (!result.passed) {
        console.error('[KernelTests] attention_tiered mismatch', result);
      }
      return result.passed;
    },
  ]);

  tests.push([
    'attention_tiered_quant',
    async () => {
      const seqLen = 1;
      const kvLen = 4;
      const numHeads = 2;
      const numKVHeads = 1;
      const headDim = 8;
      const Q = h.generateTestData(seqLen * numHeads * headDim, 1407);
      const K = h.generateTestData(kvLen * numKVHeads * headDim, 1408);
      const V = h.generateTestData(kvLen * numKVHeads * headDim, 1409);
      const kF16 = f32ArrayToF16(K);
      const vF16 = f32ArrayToF16(V);
      const kRounded = new Float32Array(kF16.length);
      const vRounded = new Float32Array(vF16.length);
      for (let i = 0; i < kF16.length; i++) kRounded[i] = f16ToF32(kF16[i]);
      for (let i = 0; i < vF16.length; i++) vRounded[i] = f16ToF32(vF16[i]);
      const expected = h.references.attentionRef(Q, kRounded, vRounded, seqLen, kvLen, numHeads, numKVHeads, headDim, null);
      const actual = await h.runAttentionTieredQuant(null, Q, K, V, seqLen, kvLen, numHeads, numKVHeads, headDim);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.attention);
      return result.passed;
    },
  ]);

  tests.push([
    'cross_entropy_loss',
    async () => {
      const numTokens = 4;
      const vocabSize = 16;
      const logits = h.generateTestData(numTokens * vocabSize, 1501);
      const softmax = h.softmax(logits, vocabSize, numTokens, 1.0);
      const targets = generateUint32(numTokens, vocabSize, 1502);
      const expected = crossEntropyLossRef(softmax, targets, numTokens, vocabSize);
      const actual = await h.runCrossEntropyLoss(null, softmax, targets, numTokens, vocabSize);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.residual);
      return result.passed;
    },
  ]);

  tests.push([
    'cross_entropy_backward',
    async () => {
      const numTokens = 4;
      const vocabSize = 16;
      const logits = h.generateTestData(numTokens * vocabSize, 1503);
      const softmax = h.softmax(logits, vocabSize, numTokens, 1.0);
      const targets = generateUint32(numTokens, vocabSize, 1504);
      const gradOutput = h.generateTestData(numTokens, 1505);
      const expected = crossEntropyBackwardRef(softmax, targets, gradOutput, numTokens, vocabSize);
      const actual = await h.runCrossEntropyBackward(null, softmax, targets, gradOutput, numTokens, vocabSize);
      const result = h.compareArrays(expected, actual, h.KERNEL_TOLERANCES.residual);
      return result.passed;
    },
  ]);

  tests.push([
    'kv_quantize_smoke',
    async () => {
      const numKVHeads = 1;
      const headDim = 4;
      const numTokens = 1;
      const keyVals = h.generateTestData(numKVHeads * numTokens * headDim, 1601);
      const valVals = h.generateTestData(numKVHeads * numTokens * headDim, 1602);
      const keys = f32ArrayToF16(keyVals);
      const values = f32ArrayToF16(valVals);
      const result = await h.runKVQuantize(null, keys, values, numKVHeads, headDim, numTokens, 'int8');
      return result.outputK.length > 0 && result.scalesK.length > 0;
    },
  ]);

  tests.push([
    'argmax',
    async () => {
      const logits = h.generateTestData(64, 1701);
      const expected = h.references.argmaxRef(logits);
      const actual = await h.runArgmax(null, logits);
      return expected === actual;
    },
  ]);

  tests.push([
    'sample_topk_smoke',
    async () => {
      const logits = h.generateTestData(64, 1702);
      const temperature = 1.0;
      const topK = 4;
      const randomValue = h.references.seededRandom(42);
      const actual = await h.runSampleTopK(null, logits, temperature, topK, randomValue);
      return Number.isFinite(actual) && actual >= 0 && actual < logits.length;
    },
  ]);

  tests.push([
    'check_stop',
    async () => {
      const eosTokenId = 2;
      const maxTokens = 5;
      const case1 = await h.runCheckStop(null, 1, eosTokenId, maxTokens, 3);
      const case2 = await h.runCheckStop(null, 2, eosTokenId, maxTokens, 3);
      const case3 = await h.runCheckStop(null, 1, eosTokenId, maxTokens, 5);
      return case1 === false && case2 === true && case3 === true;
    },
  ]);

  tests.push([
    'embed_backward',
    async () => {
      const input = h.generateTestData(64, 1801);
      const gradOutput = h.generateTestData(64, 1802);
      const expected = embedBackwardRef(gradOutput);
      const actual = await h.runEmbedBackward(null, input, gradOutput);
      const result = h.compareArrays(expected, actual, { rtol: 1e-5, atol: 1e-6 });
      if (!result.passed) {
        console.error('[KernelTests] embed_backward mismatch', result);
      }
      return result.passed;
    },
  ]);

  tests.push([
    'attention_backward_smoke',
    async () => {
      const seqLen = 2;
      const numHeads = 1;
      const headDim = 4;
      const scale = 1 / Math.sqrt(headDim);
      const q = h.generateTestData(seqLen * numHeads * headDim, 1803);
      const k = h.generateTestData(seqLen * numHeads * headDim, 1804);
      const v = h.generateTestData(seqLen * numHeads * headDim, 1805);
      const softmax = new Float32Array(numHeads * seqLen * seqLen);
      for (let hIdx = 0; hIdx < numHeads; hIdx++) {
        for (let i = 0; i < seqLen; i++) {
          let maxVal = -Infinity;
          for (let j = 0; j < seqLen; j++) {
            let score = 0;
            for (let d = 0; d < headDim; d++) {
              const qIdx = (i * numHeads + hIdx) * headDim + d;
              const kIdx = (j * numHeads + hIdx) * headDim + d;
              score += q[qIdx] * k[kIdx];
            }
            score *= scale;
            const idx = (hIdx * seqLen + i) * seqLen + j;
            softmax[idx] = score;
            maxVal = Math.max(maxVal, score);
          }
          let sum = 0;
          for (let j = 0; j < seqLen; j++) {
            const idx = (hIdx * seqLen + i) * seqLen + j;
            const val = Math.exp(softmax[idx] - maxVal);
            softmax[idx] = val;
            sum += val;
          }
          for (let j = 0; j < seqLen; j++) {
            const idx = (hIdx * seqLen + i) * seqLen + j;
            softmax[idx] /= sum;
          }
        }
      }
      const gradOutput = h.generateTestData(seqLen * numHeads * headDim, 1806);
      const result = await h.runAttentionBackward(null, q, k, v, softmax, gradOutput, seqLen, numHeads, headDim, scale);
      return result && isFiniteArray(result.gradQ) && isFiniteArray(result.gradK) && isFiniteArray(result.gradV);
    },
  ]);

  tests.push([
    'adam',
    async () => {
      const count = 32;
      const params = h.generateTestData(count, 1901);
      const grads = h.generateTestData(count, 1902);
      const moment1 = h.generateTestData(count, 1903);
      const moment2 = h.generateTestData(count, 1904);
      const options = { count, step: 2, lr: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
      const expected = adamRef(params, grads, moment1, moment2, options);
      const actual = await h.runAdam(null, params, grads, moment1, moment2, options);
      const pPass = h.compareArrays(expected.params, actual.params, h.KERNEL_TOLERANCES.residual).passed;
      const m1Pass = h.compareArrays(expected.moment1, actual.moment1, h.KERNEL_TOLERANCES.residual).passed;
      const m2Pass = h.compareArrays(expected.moment2, actual.moment2, h.KERNEL_TOLERANCES.residual).passed;
      return pPass && m1Pass && m2Pass;
    },
  ]);

  const results = [];
  for (const [name, fn] of tests) {
    const start = performance.now();
    try {
      const passed = await fn();
      results.push({ name, passed, duration: Math.round(performance.now() - start) });
    } catch (error) {
      console.error(`[KernelTests] ${name} failed:`, error);
      results.push({ name, passed: false, duration: Math.round(performance.now() - start) });
    }
  }

  return results;
}
