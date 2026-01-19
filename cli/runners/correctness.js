


import { setHarnessConfig, appendRuntimeConfigParams } from '../args/index.js';

export async function runCorrectnessTests(page, opts, tests) {
  console.log('\n' + '='.repeat(60));
  console.log('KERNEL CORRECTNESS TESTS');
  console.log('='.repeat(60));

  setHarnessConfig(opts, {
    mode: 'kernels',
    autorun: false,
    skipLoad: false,
    modelId: null,
  });
  const harnessParams = new URLSearchParams();
  appendRuntimeConfigParams(harnessParams, opts);
  await page.goto(`${opts.baseUrl}/doppler/tests/harness.html?${harnessParams.toString()}`, {
    timeout: opts.timeout,
  });

  await page.waitForTimeout(500);

  try {
    await page.evaluate(async () => {
      const w =  (window);
      if (!w.gpuReady && w.testHarness?.getGPU) {
        await w.testHarness.getGPU();
        w.gpuReady = true;
      }
    });
  } catch (err) {
    // Will be caught by waitForFunction below
  }

  await page.waitForFunction(
    () => {
      const w =  (window);
      if (w.gpuError) {
        throw new Error(`WebGPU init failed: ${w.gpuError}`);
      }
      return w.gpuReady === true && w.testHarness && w.testHarness.references;
    },
    { timeout: 30000 }
  );

    const results = [];
  const startTime = Date.now();

  const testsToRun = opts.filter
    ? tests.filter((t) => t.includes(opts.filter))
    : tests;

  for (const testName of testsToRun) {
    console.log(`\n  Running: ${testName}...`);
    const testStart = Date.now();

    try {
      const result = await page.evaluate(
        async (name) => {
          const harness =  (window).testHarness;
          const gpu = await harness.getGPU();
          const refs = harness.references;

          switch (name) {
            case 'matmul': {
              const M = 64, K = 128, N = 64;
              const A = new Float32Array(M * K).map(() => Math.random() * 2 - 1);
              const B = new Float32Array(K * N).map(() => Math.random() * 2 - 1);
              const ref = refs.matmulRef(A, B, M, N, K);
              const gpuResult = await harness.runMatmul(gpu.device, A, B, M, N, K);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-4, maxError };
            }

            case 'matmul-q4k': {
              const M = 8, K = 256, N = 32;
              const A = new Float32Array(M * K).map(() => (Math.random() * 2 - 1) * 0.5);
              const B_f32 = new Float32Array(N * K).map(() => (Math.random() * 2 - 1) * 0.5);
              const numBlocks = N * (K / 256);
              const B_q4k = refs.quantizeQ4_KRef(B_f32, numBlocks);
              const B_dequant = refs.dequantQ4_KRef(B_q4k, numBlocks);
              const refC = new Float32Array(M * N);
              for (let m = 0; m < M; m++) {
                for (let n = 0; n < N; n++) {
                  let sum = 0;
                  for (let k = 0; k < K; k++) {
                    sum += A[m * K + k] * B_dequant[n * K + k];
                  }
                  refC[m * N + n] = sum;
                }
              }
              const gpuC = await harness.runMatmulQ4K(gpu.device, A, B_q4k, M, N, K);
              let maxError = 0;
              let hasNaN = false;
              let zeroCount = 0;
              for (let i = 0; i < refC.length; i++) {
                if (isNaN(gpuC[i])) hasNaN = true;
                if (gpuC[i] === 0 && refC[i] !== 0) zeroCount++;
                maxError = Math.max(maxError, Math.abs(gpuC[i] - refC[i]));
              }
              const passed = maxError < 0.1 && !hasNaN && zeroCount < refC.length / 2;
              return { passed, maxError, hasNaN, zeroCount, M, N, K };
            }

            case 'matmul-q4k-large': {
              const M = 16, K = 1152, N = 1024;
              const A = new Float32Array(M * K).map(() => (Math.random() * 2 - 1) * 0.3);
              const B_f32 = new Float32Array(N * K).map(() => (Math.random() * 2 - 1) * 0.3);
              const blocksPerRow = Math.ceil(K / 256);
              const paddedK = blocksPerRow * 256;
              const numBlocks = N * blocksPerRow;
              const B_padded = new Float32Array(N * paddedK);
              for (let n = 0; n < N; n++) {
                for (let k = 0; k < K; k++) {
                  B_padded[n * paddedK + k] = B_f32[n * K + k];
                }
              }
              const B_q4k = refs.quantizeQ4_KRef(B_padded, numBlocks);
              const B_dequant = refs.dequantQ4_KRef(B_q4k, numBlocks);
              const dequantRowStride = blocksPerRow * 256;
              const refC = new Float32Array(M * N);
              for (let m = 0; m < M; m++) {
                for (let n = 0; n < N; n++) {
                  let sum = 0;
                  for (let k = 0; k < K; k++) {
                    sum += A[m * K + k] * B_dequant[n * dequantRowStride + k];
                  }
                  refC[m * N + n] = sum;
                }
              }
              const gpuC = await harness.runMatmulQ4K(gpu.device, A, B_q4k, M, N, K);
              let maxError = 0;
              let hasNaN = false;
              let zeroCount = 0;
              let allZero = true;
              for (let i = 0; i < refC.length; i++) {
                if (isNaN(gpuC[i])) hasNaN = true;
                if (gpuC[i] !== 0) allZero = false;
                if (gpuC[i] === 0 && Math.abs(refC[i]) > 0.001) zeroCount++;
                maxError = Math.max(maxError, Math.abs(gpuC[i] - refC[i]));
              }
              const passed = maxError < 0.5 && !hasNaN && !allZero && zeroCount < refC.length / 4;
              return { passed, maxError, hasNaN, zeroCount, allZero, M, N, K, numBlocks };
            }

            case 'rmsnorm': {
              const batchSize = 4, hiddenSize = 256;
              const input = new Float32Array(batchSize * hiddenSize).map(() => Math.random() * 2 - 1);
              const weight = new Float32Array(hiddenSize).map(() => Math.random() * 0.5 + 0.5);
              const ref = refs.rmsNormRef(input, weight, batchSize, hiddenSize);
              const gpuResult = await harness.runRMSNorm(gpu.device, input, weight, batchSize, hiddenSize);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-4, maxError };
            }

            case 'softmax': {
              const innerSize = 128, outerSize = 32;
              const input = new Float32Array(innerSize * outerSize).map(() => Math.random() * 4 - 2);
              const ref = refs.softmaxRef(input, innerSize, outerSize);
              const gpuResult = await harness.runSoftmax(gpu.device, input, innerSize, outerSize);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-5, maxError };
            }

            case 'gather': {
              const vocabSize = 1000, embedDim = 128, numTokens = 16;
              const embeddings = new Float32Array(vocabSize * embedDim).map(() => Math.random() * 2 - 1);
              const indices = new Uint32Array(numTokens).map(() => Math.floor(Math.random() * vocabSize));
              const ref = refs.gatherRef(embeddings, indices, vocabSize, embedDim);
              const gpuResult = await harness.runGather(gpu.device, embeddings, indices, vocabSize, embedDim);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-6, maxError };
            }

            case 'silu': {
              const size = 1024;
              const input = new Float32Array(size).map(() => Math.random() * 4 - 2);
              const ref = refs.siluRef(input);
              const gpuResult = await harness.runSiLU(gpu.device, input);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-5, maxError };
            }

            case 'rope': {
              const seqLen = 16, numHeads = 8, headDim = 64;
              const input = new Float32Array(seqLen * numHeads * headDim).map(() => Math.random() * 2 - 1);
              const { cos, sin } = refs.computeRopeFreqs(headDim, seqLen);
              const ref = refs.ropeRef(input, cos, sin, seqLen, numHeads, headDim, 0);
              const gpuResult = await harness.runRoPE(gpu.device, input, seqLen, numHeads, headDim);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-4, maxError };
            }

            case 'residual': {
              const size = 1024;
              const x = new Float32Array(size).map(() => Math.random() * 2 - 1);
              const residual = new Float32Array(size).map(() => Math.random() * 2 - 1);
              const ref = refs.residualAddRef(x, residual);
              const gpuResult = await harness.runResidual(gpu.device, x, residual);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-6, maxError };
            }

            case 'attention': {
              const seqLen = 16, numHeads = 4, headDim = 32;
              const Q = new Float32Array(seqLen * numHeads * headDim).map(() => Math.random() * 0.5);
              const K = new Float32Array(seqLen * numHeads * headDim).map(() => Math.random() * 0.5);
              const V = new Float32Array(seqLen * numHeads * headDim).map(() => Math.random() * 0.5);
              const mask = refs.createCausalMask(seqLen);
              const ref = refs.attentionRef(Q, K, V, seqLen, seqLen, numHeads, numHeads, headDim, mask);
              const gpuResult = await harness.runAttention(gpu.device, Q, K, V, seqLen, seqLen, numHeads, numHeads, headDim, mask);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-3, maxError };
            }

            case 'scatter-add': {
              const numTokens = 8, hiddenSize = 64, numExperts = 4, topK = 2;
              const expertOutputs = new Float32Array(numExperts * numTokens * hiddenSize).map(() => Math.random());
              const indices = new Uint32Array(numTokens * topK);
              const weights = new Float32Array(numTokens * topK);
              for (let t = 0; t < numTokens; t++) {
                for (let k = 0; k < topK; k++) {
                  indices[t * topK + k] = Math.floor(Math.random() * numExperts);
                  weights[t * topK + k] = 1.0 / topK;
                }
              }
              const ref = refs.scatterAddRef(expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK);
              const gpuResult = await harness.runScatterAdd(gpu.device, expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK);
              let maxError = 0;
              for (let i = 0; i < ref.length; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - ref[i]));
              }
              return { passed: maxError < 1e-4, maxError };
            }

            case 'moe-gather': {
              const numTokens = 8, hiddenSize = 64, numExperts = 4, topK = 2;
              const tokens = new Float32Array(numTokens * hiddenSize).map(() => Math.random());
              const expertIndices = new Uint32Array(numTokens * topK);
              for (let t = 0; t < numTokens; t++) {
                for (let k = 0; k < topK; k++) {
                  expertIndices[t * topK + k] = Math.floor(Math.random() * numExperts);
                }
              }
              const ref = refs.moeGatherRef(tokens, expertIndices, numTokens, hiddenSize, numExperts, topK);
              const gpuResult = await harness.runMoEGather(gpu.device, tokens, expertIndices, numTokens, hiddenSize, numExperts, topK);
              let passed = true;
              for (let i = 0; i < numExperts; i++) {
                if (ref.tokenCounts[i] !== gpuResult.tokenCounts[i]) {
                  passed = false;
                  break;
                }
              }
              return { passed, tokenCounts: Array.from(gpuResult.tokenCounts) };
            }

            case 'topk': {
              const numTokens = 4, numExperts = 8, topK = 2;
              const logits = new Float32Array(numTokens * numExperts).map(() => Math.random() * 4 - 2);
              const ref = refs.softmaxTopkRef(logits, numTokens, numExperts, topK, true);
              const gpuResult = await harness.runSoftmaxTopK(gpu.device, logits, numTokens, numExperts, topK);
              let passed = true;
              for (let t = 0; t < numTokens; t++) {
                                const refSet = new Set();
                                const gpuSet = new Set();
                for (let k = 0; k < topK; k++) {
                  refSet.add(ref.indices[t * topK + k]);
                  gpuSet.add(gpuResult.indices[t * topK + k]);
                }
                for (const idx of refSet) {
                  if (!gpuSet.has(idx)) {
                    passed = false;
                    break;
                  }
                }
              }
              return { passed };
            }

            case 'dequant': {
              const numBlocks = 4;
              const blockSize = 32;
              const quantized = new Uint8Array(numBlocks * 18);
              for (let i = 0; i < quantized.length; i++) {
                quantized[i] = Math.floor(Math.random() * 256);
              }
              const ref = refs.dequantQ4_0Ref(quantized, numBlocks);
              return { passed: ref.length === numBlocks * blockSize, refLength: ref.length };
            }

            case 'dequant-q4k': {
              const numBlocks = 4;
              const blockElems = 256;
              const values = new Float32Array(numBlocks * blockElems);
              for (let i = 0; i < values.length; i++) {
                values[i] = Math.sin(i * 0.1) * 0.75 + Math.cos(i * 0.03) * 0.25;
              }
              const quantized = refs.quantizeQ4_KRef(values, numBlocks);
              const expected = refs.dequantQ4_KRef(quantized, numBlocks);
              const gpuResult = await harness.runDequantQ4K(gpu.device, quantized, numBlocks);
              let maxError = 0;
              let maxErrorIdx = -1;
              for (let i = 0; i < expected.length; i++) {
                const err = Math.abs(gpuResult[i] - expected[i]);
                if (err > maxError) {
                  maxError = err;
                  maxErrorIdx = i;
                }
              }
              const passed = maxError < 1e-3;
              return {
                passed,
                maxError,
                maxErrorIdx,
                length: gpuResult.length,
                first10Actual: Array.from(gpuResult.slice(0, 10)),
                first10Expected: Array.from(expected.slice(0, 10)),
              };
            }

            case 'dequant-q4k-f16': {
              const numBlocks = 2048 * 9;
              const blockElems = 256;
              const values = new Float32Array(numBlocks * blockElems);
              for (let i = 0; i < values.length; i++) {
                values[i] = Math.sin(i * 0.1) * 0.75 + Math.cos(i * 0.03) * 0.25;
              }
              const quantized = refs.quantizeQ4_KRef(values, numBlocks);
              const expected = refs.dequantQ4_KRef(quantized, numBlocks);
              const gpuResult = await harness.runDequantQ4K_F16(gpu.device, quantized, numBlocks);
              let maxError = 0;
              let maxErrorIdx = -1;
              const sampleErrors = [];
              for (let i = 0; i < expected.length; i++) {
                const err = Math.abs(gpuResult[i] - expected[i]);
                if (err > maxError) {
                  maxError = err;
                  maxErrorIdx = i;
                }
                if (err > 0.01 && sampleErrors.length < 10) {
                  sampleErrors.push({i, expected: expected[i], actual: gpuResult[i], err});
                }
              }
              const passed = maxError < 0.01;
              return {
                passed,
                maxError,
                maxErrorIdx,
                length: gpuResult.length,
                first10Actual: Array.from(gpuResult.slice(0, 10)),
                first10Expected: Array.from(expected.slice(0, 10)),
                sampleErrors,
              };
            }

            case 'matmul-f16w': {
              const M = 10;
              const K = 2304;
              const N = 2048;
              const A = new Float32Array(M * K).map((_, i) =>
                Math.sin(i * 0.1) * 50 + Math.cos(i * 0.03) * 80
              );
              const B_f32 = new Float32Array(N * K).map((_, i) =>
                Math.sin(i * 0.07) * 0.02 + Math.cos(i * 0.11) * 0.01
              );
              const blocksPerRow = Math.ceil(K / 256);
              const numBlocks = N * blocksPerRow;
              const B_q4k = refs.quantizeQ4_KRef(B_f32, numBlocks);
              const B_cpu_dequant = refs.dequantQ4_KRef(B_q4k, numBlocks);
              const refC = new Float32Array(M * N);
              for (let m = 0; m < M; m++) {
                for (let n = 0; n < N; n++) {
                  let sum = 0;
                  for (let k = 0; k < K; k++) {
                    sum += A[m * K + k] * B_cpu_dequant[n * K + k];
                  }
                  refC[m * N + n] = sum;
                }
              }
              const gpuC = await harness.runDequantAndMatmulF16W(gpu.device, A, B_q4k, M, N, K, numBlocks);
              let maxError = 0;
              let maxErrorIdx = -1;
              let hasNaN = false;
              const sampleErrors = [];
              for (let i = 0; i < refC.length; i++) {
                if (isNaN(gpuC[i])) hasNaN = true;
                const err = Math.abs(gpuC[i] - refC[i]);
                if (err > maxError) {
                  maxError = err;
                  maxErrorIdx = i;
                }
                if (err > 0.1 && sampleErrors.length < 10) {
                  sampleErrors.push({i, expected: refC[i], actual: gpuC[i], err});
                }
              }
              const passed = maxError < 0.5 && !hasNaN;
              return {
                passed,
                maxError,
                maxErrorIdx,
                hasNaN,
                length: gpuC.length,
                first10Actual: Array.from(gpuC.slice(0, 10)),
                first10Ref: Array.from(refC.slice(0, 10)),
                sampleErrors,
                M, N, K, numBlocks,
              };
            }

            case 'swiglu': {
              const size = 256;
              const gate = new Float32Array(size).map(() => Math.random() * 2 - 1);
              const up = new Float32Array(size).map(() => Math.random() * 2 - 1);
              const gateBias = new Float32Array(size).map(() => Math.random() * 0.1 - 0.05);
              const upBias = new Float32Array(size).map(() => Math.random() * 0.1 - 0.05);
              const expected = new Float32Array(size);
              for (let i = 0; i < size; i++) {
                const gatedValue = gate[i] + gateBias[i];
                const silu = gatedValue / (1 + Math.exp(-gatedValue));
                expected[i] = silu * (up[i] + upBias[i]);
              }
              const gpuResult = await harness.runSwiGLU(gpu.device, gate, up, gateBias, upBias);
              let maxError = 0;
              for (let i = 0; i < size; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - expected[i]));
              }
              return { passed: maxError < 1e-4, maxError };
            }

            case 'scale': {
              const size = 512;
              const input = new Float32Array(size).map(() => Math.random() * 10 - 5);
              const scale = 0.125;
              const expected = new Float32Array(size);
              for (let i = 0; i < size; i++) {
                expected[i] = input[i] * scale;
              }
              const gpuResult = await harness.runScale(gpu.device, input, scale);
              let maxError = 0;
              for (let i = 0; i < size; i++) {
                maxError = Math.max(maxError, Math.abs(gpuResult[i] - expected[i]));
              }
              return { passed: maxError < 1e-6, maxError };
            }

            case 'dequant-q6k': {
              const numBlocks = 2;
              const blockSize = 256;
              const Q6K_BLOCK_BYTES = 210;
              const D_OFFSET = 208;
              const quantized = new Uint8Array(numBlocks * Q6K_BLOCK_BYTES);
              for (let i = 0; i < quantized.length; i++) {
                quantized[i] = Math.floor(Math.random() * 256);
              }
              for (let b = 0; b < numBlocks; b++) {
                const base = b * Q6K_BLOCK_BYTES + D_OFFSET;
                quantized[base] = 0x00;
                quantized[base + 1] = 0x3C;
              }
              const gpuResult = await harness.runDequantQ6K(gpu.device, quantized, numBlocks);
              let nanCount = 0;
              let infCount = 0;
              for (let i = 0; i < gpuResult.length; i++) {
                if (isNaN(gpuResult[i])) nanCount++;
                if (!isFinite(gpuResult[i])) infCount++;
              }
              const passed = gpuResult.length === numBlocks * blockSize && nanCount === 0 && infCount === 0;
              return { passed, outputLength: gpuResult.length, expectedLength: numBlocks * blockSize, nanCount, infCount };
            }

            case 'sample': {
              const vocabSize = 128;
              const logits = new Float32Array(vocabSize).map(() => Math.random() * 10 - 5);
              const expectedIdx = 42;
              logits[expectedIdx] = 100;
              const gpuIdx = await harness.runArgmax(gpu.device, logits);
              const refIdx = refs.argmaxRef(logits);
              const passed = gpuIdx === refIdx && gpuIdx === expectedIdx;
              return { passed, gpuIdx, refIdx, expectedIdx };
            }

            default:
              return { passed: false, error: `Unknown test: ${name}` };
          }
        },
        testName
      );

      const duration = Date.now() - testStart;
      const passed = result.passed === true;

      results.push({
        name: testName,
        passed,
        duration,
        error: passed ? undefined : JSON.stringify(result),
      });

      const status = passed ? '\x1b[32mPASS\x1b[0m' : '\x1b[31mFAIL\x1b[0m';
      console.log(`  ${status} ${testName} (${duration}ms)`);
      if (!passed && opts.verbose) {
        console.log(`    Details: ${JSON.stringify(result)}`);
      }
    } catch (err) {
      const duration = Date.now() - testStart;
      results.push({
        name: testName,
        passed: false,
        duration,
        error:  (err).message,
      });
      console.log(`  \x1b[31mFAIL\x1b[0m ${testName} (${duration}ms)`);
      console.log(`    Error: ${ (err).message}`);
    }
  }

  const totalDuration = Date.now() - startTime;
  const passed = results.filter((r) => r.passed).length;
  const failed = results.filter((r) => !r.passed).length;

  return {
    suite: 'correctness',
    passed,
    failed,
    skipped: tests.length - testsToRun.length,
    duration: totalDuration,
    results,
  };
}
