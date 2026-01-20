


import { setHarnessConfig, appendRuntimeConfigParams } from '../args/index.js';
import { KERNEL_BENCHMARKS } from '../suites.js';

export async function runKernelBenchmarks(page, opts) {
  console.log('\n' + '='.repeat(60));
  console.log('KERNEL BENCHMARKS');
  console.log('='.repeat(60));

  const runtimeConfig = opts.runtimeConfig;
  if (!runtimeConfig) {
    throw new Error('Runtime config is required for benchmarks.');
  }
  const benchmarkRun = runtimeConfig.shared.benchmark.run;

  await page.addInitScript(() => {
        (window).__name = (target) => target;
  });

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

  await page.waitForFunction(
    () =>  (window).testHarness &&  (window).testHarness.references,
    { timeout: 10000 }
  );

    const results = [];
  const startTime = Date.now();

  const benchmarks = opts.filter
    ? KERNEL_BENCHMARKS.filter((b) => b.includes(opts.filter))
    : KERNEL_BENCHMARKS;

  for (const benchName of benchmarks) {
    console.log(`\n  Benchmarking: ${benchName}...`);

    try {
      const result = await page.evaluate(
        async (config) => {
          const __name = (target) => target;
          const { name, warmup, runs } = config;
          const harness =  (window).testHarness;
          const gpu = await harness.getGPU();
          const cache = {};

                    const benchmarks = {
            matmul: async () => {
              if (!cache.matmul) {
                const M = 1, N = 4096, K = 4096;
                const A = new Float32Array(M * K).fill(1);
                const B = new Float32Array(K * N).fill(1);
                cache.matmul = { A, B, M, N, K };
              }
              const { A, B, M, N, K } = cache.matmul;
              await harness.runMatmul(gpu.device, A, B, M, N, K);
            },
            'matmul-q4k': async () => {
              if (!cache.matmulQ4K) {
                const M = 1, N = 1024, K = 1024;
                const A = new Float32Array(M * K).fill(1);
                const B = new Float32Array(N * K).fill(1);
                const numBlocks = N * (K / 256);
                const B_q4k = harness.references.quantizeQ4_KRef(B, numBlocks);
                cache.matmulQ4K = { A, B_q4k, M, N, K, numBlocks };
              }
              const { A, B_q4k, M, N, K } = cache.matmulQ4K;
              await harness.runMatmulQ4K(gpu.device, A, B_q4k, M, N, K);
            },
            'matmul-q4k-large': async () => {
              if (!cache.matmulQ4KLarge) {
                const M = 8, N = 2048, K = 1024;
                const A = new Float32Array(M * K).fill(1);
                const B = new Float32Array(N * K).fill(1);
                const numBlocks = N * (K / 256);
                const B_q4k = harness.references.quantizeQ4_KRef(B, numBlocks);
                cache.matmulQ4KLarge = { A, B_q4k, M, N, K, numBlocks };
              }
              const { A, B_q4k, M, N, K } = cache.matmulQ4KLarge;
              await harness.runMatmulQ4K(gpu.device, A, B_q4k, M, N, K);
            },
            'matmul-f16w': async () => {
              if (!cache.matmulF16W) {
                const M = 1, N = 1024, K = 1024;
                const A = new Float32Array(M * K).fill(1);
                const B_f32 = new Float32Array(N * K).fill(1);
                const B_f16 = await harness.runF32ToF16(gpu.device, B_f32);
                cache.matmulF16W = { A, B_f16, M, N, K };
              }
              const { A, B_f16, M, N, K } = cache.matmulF16W;
              await harness.runMatmulF16W(gpu.device, A, B_f16, M, N, K);
            },
            rmsnorm: async () => {
              if (!cache.rmsnorm) {
                const batchSize = 1, hiddenSize = 4096;
                const input = new Float32Array(batchSize * hiddenSize).fill(1);
                const weight = new Float32Array(hiddenSize).fill(1);
                cache.rmsnorm = { input, weight, batchSize, hiddenSize };
              }
              const { input, weight, batchSize, hiddenSize } = cache.rmsnorm;
              await harness.runRMSNorm(gpu.device, input, weight, batchSize, hiddenSize);
            },
            softmax: async () => {
              if (!cache.softmax) {
                const innerSize = 32000, outerSize = 1;
                const input = new Float32Array(innerSize * outerSize).fill(1);
                cache.softmax = { input, innerSize, outerSize };
              }
              const { input, innerSize, outerSize } = cache.softmax;
              await harness.runSoftmax(gpu.device, input, innerSize, outerSize);
            },
            silu: async () => {
              if (!cache.silu) {
                const size = 14336;
                const input = new Float32Array(size).fill(1);
                cache.silu = { input };
              }
              await harness.runSiLU(gpu.device, cache.silu.input);
            },
            rope: async () => {
              if (!cache.rope) {
                const seqLen = 1, numHeads = 32, headDim = 128;
                const input = new Float32Array(seqLen * numHeads * headDim).fill(1);
                cache.rope = { input, seqLen, numHeads, headDim };
              }
              const { input, seqLen, numHeads, headDim } = cache.rope;
              await harness.runRoPE(gpu.device, input, seqLen, numHeads, headDim);
            },
            swiglu: async () => {
              if (!cache.swiglu) {
                const size = 14336;
                const gate = new Float32Array(size).fill(0.5);
                const up = new Float32Array(size).fill(0.5);
                const gateBias = new Float32Array(size).fill(0.01);
                const upBias = new Float32Array(size).fill(0.01);
                cache.swiglu = { gate, up, gateBias, upBias };
              }
              const { gate, up, gateBias, upBias } = cache.swiglu;
              await harness.runSwiGLU(gpu.device, gate, up, gateBias, upBias);
            },
            gather: async () => {
              if (!cache.gather) {
                const vocabSize = 8192, embedDim = 512, numTokens = 16;
                const embeddings = new Float32Array(vocabSize * embedDim).fill(0.01);
                const indices = new Uint32Array(numTokens);
                for (let i = 0; i < numTokens; i++) {
                  indices[i] = (i * 127) % vocabSize;
                }
                cache.gather = { embeddings, indices, vocabSize, embedDim };
              }
              const { embeddings, indices, vocabSize, embedDim } = cache.gather;
              await harness.runGather(gpu.device, embeddings, indices, vocabSize, embedDim);
            },
            'scatter-add': async () => {
              if (!cache.scatterAdd) {
                const numTokens = 4, hiddenSize = 1024, numExperts = 8, topK = 2;
                const expertOutputs = new Float32Array(numExperts * numTokens * hiddenSize).fill(0.01);
                const indices = new Uint32Array(numTokens * topK);
                const weights = new Float32Array(numTokens * topK);
                for (let i = 0; i < numTokens * topK; i++) {
                  indices[i] = i % numExperts;
                  weights[i] = 1 / topK;
                }
                cache.scatterAdd = {
                  expertOutputs,
                  indices,
                  weights,
                  numTokens,
                  hiddenSize,
                  numExperts,
                  topK,
                };
              }
              const {
                expertOutputs,
                indices,
                weights,
                numTokens,
                hiddenSize,
                numExperts,
                topK,
              } = cache.scatterAdd;
              await harness.runScatterAdd(
                gpu.device,
                expertOutputs,
                indices,
                weights,
                numTokens,
                hiddenSize,
                numExperts,
                topK
              );
            },
            'moe-gather': async () => {
              if (!cache.moeGather) {
                const numTokens = 4, hiddenSize = 1024, numExperts = 8, topK = 2;
                const tokens = new Float32Array(numTokens * hiddenSize).fill(0.01);
                const expertIndices = new Uint32Array(numTokens * topK);
                for (let i = 0; i < numTokens * topK; i++) {
                  expertIndices[i] = i % numExperts;
                }
                cache.moeGather = {
                  tokens,
                  expertIndices,
                  numTokens,
                  hiddenSize,
                  numExperts,
                  topK,
                };
              }
              const {
                tokens,
                expertIndices,
                numTokens,
                hiddenSize,
                numExperts,
                topK,
              } = cache.moeGather;
              await harness.runMoEGather(
                gpu.device,
                tokens,
                expertIndices,
                numTokens,
                hiddenSize,
                numExperts,
                topK
              );
            },
            attention: async () => {
              if (!cache.attention) {
                const seqLen = 1, kvLen = 128, numHeads = 32, headDim = 128;
                const Q = new Float32Array(seqLen * numHeads * headDim).fill(0.1);
                const K = new Float32Array(kvLen * numHeads * headDim).fill(0.1);
                const V = new Float32Array(kvLen * numHeads * headDim).fill(0.1);
                cache.attention = { Q, K, V, seqLen, kvLen, numHeads, headDim };
              }
              const { Q, K, V, seqLen, kvLen, numHeads, headDim } = cache.attention;
              await harness.runAttention(gpu.device, Q, K, V, seqLen, kvLen, numHeads, numHeads, headDim);
            },
            moe: async () => {
              if (!cache.moe) {
                const numTokens = 1, numExperts = 8, topK = 2;
                const logits = new Float32Array(numTokens * numExperts).fill(1);
                cache.moe = { logits, numTokens, numExperts, topK };
              }
              const { logits, numTokens, numExperts, topK } = cache.moe;
              await harness.runSoftmaxTopK(gpu.device, logits, numTokens, numExperts, topK);
            },
            residual: async () => {
              if (!cache.residual) {
                const size = 4096;
                const x = new Float32Array(size).fill(0.25);
                const residual = new Float32Array(size).fill(0.25);
                cache.residual = { x, residual };
              }
              const { x, residual } = cache.residual;
              await harness.runResidual(gpu.device, x, residual);
            },
            scale: async () => {
              if (!cache.scale) {
                const size = 4096;
                const input = new Float32Array(size).fill(0.25);
                cache.scale = { input };
              }
              await harness.runScale(gpu.device, cache.scale.input, 0.5);
            },
            topk: async () => {
              if (!cache.topk) {
                const numTokens = 1, numExperts = 4096, topK = 8;
                const probs = new Float32Array(numTokens * numExperts);
                for (let i = 0; i < probs.length; i++) {
                  probs[i] = (i % 37) * 0.01;
                }
                cache.topk = { probs, numTokens, numExperts, topK };
              }
              const { probs, numTokens, numExperts, topK } = cache.topk;
              await harness.runTopK(gpu.device, probs, numTokens, numExperts, topK);
            },
            'dequant-q4k': async () => {
              if (!cache.dequantQ4K) {
                const N = 1024, K = 1024;
                const weights = new Float32Array(N * K).fill(1);
                const numBlocks = N * (K / 256);
                const quantized = harness.references.quantizeQ4_KRef(weights, numBlocks);
                cache.dequantQ4K = { quantized, numBlocks };
              }
              const { quantized, numBlocks } = cache.dequantQ4K;
              await harness.runDequantQ4K(gpu.device, quantized, numBlocks);
            },
            'dequant-q4k-f16': async () => {
              if (!cache.dequantQ4KF16) {
                const N = 1024, K = 1024;
                const weights = new Float32Array(N * K).fill(1);
                const numBlocks = N * (K / 256);
                const quantized = harness.references.quantizeQ4_KRef(weights, numBlocks);
                cache.dequantQ4KF16 = { quantized, numBlocks };
              }
              const { quantized, numBlocks } = cache.dequantQ4KF16;
              await harness.runDequantQ4K_F16(gpu.device, quantized, numBlocks);
            },
            'dequant-q6k': async () => {
              if (!cache.dequantQ6K) {
                const numBlocks = 64;
                const quantized = new Uint8Array(numBlocks * 210);
                for (let i = 0; i < quantized.length; i++) {
                  quantized[i] = i % 255;
                }
                cache.dequantQ6K = { quantized, numBlocks };
              }
              const { quantized, numBlocks } = cache.dequantQ6K;
              await harness.runDequantQ6K(gpu.device, quantized, numBlocks);
            },
            sample: async () => {
              if (!cache.sample) {
                const vocabSize = 4096, topK = 40, temperature = 1.0;
                const logits = new Float32Array(vocabSize);
                for (let i = 0; i < logits.length; i++) {
                  logits[i] = (i % 29) * 0.01;
                }
                cache.sample = { logits, vocabSize, topK, temperature };
              }
              const { logits, topK, temperature } = cache.sample;
              await harness.runSampleTopK(gpu.device, logits, temperature, topK, 0.42);
            },
          };

          const fn = benchmarks[name];
          if (!fn) return { error: `Unknown benchmark: ${name}` };

          for (let i = 0; i < warmup; i++) {
            await fn();
            await gpu.device.queue.onSubmittedWorkDone();
          }

                    const times = [];
          for (let i = 0; i < runs; i++) {
            const start = performance.now();
            await fn();
            await gpu.device.queue.onSubmittedWorkDone();
            times.push(performance.now() - start);
          }

          const sorted = [...times].sort((a, b) => a - b);
          const median = sorted[Math.floor(sorted.length / 2)];
          const mean = times.reduce((a, b) => a + b, 0) / times.length;
          const min = sorted[0];
          const max = sorted[sorted.length - 1];

          return { median, mean, min, max, samples: times.length };
        },
        { name: benchName, warmup: benchmarkRun.warmupRuns, runs: benchmarkRun.timedRuns }
      );

      if ('error' in result) {
        results.push({
          name: benchName,
          passed: false,
          duration: 0,
          error: result.error,
        });
        console.log(`  \x1b[31mFAIL\x1b[0m ${benchName}: ${result.error}`);
      } else {
        results.push({
          name: benchName,
          passed: true,
          duration: result.median,
        });
        console.log(
          `  ${benchName}: median=${result.median.toFixed(3)}ms ` +
          `mean=${result.mean.toFixed(3)}ms ` +
          `min=${result.min.toFixed(3)}ms max=${result.max.toFixed(3)}ms`
        );
      }
    } catch (err) {
      results.push({
        name: benchName,
        passed: false,
        duration: 0,
        error:  (err).message,
      });
      console.log(`  \x1b[31mFAIL\x1b[0m ${benchName}: ${ (err).message}`);
    }
  }

  const totalDuration = Date.now() - startTime;

  return {
    suite: 'bench:kernels',
    passed: results.filter((r) => r.passed).length,
    failed: results.filter((r) => !r.passed).length,
    skipped: 0,
    duration: totalDuration,
    results,
  };
}
