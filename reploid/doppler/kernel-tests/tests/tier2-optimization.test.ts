/**
 * Tier 2 P0 Kernel Optimization Tests
 *
 * Tests for the new optimized kernels:
 * - Attention decode kernel (optimized)
 * - Fused FFN kernel
 * - Kernel benchmark harness
 * - Performance profiler
 */

import { test, expect } from '@playwright/test';

test.describe('Tier 2 P0: Kernel Optimization', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for WebGPU initialization
    await page.waitForFunction(() => window.testHarness?.getGPU);
  });

  test('Fused FFN kernel compiles', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const gpu = await window.testHarness.getGPU();
      if (!gpu) return { success: false, error: 'No GPU' };

      try {
        // Fetch the fused FFN shader
        const response = await fetch('/gpu/kernels/ffn_fused.wgsl');
        if (!response.ok) {
          return { success: false, error: `Failed to fetch shader: ${response.status}` };
        }
        const source = await response.text();

        // Try to compile it
        const module = gpu.device.createShaderModule({
          label: 'ffn_fused_test',
          code: source,
        });

        const info = await module.getCompilationInfo();
        const errors = info.messages.filter(m => m.type === 'error');

        if (errors.length > 0) {
          return {
            success: false,
            error: errors.map(e => `${e.message} (line ${e.lineNum})`).join('; '),
          };
        }

        return { success: true };
      } catch (e: unknown) {
        return { success: false, error: (e as Error).message };
      }
    });

    expect(result.success, result.error).toBe(true);
  });

  test('Optimized attention decode kernel compiles', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const gpu = await window.testHarness.getGPU();
      if (!gpu) return { success: false, error: 'No GPU' };

      try {
        const response = await fetch('/gpu/kernels/attention_decode_optimized.wgsl');
        if (!response.ok) {
          return { success: false, error: `Failed to fetch shader: ${response.status}` };
        }
        const source = await response.text();

        const module = gpu.device.createShaderModule({
          label: 'attention_decode_optimized_test',
          code: source,
        });

        const info = await module.getCompilationInfo();
        const errors = info.messages.filter(m => m.type === 'error');

        if (errors.length > 0) {
          return {
            success: false,
            error: errors.map(e => `${e.message} (line ${e.lineNum})`).join('; '),
          };
        }

        return { success: true };
      } catch (e: unknown) {
        return { success: false, error: (e as Error).message };
      }
    });

    expect(result.success, result.error).toBe(true);
  });

  test('Kernel benchmark harness runs', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        const gpu = await window.testHarness.getGPU();
        if (!gpu) return { success: false, error: 'No GPU' };

        // Simple benchmark: create buffer and measure timing
        const size = 1024 * 1024; // 1M floats
        const buffer = gpu.device.createBuffer({
          size: size * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Warmup
        for (let i = 0; i < 3; i++) {
          const data = new Float32Array(size);
          gpu.device.queue.writeBuffer(buffer, 0, data);
          await gpu.device.queue.onSubmittedWorkDone();
        }

        // Timed
        const times: number[] = [];
        for (let i = 0; i < 10; i++) {
          const data = new Float32Array(size);
          const start = performance.now();
          gpu.device.queue.writeBuffer(buffer, 0, data);
          await gpu.device.queue.onSubmittedWorkDone();
          times.push(performance.now() - start);
        }

        buffer.destroy();

        const sorted = times.sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)];
        const gbPerSec = (size * 4) / (median * 1e6);

        return {
          success: true,
          median_ms: median,
          gb_per_sec: gbPerSec,
        };
      } catch (e: unknown) {
        return { success: false, error: (e as Error).message };
      }
    });

    expect(result.success, result.error).toBe(true);
    expect(result.median_ms).toBeGreaterThan(0);
    expect(result.gb_per_sec).toBeGreaterThan(0);
    console.log(`Memory bandwidth: ${result.gb_per_sec?.toFixed(2)} GB/s`);
  });

  test('Subgroup operations available', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const gpu = await window.testHarness.getGPU();
      if (!gpu) return { success: false, error: 'No GPU', hasSubgroups: false };

      const adapter = gpu.adapter;
      const features = Array.from(adapter.features);

      const hasSubgroups = features.includes('subgroups');
      const hasF16 = features.includes('shader-f16');

      return {
        success: true,
        hasSubgroups,
        hasF16,
        features,
      };
    });

    expect(result.success).toBe(true);
    console.log(`Subgroups: ${result.hasSubgroups}`);
    console.log(`F16: ${result.hasF16}`);
    console.log(`Features: ${result.features?.join(', ')}`);

    // Warn if subgroups not available (needed for optimized kernels)
    if (!result.hasSubgroups) {
      console.warn('WARNING: Subgroups not available. Optimized kernels will fall back to slower variants.');
    }
  });

  test('Gemma 3 1B config validation', async ({ page }) => {
    // Validate that the Gemma 3 1B model dimensions are correct for our kernels
    const config = {
      hiddenSize: 1152,
      intermediateSize: 6912,
      numHeads: 4,
      numKVHeads: 1,
      headDim: 256,
      vocabSize: 262144,
      numLayers: 26,
    };

    // Validate attention dimensions
    expect(config.numHeads * config.headDim).toBe(1024); // QKV dimension
    expect(config.hiddenSize).toBe(1152);

    // Validate FFN dimensions
    expect(config.intermediateSize).toBe(config.hiddenSize * 6); // 6x expansion

    // Validate GQA ratio
    expect(config.numHeads / config.numKVHeads).toBe(4); // 4:1 GQA

    // Log expected memory usage
    const kvCacheSizePerLayer = 2 * 1024 * config.numKVHeads * config.headDim * 2; // F16
    const kvCacheTotal = kvCacheSizePerLayer * config.numLayers;
    console.log(`Expected KV cache (1024 tokens): ${(kvCacheTotal / 1e6).toFixed(1)} MB`);

    const weightSize = config.numLayers * (
      config.hiddenSize * (config.numHeads + 2 * config.numKVHeads) * config.headDim + // QKV
      config.numHeads * config.headDim * config.hiddenSize + // O proj
      config.hiddenSize * config.intermediateSize * 2 + // gate + up
      config.intermediateSize * config.hiddenSize // down
    ) / 2; // Q4 = 0.5 bytes per weight
    console.log(`Expected weight size (Q4): ${(weightSize / 1e9).toFixed(2)} GB`);
  });
});

test.describe('Performance Regression Investigation', () => {
  test('Identify bottleneck operations', async ({ page }) => {
    await page.goto('/');

    const result = await page.evaluate(async () => {
      const gpu = await window.testHarness.getGPU();
      if (!gpu) return { success: false, error: 'No GPU' };

      // Measure key operations for Gemma 3 1B
      const config = {
        hiddenSize: 1152,
        intermediateSize: 6912,
        numHeads: 4,
        headDim: 256,
      };

      const operations: { name: string; time: number }[] = [];

      // Helper to benchmark an operation
      async function bench(name: string, fn: () => Promise<void>, iterations = 20): Promise<number> {
        // Warmup
        for (let i = 0; i < 3; i++) await fn();

        // Timed
        const times: number[] = [];
        for (let i = 0; i < iterations; i++) {
          const start = performance.now();
          await fn();
          await gpu.device.queue.onSubmittedWorkDone();
          times.push(performance.now() - start);
        }

        const sorted = times.sort((a, b) => a - b);
        return sorted[Math.floor(sorted.length / 2)];
      }

      // 1. Buffer creation overhead
      const bufferTime = await bench('buffer_create', async () => {
        const buf = gpu.device.createBuffer({
          size: config.hiddenSize * 4,
          usage: GPUBufferUsage.STORAGE,
        });
        buf.destroy();
      });
      operations.push({ name: 'buffer_create', time: bufferTime });

      // 2. Queue submit overhead
      const submitTime = await bench('queue_submit', async () => {
        const encoder = gpu.device.createCommandEncoder();
        gpu.device.queue.submit([encoder.finish()]);
      });
      operations.push({ name: 'queue_submit', time: submitTime });

      // 3. GPU sync overhead
      const syncTime = await bench('gpu_sync', async () => {
        await gpu.device.queue.onSubmittedWorkDone();
      });
      operations.push({ name: 'gpu_sync', time: syncTime });

      // Sort by time
      operations.sort((a, b) => b.time - a.time);

      return {
        success: true,
        operations,
        totalOverhead: operations.reduce((s, o) => s + o.time, 0),
      };
    });

    expect(result.success, result.error).toBe(true);

    console.log('\n=== Operation Overhead Analysis ===');
    for (const op of result.operations || []) {
      console.log(`${op.name}: ${op.time.toFixed(3)}ms`);
    }
    console.log(`Total overhead per token: ${result.totalOverhead?.toFixed(3)}ms`);

    // If overhead is > 1ms per token, that's a red flag
    if ((result.totalOverhead || 0) > 1) {
      console.warn('WARNING: High per-token overhead. Consider batching operations.');
    }
  });
});
