/**
 * Correctness Test Setup
 * Shared utilities for GPU kernel correctness tests
 */

import { test as base, expect } from '@playwright/test';
import type { Page } from '@playwright/test';

/**
 * WebGPU device and adapter
 */
export interface GPU {
  adapter: GPUAdapter;
  device: GPUDevice;
}

/**
 * Test harness with GPU helpers
 */
export interface TestHarness {
  getGPU(): Promise<GPU>;
  runMatmul(device: GPUDevice, A: Float32Array, B: Float32Array, M: number, N: number, K: number, alpha?: number): Promise<Float32Array>;
  runBatchMatmul(device: GPUDevice, A: Float32Array, B: Float32Array, batch: number, M: number, N: number, K: number): Promise<Float32Array>;
  runMatvec(device: GPUDevice, A: Float32Array, x: Float32Array, M: number, K: number): Promise<Float32Array>;
  runMatmulQ4K(device: GPUDevice, A: Float32Array, B_q4k: Uint8Array, M: number, N: number, K: number, alpha?: number): Promise<Float32Array>;
  runAttention(device: GPUDevice, Q: Float32Array, K: Float32Array, V: Float32Array, seqLen: number, kvLen: number, numHeads: number, numKVHeads: number, headDim: number, mask?: Float32Array): Promise<Float32Array>;
  runRMSNorm(device: GPUDevice, input: Float32Array, weight: Float32Array, batchSize: number, hiddenSize: number): Promise<Float32Array>;
  runSoftmax(device: GPUDevice, input: Float32Array, innerSize: number, outerSize: number, temperature?: number): Promise<Float32Array>;
  runRoPE(device: GPUDevice, input: Float32Array, seqLen: number, numHeads: number, headDim: number, startPos?: number): Promise<Float32Array>;
  runSiLU(device: GPUDevice, input: Float32Array): Promise<Float32Array>;
  runSiLUGated(device: GPUDevice, gate: Float32Array, up: Float32Array): Promise<Float32Array>;
  runGather(device: GPUDevice, embeddings: Float32Array, indices: Uint32Array, vocabSize: number, embedDim: number): Promise<Float32Array>;
  runScatterAdd(device: GPUDevice, expertOutputs: Float32Array, indices: Uint32Array, weights: Float32Array, numTokens: number, hiddenSize: number, numExperts: number, topK: number): Promise<Float32Array>;
  runMoEGather(device: GPUDevice, tokens: Float32Array, expertIndices: Uint32Array, numTokens: number, hiddenSize: number, numExperts: number, topK: number): Promise<{ tokenCounts: Uint32Array }>;
  runSoftmaxTopK(device: GPUDevice, logits: Float32Array, numTokens: number, numExperts: number, topK: number): Promise<{ indices: Uint32Array; weights: Float32Array }>;
  runResidual(device: GPUDevice, x: Float32Array, residual: Float32Array): Promise<Float32Array>;
  runDequantQ4K(device: GPUDevice, quantized: Uint8Array, numBlocks: number): Promise<Float32Array>;
  references: {
    matmulRef(A: Float32Array, B: Float32Array, M: number, N: number, K: number, alpha?: number): Float32Array;
    batchMatmulRef(A: Float32Array, B: Float32Array, batch: number, M: number, N: number, K: number): Float32Array;
    matvecRef(A: Float32Array, x: Float32Array, M: number, K: number): Float32Array;
    attentionRef(Q: Float32Array, K: Float32Array, V: Float32Array, seqLen: number, kvLen: number, numHeads: number, numKVHeads: number, headDim: number, mask?: Float32Array): Float32Array;
    createCausalMask(seqLen: number): Float32Array;
    rmsNormRef(input: Float32Array, weight: Float32Array, batchSize: number, hiddenSize: number): Float32Array;
    softmaxRef(input: Float32Array, innerSize: number, outerSize: number, temperature?: number): Float32Array;
    ropeRef(input: Float32Array, cos: Float32Array, sin: Float32Array, seqLen: number, numHeads: number, headDim: number, startPos: number): Float32Array;
    computeRopeFreqs(headDim: number, seqLen: number): { cos: Float32Array; sin: Float32Array };
    siluRef(input: Float32Array): Float32Array;
    siluGatedRef(gate: Float32Array, up: Float32Array): Float32Array;
    gatherRef(embeddings: Float32Array, indices: Uint32Array, vocabSize: number, embedDim: number): Float32Array;
    scatterAddRef(expertOutputs: Float32Array, indices: Uint32Array, weights: Float32Array, numTokens: number, hiddenSize: number, numExperts: number, topK: number): Float32Array;
    scatterAddAccumulateRef(expertOutputs: Float32Array, indices: Uint32Array, weights: Float32Array, numTokens: number, hiddenSize: number, numExperts: number, topK: number, existing: Float32Array): Float32Array;
    moeGatherRef(tokens: Float32Array, expertIndices: Uint32Array, numTokens: number, hiddenSize: number, numExperts: number, topK: number): { tokenCounts: Uint32Array };
    moeComputeAssignmentsRef(expertIndices: Uint32Array, numTokens: number, numExperts: number, topK: number): { tokenCounts: Uint32Array; totalAssignments: number };
    topkRef(input: Float32Array, numTokens: number, numExperts: number, topK: number, renormalize: boolean): { indices: Uint32Array; weights: Float32Array };
    softmaxTopkRef(logits: Float32Array, numTokens: number, numExperts: number, topK: number, renormalize: boolean): { indices: Uint32Array; weights: Float32Array };
    residualAddRef(x: Float32Array, residual: Float32Array): Float32Array;
    dequantInt8Ref(quantized: Int8Array, scales: Float32Array, zeroPoints: Int8Array | null, numChannels: number, channelSize: number): Float32Array;
    dequantInt4Ref(quantized: Uint8Array, scales: Float32Array, numElements: number, groupSize: number): Float32Array;
    dequantQ4_0Ref(quantized: Uint8Array, numBlocks: number): Float32Array;
    quantizeQ4_KRef(values: Float32Array, numBlocks: number): Uint8Array;
    dequantQ4_KRef(quantized: Uint8Array, numBlocks: number): Float32Array;
  };
}

declare global {
  interface Window {
    testHarness: TestHarness;
  }
}

/**
 * GPU initialization result
 */
interface GPUReadyResult {
  success?: boolean;
  error?: string;
}

/**
 * Extended test fixture with GPU context
 */
export const test = base.extend<{ gpuPage: Page }>({
  /**
   * GPU page with WebGPU initialized
   */
  gpuPage: async ({ page }, use) => {
    // Navigate to test page
    await page.goto('/kernel-tests/browser/index.html');

    // Wait for WebGPU initialization and testHarness to be available
    const ready = await page.evaluate(async (): Promise<GPUReadyResult> => {
      // First check if WebGPU is available
      if (!navigator.gpu) return { error: 'WebGPU not supported' };
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return { error: 'No GPU adapter available' };

      // Wait for testHarness to be initialized (max 10 seconds)
      for (let i = 0; i < 100; i++) {
        if (window.testHarness && window.testHarness.references) {
          // Also wait for GPU to be ready
          try {
            const gpu = await window.testHarness.getGPU();
            if (gpu && gpu.device) {
              return { success: true };
            }
          } catch (e) {
            // GPU not ready yet, continue waiting
          }
        }
        await new Promise(r => setTimeout(r, 100));
      }

      return { error: 'testHarness not initialized after 10s' };
    });

    if (ready.error) {
      test.skip(true, ready.error);
    }

    await use(page);
  },
});

export { expect };

/**
 * Helper to run a kernel test in the browser context
 */
export async function runKernelTest<T>(page: Page, testFn: () => T | Promise<T>): Promise<T> {
  return page.evaluate(testFn);
}

/**
 * Test size configuration
 */
export interface TestSize {
  tokens: number;
  experts: number;
  hidden: number;
}

/**
 * Common test configurations
 */
export const TEST_SIZES: Record<string, TestSize> = {
  small: { tokens: 8, experts: 4, hidden: 64 },
  medium: { tokens: 64, experts: 8, hidden: 256 },
  large: { tokens: 256, experts: 16, hidden: 512 },
};

/**
 * Generate random float32 array
 */
export function randomFloat32(size: number, min: number = -1, max: number = 1): Float32Array {
  const arr = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    arr[i] = Math.random() * (max - min) + min;
  }
  return arr;
}

/**
 * Generate random uint32 array
 */
export function randomUint32(size: number, max: number): Uint32Array {
  const arr = new Uint32Array(size);
  for (let i = 0; i < size; i++) {
    arr[i] = Math.floor(Math.random() * max);
  }
  return arr;
}
