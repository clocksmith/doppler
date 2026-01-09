/**
 * Correctness Test Setup
 * Shared utilities for GPU kernel correctness tests
 */

import { test as base, expect } from '@playwright/test';

/**
 * Extended test fixture with GPU context
 */
export const test = base.extend({
  /**
   * GPU page with WebGPU initialized
   */
  gpuPage: async ({ page }, use) => {
    // Navigate to test page
    await page.goto('/tests/kernels/browser/index.html');

    // Wait for WebGPU initialization and testHarness to be available
    const ready = await page.evaluate(async () => {
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
export async function runKernelTest(page, testFn) {
  return page.evaluate(testFn);
}

/**
 * Common test configurations
 */
export const TEST_SIZES = {
  small: { tokens: 8, experts: 4, hidden: 64 },
  medium: { tokens: 64, experts: 8, hidden: 256 },
  large: { tokens: 256, experts: 16, hidden: 512 },
};

/**
 * Generate random float32 array
 */
export function randomFloat32(size, min = -1, max = 1) {
  const arr = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    arr[i] = Math.random() * (max - min) + min;
  }
  return arr;
}

/**
 * Generate random uint32 array
 */
export function randomUint32(size, max) {
  const arr = new Uint32Array(size);
  for (let i = 0; i < size; i++) {
    arr[i] = Math.floor(Math.random() * max);
  }
  return arr;
}
